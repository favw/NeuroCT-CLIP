from pathlib import Path
from eval import evaluate_internal
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_inference_nii import CTReportDatasetinfer, HeadCTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from ct_clip import CTCLIP
from head_utils import load_label_columns
from text_model_utils import build_tokenizer


# helpers
def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        meta_file: "meta_data.csv",
        results_folder = './results',
        labels = "labels.csv",
        head: bool = False,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = build_tokenizer(head=head)
        self.results_folder = results_folder
        self.pathologies = load_label_columns(labels)
        self.register_buffer('steps', torch.Tensor([0]))

        # Load the pre-trained weights
        dataset_cls = HeadCTReportDatasetinfer if head else CTReportDatasetinfer
        self.ds = dataset_cls(data_folder=data_folder, reports_file=reports_file, meta_file=meta_file, labels=labels)

        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=6,
            batch_size=1,
            shuffle = True,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
        )

        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def infer(self, log_fn=noop):
        device = self.device

        steps = int(self.steps.item())

        # logs
        logs = {}

        with torch.no_grad():

            models_to_evaluate = ((self.CTClip, str(steps)),)

            for model, filename in models_to_evaluate:
                model.eval()
                predictedall=[]
                realall=[]

                accession_names=[]
                for i in tqdm.tqdm(range(len(self.ds))):
                    valid_data, text, onehotlabels, acc_name = next(self.dl_iter)

                    plotdir = self.result_folder_txt
                    Path(plotdir).mkdir(parents=True, exist_ok=True)

                    predictedlabels=[]

                    for pathology in self.pathologies:
                        text = [f"{pathology} is present.", f"{pathology} is not present."]
                        text_tokens=self.tokenizer(
                                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                        output = model(text_tokens, valid_data.cuda(),  device=device)

                        output = apply_softmax(output)

                        append_out=output.detach().cpu().numpy()
                        predictedlabels.append(append_out[0])

                    predictedall.append(predictedlabels)
                    realall.append(onehotlabels.detach().cpu().numpy()[0])
                    accession_names.append(acc_name[0])

                realall=np.array(realall)
                predictedall=np.array(predictedall)

                np.savez(self.results_folder / "labels_weights.npz", data=realall)
                np.savez(self.results_folder / "predicted_weights.npz", data=predictedall)
                with open(self.results_folder / "accessions.txt", "w") as file:
                    for item in accession_names:
                        file.write(item + "\n")

                plotdir = f"{self.results_folder}/"
                dfs=evaluate_internal(predictedall,realall,self.pathologies, plotdir)

                writer = pd.ExcelWriter(self.results_folder / "aurocs.xlsx", engine='xlsxwriter')

                dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                writer.close()

                threshold = 0.5
                predicted_binary = (predictedall >= threshold).astype(int)
                report = classification_report(
                    realall,
                    predicted_binary,
                    target_names=self.pathologies,
                    digits=4,
                    zero_division=0,
                )
                report_dict = classification_report(
                    realall,
                    predicted_binary,
                    target_names=self.pathologies,
                    output_dict=True,
                    zero_division=0,
                )
                summary = (
                    f"Decision threshold: {threshold:.2f}\n"
                    f"Exact-match accuracy: {accuracy_score(realall, predicted_binary):.4f}\n"
                    f"Hamming loss: {hamming_loss(realall, predicted_binary):.4f}\n\n"
                    f"{report}"
                )

                self.print("\nScikit-learn classification metrics\n" + summary)
                (self.results_folder / "classification_report.txt").write_text(
                    summary,
                    encoding="utf-8",
                )
                pd.DataFrame(report_dict).transpose().to_csv(
                    self.results_folder / "classification_metrics.csv"
                )

        self.steps += 1

        log_fn(logs)

        self.print('Inference complete')
