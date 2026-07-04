from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from CTCLIPTrainer import CTClipTrainer
from head_mlm import ensure_head_text_model_prepared, get_head_mlm_prepare_dir
from logging_utils import timestamped_message
from src.args import parse_arguments
from text_model_utils import build_text_encoder, build_tokenizer


def main():
    args = parse_arguments()

    #TODO: allow for MLM finetuning of BERT models / compare performance 
    #if args.head:
    #    ensure_head_text_model_prepared()
    #    print(f"Head-CT MLM preparation verified at {get_head_mlm_prepare_dir()}.")

    channels = 4 if args.head else 1

    train_data_folder = args.train_data_folder or args.data_folder
    valid_data_folder = args.valid_data_folder or args.data_folder
    train_reports_file = args.train_reports_file or args.reports_file
    valid_reports_file = args.valid_reports_file or args.reports_file
    train_meta_file = args.train_meta_file or args.meta_file
    valid_meta_file = args.valid_meta_file or args.meta_file
    results_folder = args.results_folder or args.save

    required_args = {
        "train_data_folder": train_data_folder,
        "valid_data_folder": valid_data_folder,
        "train_reports_file": train_reports_file,
        "valid_reports_file": valid_reports_file,
        "labels": args.labels,
        "results_folder": results_folder,
    }
    if not args.head:
        required_args.update(
            {
                "train_meta_file": train_meta_file,
                "valid_meta_file": valid_meta_file,
            }
        )
    missing_args = [name for name, value in required_args.items() if value is None]
    if missing_args:
        raise ValueError(f"Missing required training arguments: {', '.join(missing_args)}")

    tokenizer = build_tokenizer(head=args.head)
    text_encoder = build_text_encoder(head=args.head, tokenizer=tokenizer)

    print(timestamped_message("---------"))
    print(timestamped_message(f"tokenizer.pad_token_id={tokenizer.pad_token_id}"))
    print(timestamped_message(f"tokenizer.mask_token_id={tokenizer.mask_token_id}"))
    print(timestamped_message("-----------"))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        channels=channels,
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_text=text_encoder.config.hidden_size,
        dim_image=294912,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
        tokenizer=tokenizer,
    )
    trainer = CTClipTrainer(
        clip,
        reports_file_train=train_reports_file,
        reports_file_valid=valid_reports_file,
        data_train=train_data_folder,
        data_valid=valid_data_folder,
        train_meta_file=train_meta_file,
        valid_meta_file=valid_meta_file,
        labels=args.labels,
        batch_size=args.batch_size,
        results_folder=results_folder,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        head=args.head,
    )

    trainer.train()


if __name__ == "__main__":
    main()
