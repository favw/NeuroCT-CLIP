"""
Detached Head-CT masked-language-modeling scaffold.

This module is intentionally not wired into CT-CLIP training or inference yet.
"""

from pathlib import Path

from datasets import Dataset
from transformers import (
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from text_model_utils import HEAD_TEXT_MODEL_NAME, build_tokenizer


class HeadCTMLMBridge:
    def __init__(self, tokenizer, mlm=None):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def attach_mlm(self, mlm):
        # bridge instance for chaining
        self.mlm = mlm
        return self

    def tokenize(self, text, **kwargs):
        # returns BatchEncoding for a given input 
        return self.tokenizer(text, **kwargs)


def build_head_mlm_tokenizer():
    # returns BertTokenizer
    return build_tokenizer(head=True)


def get_head_mlm_prepare_dir():
    # canonical output root for the prepared Head-CT text model
    return Path(__file__).resolve().parents[2] / "head_mlm_training" / "outputs" / "prepared_head_text_model"


def get_head_mlm_mlm_dir():
    # directory for the full MLM checkpoint
    return get_head_mlm_prepare_dir() / "mlm"


def get_head_mlm_encoder_dir():
    # directory for the extracted plain BERT encoder
    return get_head_mlm_prepare_dir() / "encoder"


# returns BertForMaskedLM (MLM model from given BERT(-like) model) for finetuning
def build_head_mlm_model():
    return BertForMaskedLM.from_pretrained(HEAD_TEXT_MODEL_NAME)

# collator for masking 
def build_head_mlm_data_collator(tokenizer, mlm_probability=0.15):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

# helper for file parsing
def read_text_dir(text_dir):
        text_files = sorted(text_dir.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(
                f"No .txt reports found in {text_dir}. Add raw report text files manually."
            )
        return [text_file.read_text(encoding="utf-8").strip() for text_file in text_files]

# function loads raw head-ct reports as .txt from ./head_mlm_training/training and ./head_mlm_training/validation
def load_head_mlm_texts():
    base_dir = Path(__file__).resolve().parents[2] / "head_mlm_training"
    train_dir = base_dir / "training"
    eval_dir = base_dir / "validation"

    train_texts = read_text_dir(train_dir)
    eval_texts = read_text_dir(eval_dir)
    return train_texts, eval_texts

# returns Dataset after tokenisation 
def tokenize_head_mlm_texts(texts, tokenizer, max_length=512):
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=["text"],
    )

# returns tuple of Dataset, (training, evaluation)
def build_head_mlm_datasets(tokenizer, max_length=512):
    train_texts, eval_texts = load_head_mlm_texts()
    train_dataset = tokenize_head_mlm_texts(train_texts, tokenizer, max_length=max_length)
    eval_dataset = tokenize_head_mlm_texts(eval_texts, tokenizer, max_length=max_length)
    return train_dataset, eval_dataset


def build_head_mlm_training_args(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )


def is_head_text_model_prepared():
    # prepared means both the MLM model and the encoder export exist as HF directories
    mlm_config = get_head_mlm_mlm_dir() / "config.json"
    encoder_config = get_head_mlm_encoder_dir() / "config.json"
    return mlm_config.exists() and encoder_config.exists()


# build trainer 
def build_head_mlm_trainer(output_dir=None, mlm_probability=0.15, max_length=512):
    if output_dir is None:
        output_dir = get_head_mlm_mlm_dir()

    tokenizer = build_head_mlm_tokenizer()
    model = build_head_mlm_model()
    data_collator = build_head_mlm_data_collator(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )
    train_dataset, eval_dataset = build_head_mlm_datasets(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    training_args = build_head_mlm_training_args(output_dir)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


def train_and_export_head_text_model(output_dir=None, mlm_probability=0.15, max_length=512):
    # fine-tune the MLM model and export both the full MLM checkpoint and the plain encoder
    if output_dir is None:
        output_dir = get_head_mlm_mlm_dir()

    trainer = build_head_mlm_trainer(
        output_dir=output_dir,
        mlm_probability=mlm_probability,
        max_length=max_length,
    )
    trainer.train()

    tokenizer = trainer.tokenizer
    mlm_dir = get_head_mlm_mlm_dir()
    encoder_dir = get_head_mlm_encoder_dir()

    mlm_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(mlm_dir))
    tokenizer.save_pretrained(str(mlm_dir))

    trainer.model.bert.save_pretrained(str(encoder_dir))
    tokenizer.save_pretrained(str(encoder_dir))

    return encoder_dir


def ensure_head_text_model_prepared():
    # local prepared artifact gate for Head-CT text-model setup
    if is_head_text_model_prepared():
        return get_head_mlm_encoder_dir()

    return train_and_export_head_text_model()
