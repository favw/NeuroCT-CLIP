from transformers import BertModel, BertTokenizer


DEFAULT_TEXT_MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
HEAD_TEXT_MODEL_NAME = "placeholder"


def get_text_model_name(head: bool = False) -> str:
    return HEAD_TEXT_MODEL_NAME if head else DEFAULT_TEXT_MODEL_NAME


def build_tokenizer(head: bool = False):
    model_name = get_text_model_name(head)
    if head:
        return BertTokenizer.from_pretrained(model_name)
    return BertTokenizer.from_pretrained(model_name, do_lower_case=True)


def build_text_encoder(head: bool = False, tokenizer=None):
    text_encoder = BertModel.from_pretrained(get_text_model_name(head))
    if tokenizer is not None and not head:
        text_encoder.resize_token_embeddings(len(tokenizer))
    return text_encoder
