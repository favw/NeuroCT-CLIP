from transformers import BertModel, BertTokenizer


DEFAULT_TEXT_MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
HEAD_TEXT_MODEL_NAME = "GerMedBERT/medbert-512"


def build_german_head_prompt_pair(pathology: str):
    """Return grammatically valid positive/negative prompts for head-CT labels."""
    pathology = str(pathology).strip()
    if pathology.casefold() == "nichts":
        return [
            "Die kraniale CT zeigt keinen pathologischen Befund.",
            "Die kraniale CT zeigt mindestens einen pathologischen Befund.",
        ]

    return [
        f"In der kranialen CT ist folgender Befund vorhanden: {pathology}.",
        f"In der kranialen CT ist folgender Befund nicht vorhanden: {pathology}.",
    ]


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
