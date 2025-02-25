import torch
import numpy as np

from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

DEFAULT_T5_NAME = "base1.1"
T5_VERSIONS = {
    "small1.1": {"handle": "google/t5-v1_1-small", "dim": 512, "size": .3},
    "base1.1": {"handle": "google/t5-v1_1-base", "dim": 768, "size": .99},
    "large1.1": {"handle": "google/t5-v1_1-large", "dim": 1024, "size": 3.13}
}

TOKENIZER = None
MODEL = None
DEVICE = None

# Fast tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
def initialize_t5(idx: int, name=DEFAULT_T5_NAME):
    global TOKENIZER, MODEL, DEVICE
    
    DEVICE = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    if TOKENIZER is None:
        TOKENIZER = T5Tokenizer.from_pretrained(T5_VERSIONS[name]["handle"])
    
    if MODEL is None:
        MODEL = T5EncoderModel.from_pretrained(T5_VERSIONS[name]["handle"]).to(DEVICE)

def find_optimal_max_length(text_list, batch_size=16, percentile=95):
    global TOKENIZER
    
    lengths = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        tokenized = TOKENIZER(batch, padding=False, truncation=False)
        lengths.extend([len(ids) for ids in tokenized["input_ids"]])
        
    max_length = int(np.percentile(lengths, percentile))
    
    return max_length

def compute_global_max_length(df_train, df_valid = None, df_test = None, batch_size=16, percentile=95):
    all_texts = list(df_train["prompt"].unique())
    
    if df_valid is not None:
        all_texts += list(df_valid["prompt"].unique())
        
        if df_test is not None:
            all_texts += list(df_test["prompt"].unique())
    
    max_length = find_optimal_max_length(text_list=all_texts, batch_size=batch_size, percentile=percentile)
    print(f"Global max_length determined: {max_length}")
    
    return max_length

def t5_encode_text(text_list, max_length=None):
    """
    Encodes a sequence of text with a T5 text encoder.

    :param text: List of text to encode.
    :param name: Name of T5 model to use. Options are:

        - :code:`'small1.1'` (~0.3 GB, 512 encoding dim),

        - :code:`'base1.1'` (~0.99 GB, 768 encoding dim),

        - :code:`'large1.1'` (~3.13 GB, 1024 encoding dim),

    :return: Returns encodings and attention mask. Element **[i,j,k]** of the final encoding corresponds to the **k**-th
        encoding component of the **j**-th token in the **i**-th input list element.
    """
    global TOKENIZER, MODEL, DEVICE
    
    if TOKENIZER is None or MODEL is None:
        raise ValueError("Tokenizer is not initialized yet. Call initialize_t5() first.")
    
    if max_length is None:
        max_length = find_optimal_max_length(text_list=text_list)
    
    tokenized = TOKENIZER.batch_encode_plus(
        text_list,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = tokenized.input_ids.to(DEVICE)
    attention_mask = tokenized.attention_mask.to(DEVICE)
    
    MODEL.eval()
    with torch.no_grad():
        t5_output = MODEL(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()
        
    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)
    
    return final_encoding, attention_mask.bool()