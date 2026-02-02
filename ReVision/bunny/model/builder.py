import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import transformers
# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

import sys

# 把 /data/xmyu/Bunny_text/ 加进 sys.path，以便后续 import
sys.path.insert(0, "/data/xmyu/Bunny_text")
from bunny.model.language_model.bunny_llama import BunnyLlamaConfig, BunnyLlamaForCausalLM


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):

    # Our Model
    # model = AutoModelForCausalLM.from_pretrained(
    #     '/data/xmyu/finished-checkpoints/no-transfer/checkpoints-llama3-8b/bunny-llama3-8b',
    #     torch_dtype=torch.float16, # float32 for cpu
    #     trust_remote_code=True
    #     # device_map='auto'
    # ).to("cuda")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     '/data/xmyu/finished-checkpoints/no-transfer/checkpoints-llama3-8b/bunny-llama3-8b',
    #     trust_remote_code=True
    #     )


    # Our Model
    model = AutoModelForCausalLM.from_pretrained(
        '/data/xmyu/finished-checkpoints/mean_shift/checkpoints-llama3-8b/bunny-llama3-8b',
        torch_dtype=torch.float16, # float32 for cpu
        trust_remote_code=True
        # device_map='auto'
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        '/data/xmyu/finished-checkpoints/mean_shift/checkpoints-llama3-8b/bunny-llama3-8b',
        trust_remote_code=True
        )


    return tokenizer, model, 512
