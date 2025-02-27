import os

from models.Llama.Llama import Llama


def llama_3_2_1b(root, **kwargs):
    # Default case
    if len(kwargs) == 0:
        return Llama.build(
            ckpt_path=os.path.join(root, "consolidated.00.pth"),
            params_path=os.path.join(root, "params.json"),
            tokenizer_path=os.path.join(root, "tokenizer.model"),
            max_seq_len=8192,
            max_batch_size=8,
            seed=42,
        )
