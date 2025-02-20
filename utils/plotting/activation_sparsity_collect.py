import os

import torch
from tqdm import tqdm

from models.tiny_vit.tiny_vit_constants import TINY_VIT_5M_ACTIVATIONS_OF_INTEREST


def main():
    for el in TINY_VIT_5M_ACTIVATIONS_OF_INTEREST["output"]:
        final_tensor = torch.Tensor()
        for i in tqdm(range(4096)):
            tensor = torch.load(f"maps/output_{el}{i}.pt")
            final_tensor = torch.cat((final_tensor, tensor.flatten()))
            os.remove(f"maps/output_{el}{i}.pt")

        torch.save(final_tensor, f"maps/output_{el}.pt")


if __name__ == "__main__":
    main()
