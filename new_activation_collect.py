import glob

import torch


def main():
    for block in range(12):
        final_tensors = [torch.Tensor() for _ in range(50)]

        for obj in range(1, 51):
            paths = glob.glob(f"activations/{block}/transformer_block_{obj}_*.pth")

            for path in paths:
                tensor = torch.load(path)
                final_tensors[obj - 1] = torch.cat((final_tensors[obj - 1], tensor.flatten()))

        for obj in range(50):
            torch.save(final_tensors[obj], f"activations/{block}/transformer_block_{obj + 1}.pth")


if __name__ == "__main__":
    main()
