import argparse
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import build_dataset
from models.model_utils import build_model
from models.original_wanda_vit.wanda_vit_constants import VIT_B_ACTIVATIONS_OF_INTEREST
from models.tiny_vit.tiny_vit_constants import TINY_VIT_5M_ACTIVATIONS_OF_INTEREST
from models.tiny_vit.tiny_vit_factory import tiny_vit_5m
from utils.arg_parser import get_args

input_hists = dict()
output_hists = dict()


def store_input_hist_hook_fn(module, inputs, _):
    # Store module name
    if module not in input_hists:
        input_hists[module] = None

    # Obtain inputs
    if isinstance(inputs, tuple):
        to_write = inputs[0].detach().cpu()
    else:
        to_write = inputs.detach().cpu()

    # Store histogram
    if input_hists[module] is None:
        hist = torch.Tensor().cpu()
        edges = torch.Tensor().cpu()

        torch.histogram(
            to_write.cpu(),
            bins=100,
            out=(hist, edges),
        )
        input_hists[module] = (hist, edges)
    else:
        edges = input_hists[module][1]
        hist = torch.histogram(to_write.cpu(), bins=edges)
        input_hists[module] = (
            input_hists[module][0] + hist.hist,
            input_hists[module][1],
        )


def store_output_hist_hook_fn(module, _, outputs):
    # Store module name
    if module not in output_hists:
        output_hists[module] = None

    # Obtain outputs
    if isinstance(outputs, tuple):
        to_write = outputs[0].detach().cpu()
    else:
        to_write = outputs.detach().cpu()

    # Store histogram
    if output_hists[module] is None:
        hist = torch.Tensor().cpu()
        edges = torch.Tensor().cpu()

        torch.histogram(
            to_write.cpu(),
            bins=100,
            out=(hist, edges),
        )
        output_hists[module] = (hist, edges)
    else:
        edges = output_hists[module][1]
        hist = torch.histogram(to_write.cpu(), bins=edges)
        output_hists[module] = (
            output_hists[module][0] + hist.hist,
            output_hists[module][1],
        )


def main():
    # Parse arguments
    args = get_args()

    # Select the last GPU if available
    device = torch.device(
        "cuda:" + str(torch.cuda.device_count() - 1)
        if torch.cuda.is_available()
        else "cpu"
    )

    # Fix the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    print("Preparing model...")
    if args.model.lower() == "tiny_vit_5m":
        model = tiny_vit_5m()
        activations_of_interest = TINY_VIT_5M_ACTIVATIONS_OF_INTEREST
    elif args.model.lower() == "vit_base_patch16_224":
        model = build_model(
            args=argparse.Namespace(
                model="vit_base_patch16_224",
                nb_classes=1000,
                drop_path=0.0,
                dropout=0.0,
            )
        )
        activations_of_interest = VIT_B_ACTIVATIONS_OF_INTEREST
    else:
        raise ValueError(f"Model {args.model} not supported.")
    model.to(device)

    # Load model weights
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    # Register model's forward hooks for activation sparsity analysis
    for name, module in model.named_modules():
        if name in activations_of_interest["input"]:
            module.register_forward_hook(store_input_hist_hook_fn)

        if name in activations_of_interest["output"]:
            module.register_forward_hook(store_output_hist_hook_fn)

    # Prepare validation dataset (used for evaluation)
    print("Preparing data loader...")
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Iterate through samples
    checked = False
    it = 0
    for batch in tqdm(data_loader_val, desc="Analyzing activation sparsity"):
        it += 1
        if it == 50:
            break
        # Get batch and put on device
        images = batch[0]
        images = images.to(device, non_blocking=True)

        # Perform forward pass
        _ = model(images)

        # Check if all activations have been stored
        if not checked:
            assert len(input_hists) == len(
                activations_of_interest["input"]
            ), "Input information improperly stored."
            assert len(output_hists) == len(
                activations_of_interest["output"]
            ), "Output information improperly stored."

            checked = True

    # Plot and save histograms
    for module, hist in input_hists.items():
        # Get histogram values
        hist, bins = hist
        hist = hist.numpy()
        bins = bins.numpy()

        # Prepare bins
        bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        # Plot histogram
        plt.clf()
        plt.bar(x=bins, height=hist, width=0.02)
        plt.show()

    for module, hist in output_hists.items():
        # Get histogram values
        hist, bins = hist
        hist = hist.numpy()
        bins = bins.numpy()

        # Prepare bins
        bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        # Plot histogram
        plt.clf()
        plt.bar(x=bins, height=hist, width=0.02)
        plt.title(str(type(module)) + " OUTPUT")
        plt.show()

    return None


if __name__ == "__main__":
    main()
