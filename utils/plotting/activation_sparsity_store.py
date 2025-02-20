import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from datasets import build_dataset
from models.model_utils import build_model
from models.original_wanda_vit.wanda_vit_constants import VIT_B_ACTIVATIONS_OF_INTEREST
from models.tiny_vit.tiny_vit_constants import TINY_VIT_5M_ACTIVATIONS_OF_INTEREST
from models.tiny_vit.tiny_vit_factory import tiny_vit_5m
from utils.arg_parser import get_args

input_hists = dict()
output_hists = dict()
file_n = [
    0,
]
current_output = [
    0,
]


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

    # Save tensor
    torch.save(
        to_write,
        "maps/output_"
        + str(TINY_VIT_5M_ACTIVATIONS_OF_INTEREST["output"][current_output[0]])
        + str(file_n[0])
        + ".pt",
    )
    file_n[0] += 1


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
    if args.model.lower() == "vit_base_patch16_224":
        model = build_model(
            args=argparse.Namespace(
                model="vit_base_patch16_224",
                nb_classes=1000,
                drop_path=0.0,
                dropout=0.0,
            )
        )
        activations_of_interest = VIT_B_ACTIVATIONS_OF_INTEREST
    elif "tiny_vit" in args.model.lower():
        if "5m" in args.model.lower():
            model = tiny_vit_5m()
            activations_of_interest = TINY_VIT_5M_ACTIVATIONS_OF_INTEREST
        else:
            raise ValueError(f"Model {args.model} not supported.")
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

    # Prepare "calibration" dataset
    print("Preparing datasets...")
    if args.train_data_loader_path is None:
        # Prepare saving directory
        if not os.path.exists("other_resources"):
            os.mkdir("other_resources")

        # Build dataset
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

        # Save dataset if required
        if args.save_train_data_loader:
            torch.save(dataset_train, "other_resources/dataset_train.loader")

        # Randomly select calibration data ids
        calibration_ids = np.random.choice(len(dataset_train), args.n_samples)

        # Save calibration ids
        np.save("other_resources/calib_ids", calibration_ids)
    else:
        # Load dataset
        dataset_train = torch.load(args.train_data_loader_path, weights_only=False)

        # Update root path in dataset object
        old_root = dataset_train.root[:-5]
        dataset_train.root = args.data_path

        for i in range(len(dataset_train.samples)):
            dataset_train.samples[i] = (
                dataset_train.samples[i][0].replace(old_root, dataset_train.root),
                dataset_train.samples[i][1],
            )

            dataset_train.imgs[i] = (
                dataset_train.imgs[i][0].replace(old_root, dataset_train.root),
                dataset_train.imgs[i][1],
            )

        # Load calibration ids
        calibration_ids = np.load(args.calibration_ids_path)

    # Select calibration data
    calib_data = []
    for i in tqdm(calibration_ids, desc="Preparing calibration data"):
        calib_data.append(dataset_train[i][0].unsqueeze(dim=0))

    # Prepare output dir
    if not os.path.exists("maps"):
        os.mkdir("maps")

    # Register model's forward hooks for activation sparsity analysis
    for name, module in model.named_modules():
        if name in activations_of_interest["output"]:
            the_hook = module.register_forward_hook(store_output_hist_hook_fn)

            # Iterate through samples
            for batch in tqdm(calib_data, desc="Analyzing activation sparsity"):
                # Get batch and put on device
                images = batch
                images = images.to(device, non_blocking=True)

                # Perform forward pass
                _ = model(images)

            # Remove forward hook
            the_hook.remove()
            current_output[0] += 1

    return None


if __name__ == "__main__":
    main()
