import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from datasets import build_dataset
from evaluation.evaluation_pipeline import evaluate
from models.model_utils import build_model
from utils.arg_parser import get_args
from wanda_pruning.pruning_essentials import prune_vit, check_sparsity


def main():
    # Parse command line arguments
    args = get_args()

    # At most one of dropout and stochastic depth should be enabled.
    assert args.dropout == 0 or args.drop_path == 0

    # Set up device
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up cudnn.benchmark
    cudnn.benchmark = True

    # Prepare train dataset (used for calibration sampling)
    print("Preparing datasets...")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    # Prepare validation dataset (used for evaluation)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Set up validation data loader
    print("Preparing data loader...")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Prepare model
    print("Preparing model...")
    model = build_model(args, pretrained=False)
    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model parameters:", n_parameters, "\n")

    # Load model weights
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint)

    # Randomly select calibration data
    calibration_ids = np.random.choice(len(dataset_train), args.n_samples)
    calib_data = []

    for i in tqdm(calibration_ids, desc="Preparing calibration data"):
        calib_data.append(dataset_train[i][0].unsqueeze(dim=0))

    calib_data = torch.cat(calib_data, dim=0).to(device)
    print()

    # Prune model
    if args.sparsity != 0:
        with torch.no_grad():
            prune_vit(args, model, calib_data)

    # Check sparsity
    print("\nChecking sparsity...")
    actual_sparsity = check_sparsity(model)
    print(f"Actual sparsity: {actual_sparsity}.\n")

    # Evaluate model
    test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    print(
        f"\nAccuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%"
    )

    return None


if __name__ == "__main__":
    main()
