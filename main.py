from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from datasets import build_dataset
from engine import evaluate
from prune_utils import prune_convnext, prune_vit, check_sparsity
from utils import utils
from utils.arg_parser import get_args


def main():
    # Parse command line arguments
    args = get_args()

    # Check arguments
    # At most one of dropout and stochastic depth should be enabled.
    assert args.dropout == 0 or args.drop_path == 0
    # ConvNeXt does not support dropout.
    assert args.dropout == 0 if args.model.startswith("convnext") else True

    # Create output directory if necessary
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set up distributed training
    utils.init_distributed_mode(args)

    # Set up device
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up cudnn.benchmark
    cudnn.benchmark = True

    # Prepare train dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    # Prepare validation dataset
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    # Get multi-device parameters
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # Set up data sampler
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Activate wandb logging
    if global_rank == 0 and args.enable_wandb:
        utils.WandbLogger(args)

    # Set up validation data loader
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    # Prepare model
    model = utils.build_model(args, pretrained=False)
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Load model
    if "convnext" in args.model:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    elif "vit" in args.model:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.resume, map_location="cpu")
        print("Checkpoint loaded. Loading state dict...")
        model.load_state_dict(checkpoint)
        print("State dict loaded.")
    elif "deit" in args.model:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # Select calibration data
    calibration_ids = np.random.choice(len(dataset_train), args.nsamples)
    calib_data = []

    print("Selecting calibration data...")
    for i in tqdm(calibration_ids):
        calib_data.append(dataset_train[i][0].unsqueeze(dim=0))
    calib_data = torch.cat(calib_data, dim=0).to(device)

    # Prune model
    if args.sparsity != 0:
        with torch.no_grad():
            if "convnext" in args.model:
                prune_convnext(args, model, calib_data)
            elif "vit" in args.model:
                print("Pruning...")
                prune_vit(args, model, calib_data)
                print("Pruned.")
            elif "deit" in args.model:
                prune_vit(args, model, calib_data)

    # Check sparsity
    actual_sparsity = check_sparsity(model)
    print(f"actual sparsity {actual_sparsity}")

    # Evaluate model
    print(f"Eval only mode")
    test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    print(
        f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%"
    )

    return None


if __name__ == "__main__":
    main()
