import argparse


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    Returns the command line arguments for the wanda for image classification pipeline.
    """

    parser = argparse.ArgumentParser(
        "Wanda applied on vit-based models.",
    )

    # Training and evaluation parameters
    parser.add_argument("--batch_size", default=1, type=int, help="Per GPU batch size")
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation steps"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="image input size")

    # Dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        metavar="PCT",
        help="Drop path rate (default: 0.0)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0,
        metavar="PCT",
        help="Drop path rate (default: 0.0)",
    )

    # EMA-related parameters
    parser.add_argument("--model_ema", type=str2bool, default=False)

    # Optimization parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        metavar="LR",
        help="learning rate (default: 4e-3), with total batch size 4096",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        type=str2bool,
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0.8, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=1.0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/mnt/d/datasets/ImageNet",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_data_path", default=None, type=str, help="dataset path for evaluation"
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument("--imagenet_default_mean_and_std", type=str2bool, default=True)
    parser.add_argument(
        "--data_set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "image_folder"],
        type=str,
        help="ImageNet dataset path",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--checkpoint",
        "--resume",
        default="",
        help="Path to checkpoint to resume from.",
    )

    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Use PyTorch's AMP (Automatic Mixed Precision) or not",
    )

    # Arguments for pruning
    parser.add_argument("--n_samples", type=int, default=4096)

    parser.add_argument(
        "--prune_metrics",
        type=lambda arg: list(map(str, arg.split(","))),
        help="Choose the pruning metrics, between weight magnitude and wanda.",
    )

    parser.add_argument(
        "--prune_granularities",
        type=lambda arg: list(map(str, arg.split(","))),
        help="Choose the pruning structural granularities, between layer and row.",
    )

    parser.add_argument(
        "--sparsities", type=lambda arg: list(map(float, arg.split(","))), default=0.0
    )

    # Arguments for reloading training data loader and calibration set
    parser.add_argument("--train_data_loader_path", type=str, default=None)
    parser.add_argument("--save_train_data_loader", action="store_true", default=False)
    parser.add_argument("--calibration_ids_path", type=str, default=None)

    return parser.parse_args()
