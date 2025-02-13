import json

import torch
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch import optim as optim

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
):
    # Prepare group dicts
    parameter_group_names = {}
    parameter_group_vars = {}

    # Iterate through model parameters
    for name, param in model.named_parameters():
        # Skip parameters with no gradients
        if not param.requires_grad:
            continue

        # Set weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Get layer id
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            # Get layer scale
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            # Prepare parameter groups
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        # Append parameter to group
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))

    return list(parameter_group_vars.values())


def create_optimizer(
    args,
    model,
    get_num_layer=None,
    get_layer_scale=None,
    filter_bias_and_bn=True,
    skip_list=None,
):
    # Get optimizer arguments
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    # Filter bias and batch norm if needed
    if filter_bias_and_bn:
        skip = {}

        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale
        )

        weight_decay = 0.0
    else:
        parameters = model.parameters()

    if "fused" in opt_lower:
        assert (
            has_apex and torch.cuda.is_available()
        ), "APEX and CUDA required for fused optimizers"

    # Create optimizer according to passed arguments
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)

    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps

    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    # SGD with Nesterov
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args
        )

    # SGD without Nesterov
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args
        )

    # Adam
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)

    # AdamW
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)

    # AdamP
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)

    # SGDP
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)

    # Adadelta
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)

    # Adafactor
    elif opt_lower == "adafactor":
        if not args.lr:
            opt_args["lr"] = None

        optimizer = Adafactor(parameters, **opt_args)

    # Adahessian
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)

    # RMSprop
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=args.momentum, **opt_args
        )

    # RMSpropTF
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)

    # NvNovoGrad
    elif opt_lower == "nvnovograd":
        optimizer = NvNovoGrad(parameters, **opt_args)

    # FusedSGD
    elif opt_lower == "fusedsgd":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args
        )

    # FusedMomentum
    elif opt_lower == "fusedmomentum":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args
        )

    # FusedAdam
    elif opt_lower == "fusedadam":
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)

    # FusedAdamW
    elif opt_lower == "fusedadamw":
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)

    # FusedLAMB
    elif opt_lower == "fusedlamb":
        optimizer = FusedLAMB(parameters, **opt_args)

    # FusedNovoGrad
    elif opt_lower == "fusednovograd":
        opt_args.setdefault("betas", (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    # Exception case
    else:
        raise ValueError("Unknown optimizer: %s" % args.opt)

    # Add Lookahead if needed
    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer
