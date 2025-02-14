import torch
import torch.nn as nn
from tqdm import tqdm

from models.tiny_vit.tiny_vit import TinyViT
from wanda_pruning.WrappedLayer import WrappedLayer


def find_layers(module, layers=None, name=""):
    """
    Function to find layers in a module that are of one of the required types.

    Args:
        module (nn.Module): Module to search for layers in.
        layers (list[nn.Module], optional): List of layer types to search for. Defaults to None.
        name (str, optional): Name of the module. Defaults to "".
    Returns:
        dict[str, nn.Module]: Dictionary where the keys are the names of the layers
                              and the values are the layers themselves.
    """

    # If none specified, default to nn.Linear
    if layers is None:
        layers = [nn.Linear]

    # If selected module is among selected ones, return it
    if type(module) in layers:
        return {name: module}

    # Recursively search for layers that fit the requirements
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                module=child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )

    return res


def check_sparsity(model):
    """
    Function to check the actual sparsity level achieved in a model.

    Args:
        model (nn.Module): Model to check sparsity for.
    Returns:
        float: Sparsity level actually achieved in the model.
    """

    # Find linear layers in the model
    subset = find_layers(model, layers=[nn.Linear])

    # Initialize counters
    zero_cnt = 0
    fc_params = 0

    # Iterate through the linear layers and count zero weights
    for name in subset:
        # Get actual layer weight
        w = subset[name].weight.data

        # Skip the last fc layer
        if w.shape[0] == 1000:
            continue

        # Add number of zeroed weights
        zero_cnt += (w == 0).sum().item()

        # Add total number of weights
        fc_params += w.numel()

    return float(zero_cnt) / fc_params


def compute_mask(w_metric, prune_granularity, sparsity):
    """
    Function to compute a pruning mask based on a given metric.

    Args:
        w_metric (torch.Tensor): Metric to use for pruning.
        prune_granularity (str): Granularity of pruning. Can be "layer" or "row".
        sparsity (float): Sparsity level to achieve.
    Returns:
        torch.Tensor: Pruning mask of booleans, based on the metric and required sparsity.
    """

    # Compute mask layer-wise
    if prune_granularity == "layer":
        # Find threshold based on required sparsity
        thresh = torch.sort(w_metric.flatten().cuda())[0][
            int(w_metric.numel() * sparsity)
        ].cpu()

        # Create mask based on threshold
        w_mask = w_metric <= thresh

        return w_mask

    # Compute mask row-wise
    elif prune_granularity == "row":
        # Initialize boolean mask
        w_mask = torch.zeros_like(w_metric) == 1

        # Sort based on magnitude of w_metric
        sort_res = torch.sort(w_metric, dim=-1, stable=True)

        # Select indices to prune based on sparsity
        indices = sort_res[1][:, : int(w_metric.shape[1] * sparsity)]

        # Generate mask based on selected indices
        w_mask.scatter_(1, indices, True)

        return w_mask


def prune_vit(model, calibration_data, pruning_metric, pruning_granularity, sparsity):
    """
    Function to perform pruning on a Vision Transformer model.

    Args:
        model (nn.Module): Vision Transformer model to prune.
        calibration_data (torch.Tensor): Calibration samples to use for pruning.
        pruning_metric (str): Metric to use for pruning. Can be "magnitude" or "wanda".
        pruning_granularity (str): Structure on which pruning will be applied. Can be "layer" or "row".
        sparsity (float): Sparsity level to achieve.
    """

    # Obtain batch size
    batch_size = calibration_data.shape[0]

    # Check if wanda is requested
    require_forward = pruning_metric in ["wanda"]

    # Prepare metric stats
    metric_stats = []

    # Iterate through transformer blocks and store initial weights
    for blk in model.layers if isinstance(model, TinyViT) else model.blocks:
        # Select linear layers
        subset = find_layers(module=blk)

        # Get layer weights and store them in the metric stats list
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    # Apply model's patch embedding on the calibration data
    calibration_data = model.patch_embed(calibration_data)

    # Get model's cls_token
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)

    # Append cls_tokens to the calibration data
    calibration_data = torch.cat((cls_tokens, calibration_data), dim=1)

    # Apply positional embedding and dropout
    calibration_data = calibration_data + model.pos_embed

    # Apply dropout
    calibration_data = model.pos_drop(calibration_data)

    # Iterate through the transformer blocks
    for block_id, blk in enumerate(
        tqdm(model.blocks, desc="Pruning transformer blocks")
    ):
        wrapped_layers = None

        # Select linear layers
        subset = find_layers(blk)

        # Wrap layers for wanda
        if require_forward:
            # Prepare wrapped layers dictionary
            wrapped_layers = {}

            # Iterate through the linear layers and store the wrapped layers
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            # Define a function to call add_batch in the wrapped layer
            def add_batch(layer_name):
                def tmp(_, inp, out):
                    wrapped_layers[layer_name].add_batch(inp[0].data)

                return tmp

            # Register forward hooks for the add_batch function that will compute the pruning metric
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Perform forward pass through current transformer block
            if batch_size > 256:
                tmp_res = []
                for i1 in range(0, batch_size, 256):
                    j1 = min(i1 + 256, batch_size)
                    tmp_res.append(blk(calibration_data[i1:j1]))
                calibration_data = torch.cat(tmp_res, dim=0)
            else:
                calibration_data = blk(calibration_data)

            # Remove forward hooks
            for h in handles:
                h.remove()

        # Iterate through the linear layers and prune them
        for name in subset:
            # For the wanda pruning, multiply the scaler row (normalized layer input) with the metric (weight value).
            # Otherwise, for magnitude pruning, just the magnitude of the weights is used
            if pruning_metric == "wanda":
                # Update pruning metric stats
                metric_stats[block_id][name] *= torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                # Free wrapped layers memory
                del wrapped_layers[name]

            # Compute pruning mask
            w_mask = compute_mask(
                metric_stats[block_id][name], pruning_granularity, sparsity
            )

            # Free metric memory
            del metric_stats[block_id][name]

            # Set weights to zero based on the pruning mask
            subset[name].weight.data[w_mask] = 0

            # Free mask memory
            del w_mask
