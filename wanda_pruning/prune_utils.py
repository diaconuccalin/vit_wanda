import torch
import torch.nn as nn
from tqdm import tqdm

from wanda_pruning.WrappedLayer import WrappedLayer


def find_layers(module, layers=None, name=""):
    if layers is None:
        layers = [nn.Linear]

    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )

    return res


def check_sparsity(model):
    subset = find_layers(model, layers=[nn.Linear])
    zero_cnt = 0
    fc_params = 0

    for name in subset:
        w = subset[name].weight.data
        if w.shape[0] == 1000:
            continue
        zero_cnt += (w == 0).sum().item()
        fc_params += w.numel()

    return float(zero_cnt) / fc_params


def compute_mask(w_metric, prune_granularity, sparsity):
    if prune_granularity == "layer":
        thresh = torch.sort(w_metric.flatten().cuda())[0][
            int(w_metric.numel() * sparsity)
        ].cpu()
        w_mask = w_metric <= thresh

        return w_mask

    elif prune_granularity == "row":
        w_mask = torch.zeros_like(w_metric) == 1
        sort_res = torch.sort(w_metric, dim=-1, stable=True)

        indices = sort_res[1][:, : int(w_metric.shape[1] * sparsity)]
        w_mask.scatter_(1, indices, True)

        return w_mask


def prune_vit(args, model, calib_data):
    inps = calib_data
    bs = inps.shape[0]
    require_forward = args.prune_metric in ["wanda"]

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}

        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)

        metric_stats.append(res_per_layer)

    inps = model.patch_embed(inps)

    cls_tokens = model.cls_token.expand(bs, -1, -1)

    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(
        tqdm(model.blocks, desc="Pruning transformer blocks")
    ):
        wrapped_layers = None
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(nam):
                def tmp(_, inp, out):
                    wrapped_layers[nam].add_batch(inp[0].data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1 + 256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()

        # Pruning
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

            w_mask = compute_mask(
                metric_stats[block_id][name], args.prune_granularity, args.sparsity
            )

            subset[name].weight.data[w_mask] = 0
