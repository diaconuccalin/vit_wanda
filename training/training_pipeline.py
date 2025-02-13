import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
from timm.data import Mixup
from timm.utils import ModelEma

from loggers.MetricLogger import MetricLogger
from utils.SmoothedValue import SmoothedValue


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    wandb_logger=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    schedules=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    use_amp=False,
):
    # Set schedules to empty dict if not provided
    if schedules is None:
        schedules = {}

    # Set model to training mode
    model.train(True)

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    # Reset gradients
    optimizer.zero_grad()

    # Iterate over the data loader
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Calculate the current step and iteration
        step = data_iter_step // update_freq
        it = start_steps + step  # global training iteration

        # Skip if the step is greater than the number of training steps per epoch
        if step >= num_training_steps_per_epoch:
            continue

        # Update LR & WD for the first acc
        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = (
                            lr_schedule_values[it] * param_group["lr_scale"]
                        )

                    if (
                        wd_schedule_values is not None
                        and param_group["weight_decay"] > 0
                    ):
                        param_group["weight_decay"] = wd_schedule_values[it]

            if "dp" in schedules:
                model.module.update_drop_path(schedules["dp"][it])

            if "do" in schedules:
                model.module.update_dropout(schedules["do"][it])

        # Move input and labels to the device
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Perform mixup if needed
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Perform the forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:  # full precision
            output = model(samples)
            loss = criterion(output, targets)

        # Perform the backward pass
        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        # Update according to gradient
        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )

            loss /= update_freq

            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

                if model_ema is not None:
                    model_ema.update(model)

        else:  # full precision
            loss /= update_freq
            loss.backward()

            grad_norm = None

            for k, m in enumerate(model.modules()):
                if isinstance(m, nn.Linear):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    m.weight.grad.data.mul_(mask)

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        # Update the metric logger
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)

        min_lr = 10.0
        max_lr = 0.0

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if "dp" in schedules:
            metric_logger.update(drop_path=model.module.drop_path)

        if "do" in schedules:
            metric_logger.update(dropout=model.module.drop_rate)

        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        # Update log writer
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        # Update wandb logger
        if wandb_logger:
            wandb_logger._wandb.log(
                {
                    "Rank-0 Batch Wise/train_loss": loss_value,
                    "Rank-0 Batch Wise/train_max_lr": max_lr,
                    "Rank-0 Batch Wise/train_min_lr": min_lr,
                },
                commit=False,
            )

            if class_acc:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_class_acc": class_acc}, commit=False
                )

            if use_amp:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_grad_norm": grad_norm}, commit=False
                )

            wandb_logger._wandb.log({"Rank-0 Batch Wise/global_train_step": it})

    # Gather stats from all processes and print them
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
