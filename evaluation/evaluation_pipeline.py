import torch
from timm.utils import accuracy

from loggers.MetricLogger import MetricLogger


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    # Set criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Set up metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # Set model to eval mode
    model.eval()

    # Iterate through test images
    for batch in metric_logger.log_every(data_loader, 10, header):
        # Get batch and put them on device
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Perform forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # Get top-1 and top-5 accuracies
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # Update metric logger
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes and print them
    metric_logger.synchronize_between_processes(device)
    print(
        "\n* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
