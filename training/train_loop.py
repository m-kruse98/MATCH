# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import wandb

import torch
from flow_matching.path import CondOTProbPath
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount


logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def train_one_epoch(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter

    path = CondOTProbPath()

    for data_iter_step, data in enumerate(data_loader):

        optimizer.zero_grad()
        batch_loss.reset()

        if not args.samplewise:
            images, labels = data["image"], data["label"]
            # unified wrapper-version
            with torch.no_grad():
                samples = encoder.encode(images.to(device))
        else:
            B = data["image_0"].shape[0]
            idx = torch.arange(B * 5)
            lists_to_samplewise = (idx % 5) * B + (idx // 5)
            samplewise_to_lists = torch.argsort(lists_to_samplewise)
            images = torch.concat([data["image_0"], data["image_1"], data["image_2"],
                                  data["image_3"], data["image_4"]])[lists_to_samplewise]
            samples = encoder.encode(images.to(device))
            labels = torch.concat(data["label"])[lists_to_samplewise]

        # build view conditioning
        if args.cond:
            if args.samplewise:
                conditioning = torch.concat(data["camera"])[
                    lists_to_samplewise].to(device)

            else:
                conditioning = data["camera"].to(device)
        else:
            conditioning = None

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # NOTE: currently, we use t * X_1 + (1-t) * X_0
        #       aka, the regular straight-line interpolation
        # NOTE: this makes them "equivalent" to all other affine interpolation paths:
        #       see also spherical linear interp or DDPM/DDIM here:
        #       https://rectifiedflow.github.io/blog/2024/interpolation/

        if args.rd_model:
            noise = [torch.randn_like(a).to(device) for a in samples]
            t = torch.torch.rand(samples[0].shape[0]).to(device)

            paths = [path.sample(t=t, x_0=n, x_1=s)
                     for n, s in zip(noise, samples)]
            x_t = [p.x_t for p in paths]
            u_t = [p.dx_t for p in paths]

            with torch.cuda.amp.autocast():
                model_output = model(x_t, t, extra=conditioning)
                loss = [torch.pow(m - u, 2).mean()
                        for m, u in zip(model_output, u_t)]
                loss = torch.mean(torch.stack(loss))

        else:
            noise = torch.randn_like(samples).to(device)
            t = torch.torch.rand(samples.shape[0]).to(device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                loss = torch.pow(
                    model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
            clip_grad=1.0
        )

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
