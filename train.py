# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from train_arg_parser import get_args_parser

from training.eval_loop import eval_model, eval_anomaly
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch
import wandb

from ad_utils import LatentWrapper, model_size_info, manta_categories, get_datasets, get_model


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


program_start_time = time.time()


def main(args, data_base_path):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.eval_frequency = min(args.eval_frequency, args.epochs)
    logger.info("job dir: {}".format(
        os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    args_filepath = Path(args.output_dir) / "args.json"
    logger.info(f"Saving args to {args_filepath}")
    with open(args_filepath, "w") as f:
        json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, dataset_test = get_datasets(
        dataset_name, args, data_base_path)
    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    args.train_length = len(dataset_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    if not args.multiclass:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None
    logger.info(str(sampler_train))

    # Setup wandb
    wandb.run = None
    if not args.multiclass:
        if args.wandb:
            wandb_config = {"batch_size": args.batch_size, "cond": args.cond, "data_path": args.data_path,
                            "encoder": args.encoder, "epochs": args.epochs, "eval_frequency": args.eval_frequency,
                            "img_size": args.img_size, "lr": args.lr,
                            "n_train_stats": args.n_train_stats, "pos_enc": args.pos_enc, "prefix": args.prefix, "seed": args.seed,
                            "width_per_group": args.width_per_group}
            if dataset_name == "manta":
                wandb_config.update(
                    {"category": manta_categories[args.data_path]})
            run = wandb.init(project=f"{args.project}_{args.dataset}", config=wandb_config,
                             name=f"{args.prefix}_{args.data_path}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                             mode="online", settings=wandb.Settings(start_method='thread'))

            wandb.define_metric("train_loss", step_metric="train_step")
            for variant in ["logprob", "full", "denoise"]:
                wandb.define_metric(f"{variant}_I-auroc", step_metric="epoch")
                wandb.define_metric(
                    f"{variant}_I-auroc_MAX", step_metric="epoch")
                wandb.define_metric(f"{variant}_I-aupr", step_metric="epoch")
                wandb.define_metric(f"{variant}_P-auroc", step_metric="epoch")
                wandb.define_metric(f"{variant}_P-aupro", step_metric="epoch")
                wandb.define_metric(f"{variant}_P-aupr", step_metric="epoch")
            wandb.define_metric("snapshot", step_metric="epoch")
            if args.compute_fid:
                wandb.define_metric("FID", step_metric="epoch")
    else:
        run = None
    args.wandb_run = run if args.wandb else None

    model = get_model(args, device)
    vae_encoder = LatentWrapper(args.encoder, device=args.device, args=args)

    model_without_ddp = model

    eff_batch_size = (
        args.batch_size * args.accum_iter
    )

    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    print("Flow Model:")
    print(model_size_info(model=model))
    print("Encoder:")
    print(model_size_info(vae_encoder.model))

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                encoder=vae_encoder,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if args.save_model and epoch == args.epochs - 1:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            fid_samples = args.fid_samples  # // num_tasks

            with torch.no_grad():

                train_statistics = eval_anomaly(
                    model,
                    vae_encoder,
                    data_loader_train,
                    device,
                    epoch=epoch,
                    args=args,
                    train_statistics={"calculate": True, "probabilities": None, "reconstructions": None,
                                      "n_iters": args.n_train_stats},
                    ode_method="euler"
                )

            if not args.multiclass:
                anomaly_stats = eval_anomaly(
                    model,
                    vae_encoder,
                    data_loader_test,
                    device,
                    epoch=epoch,
                    args=args,
                    train_statistics={"calculate": False,
                                      "probabilities": train_statistics["probabilities"],
                                      "reconstructions": train_statistics["reconstructions"]},
                    ode_method="euler",
                    calc_div=args.calc_div,
                )
            else:
                from training.multiclass import eval_multi_class
                eval_multi_class(dataset_name, args, model,
                                 vae_encoder, device, epoch, train_statistics, data_base_path)
                continue

            if args.encoder in ["stable-diffusion", "identity"]:
                eval_stats = eval_model(
                    model,
                    vae_encoder,
                    data_loader_train,
                    device,
                    epoch=epoch,
                    fid_samples=fid_samples,
                    args=args,
                )
                log_stats.update(
                    {f"eval_{k}": v for k, v in eval_stats.items() if k != "snapshot"})
                anomaly_stats.update(
                    {"snapshot": wandb.Image(eval_stats["snapshot"])})
            anomaly_stats.update({
                "epoch": epoch,
                "FID": eval_stats["fid"] if args.compute_fid else 0,
            })
            if args.wandb:
                wandb.log(anomaly_stats)

        if args.output_dir:
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir) /
                              args.prefix / args.data_path)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mvtec2d_classes = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                       "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    realiad_classes = ["audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug", "porcelain_doll", "regulator",
                       "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock", "toothbrush_r", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper_r"]
    manta_classes = ['maize', 'paddy', 'soybean', 'wheat', 'block_inductor', 'copper_standoff', 'flat_nut', 'led_pad', 'led', 'long_button', 'power_inductor', 'short_button', 'thin_resistor', 'type_c', 'wafer_resistor', 'coffee_beans', 'goji_berries', 'pistachios', 'button', 'gear',
                     'nut_cap', 'nut', 'red_washer', 'round_button_cap', 'screw_m', 'square_button_cap', 'terminal', 'wire_cap', 'yellow_green_washer', 'capsule_m', 'coated_tablet', 'embossed_tablet', 'lettered_tablet', 'oblong_tablet', 'pink_tablet', 'red_tablet', 'white_tablet', 'yellow_tablet']

    # TODO: Make sure you adjust the paths to the data set you want to use!!!
    if args.data_path in mvtec2d_classes:
        data_base_path = "/data/anomaly_detection/mvtec/"
        dataset_name = "mvtec2d"
        args.dataset = dataset_name
    elif args.data_path in realiad_classes:
        data_base_path = "/data/anomaly_detection/realiad/"
        dataset_name = "realiad"
        # NOTE: remove '_r' for distinguishing between realiad and mvtec2d classes (namely toothbrush & zipper)
        args.data_path = re.sub(r"_r$", "", args.data_path)
    elif args.data_path in manta_classes:
        print(f"Running a MANTA class!")
        if args.full_manta:
            print(f"Encountered FULL MANTA!!!")
            data_base_path = "/data/anomaly_detection/MANTA_realiad/"
        else:
            print(f"Encountered regular MANTA-TINY!!!")
            data_base_path = "/data/anomaly_detection/MANTA-Tiny_realiad/"
        dataset_name = "manta"
        # NOTE: remove '_m' for distinguishing between realiad and mvtec2d classes (namely screw and capsule)
        args.data_path = re.sub(r"_m$", "", args.data_path)
    else:
        raise ValueError(
            f"No classname/data_path with corresponding data set found!")

    args.dataset = dataset_name

    args.cond = (dataset_name == "realiad" or dataset_name ==
                 "manta") and not args.no_cond

    main(args, data_base_path=data_base_path)
