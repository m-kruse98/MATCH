# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import logging

from torchdiffeq._impl.odeint import SOLVERS

logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")

    parser.add_argument(
        "--data_path",
        default="audiojack",
        type=str,
        help="name of the class to run AD on. NOTE: due to naming conflicts 'toothbrush' and 'zipper' of Real-IAD are written 'toothbrush_r' and 'zipper_r'. 'screw' and 'capsule' of MANTA-Tiny are named 'screw_m' and 'capsule_m'!",
    )

    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--eval_frequency", default=75, type=int,
                        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.")
    parser.add_argument("--wandb", action="store_true",
                        help="set to log to wandb")
    parser.add_argument("--full_manta", action="store_true",
                        help="switch to data path for the full MANTA data set")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--encoder", default="wideresnet", type=str,
                        choices=["stable-diffusion", "identity", "effnet", "wideresnet", "dinov3"])
    parser.add_argument("--pos_enc", action="store_true",
                        help="Whether to use positional encoding (U-Net only)")
    parser.add_argument("--prefix", default="to_delete", type=str)
    parser.add_argument("--project", default="flow_matching_ad",
                        type=str, help="project name for wandb")
    parser.add_argument("--samplewise", action="store_true",
                        help="Whether to samplewise data-loading (one sample per object)")
    parser.add_argument("--rd_model", action="store_true", default=True,
                        help="Whether to use the RD4AD inspired model. Otherwise, train a U-Net on a single feature scale")
    parser.add_argument("--calc_div", action="store_true",
                        help="Whether to use calculate the divergence in likelihood estimation")
    parser.add_argument("--width_per_group", default=768, type=int)
    parser.add_argument("--save_model", action="store_true",
                        help="Saving model at the end of the run")
    parser.add_argument("--no_cond", action="store_true",
                        help="Whether we OMIT view conditioning")
    parser.add_argument("--viz", action="store_true",
                        help="Whether we visualize anomaly maps")
    parser.add_argument("--multiclass", action="store_true",
                        help="Whether we run the experiment on all classes at once")
    parser.add_argument("--n_train_stats", default=3, type=int,
                        help="Amount of training batches we sample, to gather bounds for ADEval.")

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_fm",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--ode_method",
        default="midpoint",
        choices=list(SOLVERS.keys()) + ["edm_heun"],
        help="ODE solver used to generate samples.",
    )
    parser.add_argument(
        "--ode_options",
        default='{"step_size": 0.01}',
        type=json.loads,
        help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.2,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--fid_samples",
        default=50000,
        type=int,
        help="number of synthetic samples for FID evaluations",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )

    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.",
    )
    parser.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save all samples generated for FID computation.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )

    return parser
