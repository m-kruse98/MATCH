# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from ode_solver import ODESolver
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image, make_grid
from sklearn.metrics import roc_auc_score

from adeval import EvalAccumulatorCuda
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from ad_utils import f1_score_max
from functools import wraps

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def track_gpu_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = torch.cuda.memory_allocated()
        mem_samples = [start_mem]

        def sample_memory():
            mem_samples.append(torch.cuda.memory_allocated())

        class MemoryTracker:
            def __init__(self):
                self.running = True

            def track(self):
                while self.running:
                    sample_memory()
                    time.sleep(0.01)

        tracker = MemoryTracker()
        import threading
        thread = threading.Thread(target=tracker.track, daemon=True)
        thread.start()

        result = func(*args, **kwargs)

        tracker.running = False
        torch.cuda.synchronize()
        thread.join(timeout=0.1)
        sample_memory()

        max_mem = torch.cuda.max_memory_allocated()
        avg_mem = sum(mem_samples) / len(mem_samples)

        print(f"{func.__name__}: Max={max_mem/1e9:.3f}GB, Avg={avg_mem/1e9:.3f}GB")

        return result
    return wrapper


@track_gpu_memory
def eval_anomaly(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args: Namespace,
    train_statistics={"calculate": False,
                      "probabilities": None, "reconstructions": None},
    ode_method="midpoint",
    step_size=0.2,
    calc_div=False,
    M=1,
    use_rqmc=False,
):

    print(f"###\nUSING ODE METHOD: {ode_method}\n###")
    print(f"M= {M}. calc-div: {calc_div}")

    if train_statistics["probabilities"] is not None:
        print("####################\n\n STARTING EVALUATING!\n\n###################")
        collected_stats = train_statistics["probabilities"]
        print("Prepping accumulator for metric calculations...")
        min_stat, max_stat = torch.amin(
            collected_stats), torch.amax(collected_stats)
        min_stat = torch.floor(min_stat - 2 * abs(min_stat)).item()
        max_stat = torch.ceil(max_stat + 100 * abs(max_stat)).item()
        print(f"EvalAccumulator boundaries: [{min_stat}, {max_stat}].")

        detection_accum = EvalAccumulatorCuda(
            min_stat, max_stat, min_stat, max_stat, nstrips=2000)
        segment_accum = EvalAccumulatorCuda(
            min_stat, max_stat, min_stat, max_stat, nstrips=2000)

    else:
        print(
            "####################\n Gathering Training statistics...\n####################")
        detection_accum, segment_accum = None, None
        collected_stats, collected_reconstructions = list(), list()

    gc.collect()
    model.train(False)
    solver = ODESolver(velocity_model=model, args=args)

    all_labels, detection_scores, detection_scores_max = list(), list(), list()
    segment_scores_max, time_metrics = list(), list()

    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    def gaussian(tensor):
        # NOTE: Calculating the gaussian likelihood in channel dimension only (keep spatial dimension)
        _, C, H, W = tensor.shape
        log_likelihood = -0.5 * \
            torch.sum((tensor ** 2), dim=(1,)) - (C / 2) * \
            torch.log(torch.tensor(2 * torch.pi))
        return log_likelihood

    for load_idx, data in enumerate(tqdm.tqdm(data_loader)):

        if not args.samplewise:
            images, masks, labels = data["image"], data["mask"], data["label"]
            # unified wrapper-version
            with torch.no_grad():
                encoded_samples = encoder.encode(images.to(device))
        else:
            B = data["image_0"].shape[0]
            idx = torch.arange(B * 5)
            lists_to_samplewise = (idx % 5) * B + (idx // 5)

            images = torch.concat([data["image_0"], data["image_1"], data["image_2"],
                                  data["image_3"], data["image_4"]])[lists_to_samplewise]
            encoded_samples = encoder.encode(images.to(device))
            masks = torch.concat([data["mask_0"], data["mask_1"], data["mask_2"],
                                 data["mask_3"], data["mask_4"]])[lists_to_samplewise]
            labels = torch.concat(data["label"])[lists_to_samplewise]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.cond:
            if args.samplewise:
                conditioning = torch.concat(data["camera"])[
                    lists_to_samplewise].to(device)
            else:
                conditioning = data["camera"].to(device)
                conditioning = torch.randint_like(conditioning, 5)

        time_grid = torch.tensor([1.0, 0.0], device=device)
        step_size_tensor = torch.tensor([step_size]).to(device=device)

        # python train.py --seed 1000 --epochs 151 --eval_frequency 150 --eval_only --calc_div --encoder wideresnet --width_per_group 768 --eval_only --data_path audiojack --resume output_fm/save_ablation_models/audiojack/checkpoint-149.pth
        with torch.no_grad():
            start_time = time.time()
            # NOTE: For the RD-style model: output is layer 1, layer 2 and layer 3
            _, map_1, map_2, map_3 = solver.compute_likelihood(
                time_grid=time_grid,
                x_1=tuple(encoded_samples) if args.rd_model else encoded_samples,
                method=ode_method,
                step_size=step_size_tensor,
                exact_divergence=False,
                log_p0=gaussian,
                return_intermediates=False,
                extra=conditioning if args.cond else None,
                calc_div=calc_div,
                M=M,
                use_rqmc=use_rqmc
            )
            elapsed_time = time.time() - start_time

            batch_size = encoded_samples.shape[0] if not args.rd_model else encoded_samples[0].shape[0]
            time_per_sample = elapsed_time / batch_size
            samples_per_second = batch_size / elapsed_time

            time_metrics.append((time_per_sample, samples_per_second))

        # Convert to NEGATIVE log likelihood
        map_1 = -map_1.cpu().detach()
        map_2 = -map_2.cpu().detach()
        map_3 = -map_3.cpu().detach()

        if train_statistics["calculate"]:
            collected_stats.append(map_1)
            collected_reconstructions.append(map_1)
            # stop after n_iters batches of the trainset
            if load_idx >= train_statistics["n_iters"]:
                break
            continue

        # Warning when values are outside of the expected anomaly score ranges (estimated on train set)
        if map_1.amin() < min_stat:
            print(f"{load_idx:03d}. Warning: {(map_1.amin(dim=(0)) < min_stat).sum()} value(s) outside of accumulator min (min = {map_1.amin().item()})")
        if map_1.amax() > max_stat:
            print(f"{load_idx:03d}. Warning: {(map_1.amax(dim=(0)) > max_stat).sum()} value(s) outside of accumulator min (min = {map_1.amax().item()})")
        map_1 = torch.clamp(map_1, min_stat, max_stat)
        map_2 = torch.clamp(map_2, min_stat, max_stat)
        map_3 = torch.clamp(map_3, min_stat, max_stat)

        map_2_and_3 = map_2 + map_3
        map_1_and_2_and_3 = map_1 + map_2 + map_3

        # mean and max aggregation strategies
        segment_score = map_1_and_2_and_3.detach().cpu().mean(dim=(1, 2))
        segment_score_max = map_1_and_2_and_3.detach().cpu().amax(dim=(1, 2))

        detection_score = map_2_and_3.detach().cpu().mean(dim=(1, 2))
        detection_score_max = map_2_and_3.detach().cpu().amax(dim=(1, 2))

        masks = torch.nn.functional.interpolate(masks, (256, 256),
                                                mode="nearest").to(torch.uint8).squeeze(1)
        detection_map = torch.nn.functional.interpolate(map_2_and_3.unsqueeze(1).to(device).double(),
                                                        (256, 256), mode="bilinear").squeeze(1)
        segment_map = torch.nn.functional.interpolate(map_1_and_2_and_3.unsqueeze(1).to(device).double(),
                                                      (256, 256), mode="bilinear").squeeze(1)

        if args.viz:
            for viz_idx in range(segment_map.shape[0]):
                save_to = (Path(f"viz/maps_paper/") /
                           data["filename"][viz_idx]).with_suffix(".png")
                save_to.parent.mkdir(parents=True, exist_ok=True)

                plt.imsave(save_to, segment_map[viz_idx].cpu().detach().numpy(
                ), vmin=segment_map.cpu().amin(), vmax=segment_map.cpu().amax())

        detection_accum.add_anomap_batch(detection_map, masks)
        detection_accum.add_image(detection_score, labels)
        segment_accum.add_anomap_batch(segment_map, masks)
        segment_accum.add_image(segment_score, labels)

        detection_scores.append(detection_score)
        detection_scores_max.append(detection_score_max)
        segment_scores_max.append(segment_score_max)
        all_labels.append(labels.detach().cpu())

        # if load_idx == 10:
        #     break

    if train_statistics["calculate"]:
        return {
            "probabilities": torch.concat(collected_stats, dim=0),
            "reconstructions": torch.concat(collected_reconstructions, dim=0) if encoder.can_decode else None
        }

    all_labels = torch.concatenate(all_labels).numpy()
    detection_scores = torch.concatenate(detection_scores).numpy()
    detection_scores_max = torch.concatenate(detection_scores_max).numpy()
    segment_scores_max = torch.concatenate(segment_scores_max).numpy()

    # metrics calculation
    to_return = {}
    auroc = roc_auc_score(all_labels > 0, detection_scores)

    print("ATTENTION: FULL IS NOW MAPS 2 + 3 (FOR DETECTION), WHILE LOGPROB IS NOW MAPS  1 + 2 (FOR SEGMENTATION)")
    if encoder.can_decode:
        variants = ["full"]
        max_scores = [detection_scores_max]
        accums = [detection_accum]
    else:
        variants = ["map2+3", "map1+2+3"]
        max_scores = [detection_scores_max,  segment_scores_max]
        accums = [detection_accum, segment_accum]
    for variant, max_score, accum in zip(variants, max_scores, accums):

        sample_wise_labels = (all_labels > 0).reshape((-1, 5)).any(axis=1)
        scores_samplewise = max_score.reshape((-1, 5)).max(axis=1)

        f1_sample = f1_score_max(sample_wise_labels, scores_samplewise)

        for lbl, score in zip(sample_wise_labels, scores_samplewise):
            accum.add_sample(score, lbl)

        print(f"Evaluating variant: {variant}")

        auroc_max = roc_auc_score(all_labels > 0, max_score)
        f1_max_det = f1_score_max(all_labels, max_score)
        print("AUROC with max aggregation:", auroc_max)
        try:
            print(accum.summary())
            metrics = accum.summary()
        except Exception as e:
            print(
                f"CUDA-supported Metrics failed (values out-of-bounds!) Setting them to zero!")
            print(e)
            metrics = {
                "i_auroc": auroc,
                "p_auroc": 0,
                "p_aupro": 0,
                "i_aupr": 0,
                "p_aupr": 0,
            }
        to_return.update({
            f"{variant}_I_auroc": metrics["i_auroc"] * 100,
            f"{variant}_P_auroc": metrics["p_auroc"] * 100,
            f"{variant}_P_aupro": metrics["p_aupro"] * 100,
            f"{variant}_I_aupr": metrics["i_aupr"] * 100,
            f"{variant}_P_aupr": metrics["p_aupr"] * 100,
            f"{variant}_I_auroc_max": auroc_max * 100,
            f"{variant}_I_F1-Max": f1_max_det * 100,
            f"{variant}_S_F1-Max": f1_sample * 100,
            f"{variant}_S-auroc": metrics["s_auroc"] * 100,
            f"{variant}_S-aupr": metrics["s_aupr"] * 100,
        })

    time_per_sample = np.mean([t[0] for t in time_metrics])
    samples_per_second = np.mean([t[1] for t in time_metrics])

    to_return.update({
        "time_per_sample": time_per_sample,
        "samples_per_second": samples_per_second,
    })

    print(to_return)
    return to_return


def eval_model(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    model.train(False)

    solver = ODESolver(velocity_model=model, args=args)
    ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, data in enumerate(data_loader):
        images, labels = data["image"], data["label"]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if not args.samplewise:
            images, labels = data["image"].to(device), data["label"]
            # unified wrapper-version
            with torch.no_grad():
                samples = decoder.encode(images)
        else:
            B = data["image_0"].shape[0]
            idx = torch.arange(B * 5)
            lists_to_samplewise = (idx % 5) * B + (idx // 5)
            samplewise_to_lists = torch.argsort(lists_to_samplewise)

            # NOTE: old version where each view has its seperate list
            # samples = [encoder.encode(f.to(device, non_blocking=True)) for f in [data["image_0"], data["image_1"], data["image_2"], data["image_3"], data["image_4"]]]
            # samples_infused = model.cross_view(samples)
            # samples = torch.concat(samples_infused)[lists_to_samplewise]

            images = torch.concat([data["image_0"], data["image_1"], data["image_2"],
                                  data["image_3"], data["image_4"]])[lists_to_samplewise].to(device)
            samples = decoder.encode(images)
            labels = torch.concat(data["label"])[lists_to_samplewise]

        if args.cond:
            if args.samplewise:
                conditioning = torch.concat(data["camera"])[
                    lists_to_samplewise].to(device)
            else:
                conditioning = data["camera"].to(device)

        fid_metric.update(images, real=True)

        if num_synthetic < fid_samples:

            # Continuous sampling
            x_0 = torch.randn(
                samples.shape, dtype=torch.float32, device=device)

            time_grid = torch.tensor([0.0, 1.0], device=device)

            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                return_intermediates=False,
                atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                step_size=ode_opts["step_size"]
                if "step_size" in ode_opts
                else None,
                extra=conditioning if args.cond else None
            )
            with torch.no_grad():
                synthetic_samples = decoder.decode(synthetic_samples)

            synthetic_samples = torch.clamp(
                synthetic_samples, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                # in {cfg_scaled_model.get_nfe()} evaluations."
                f"{samples.shape[0]} samples generated:"
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

                grid = make_grid(synthetic_samples)
                # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
                image_array_to_log = grid.mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {"snapshot": image_array_to_log}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    return {"fid": float(fid_metric.compute().detach().cpu()), "snapshot": image_array_to_log}
