import wandb
from training.eval_loop import eval_anomaly
from datasets.data_builder import build_dataloader, config_dict
import torch
import datetime
import os
from ad_utils import manta_categories


def eval_multi_class(dataset_name, args, model, vae_encoder, device, epoch, train_statistics, data_base_path):
    realiad_classes = ["audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug", "porcelain_doll", "regulator",
                       "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper"]
    manta_classes = ['maize', 'paddy', 'soybean', 'wheat', 'block_inductor', 'copper_standoff', 'flat_nut', 'led_pad', 'led', 'long_button', 'power_inductor', 'short_button', 'thin_resistor', 'type_c', 'wafer_resistor', 'coffee_beans', 'goji_berries', 'pistachios', 'button', 'gear',
                     'nut_cap', 'nut', 'red_washer', 'round_button_cap', 'screw', 'square_button_cap', 'terminal', 'wire_cap', 'yellow_green_washer', 'capsule', 'coated_tablet', 'embossed_tablet', 'lettered_tablet', 'oblong_tablet', 'pink_tablet', 'red_tablet', 'white_tablet', 'yellow_tablet']

    all_classes = realiad_classes if dataset_name == "realiad" else manta_classes
    for cur_class in all_classes:

        print(
            f"###############\n\nInferencing class {cur_class}\n\n###############")
        if args.wandb:
            wandb_config = {"batch_size": args.batch_size, "cond": args.cond, "data_path": cur_class,
                            "encoder": args.encoder, "epochs": args.epochs, "eval_frequency": args.eval_frequency,
                            "img_size": args.img_size, "lr": args.lr, "maps": args.maps,
                            "n_train_stats": args.n_train_stats,
                            "pos_enc": args.pos_enc, "prefix": args.prefix, "seed": args.seed,
                            "width_per_group": args.width_per_group}
            if dataset_name == "manta":
                wandb_config.update({"category": manta_categories[cur_class]})
            run = wandb.init(project=f"{args.project}_{args.dataset}_{epoch}", config=wandb_config,
                             name=f"{args.prefix}_{cur_class}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
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

        config_obj = config_dict[dataset_name]

        config_obj = config_dict[dataset_name]

        config_obj["image_reader"]["kwargs"]["image_dir"] = os.path.join(
            data_base_path, "classes")
        config_obj["train"]["meta_file"] = os.path.join(
            data_base_path, "jsons", "realiad_jsons")
        config_obj["test"] = {"meta_file": os.path.join(
            data_base_path, "jsons", "realiad_jsons")}

        config_obj["train"]["meta_file"] = os.path.join(
            config_obj['train']['meta_file'], f"{cur_class}.json")
        config_obj["test"]["meta_file"] = os.path.join(
            config_obj['test']['meta_file'], f"{cur_class}.json")
        config_obj["classname"] = cur_class
        config_obj["input_size"] = (args.img_size, args.img_size)
        config_obj["type"] = "explicit"
        _, dataset_test = build_dataloader(config_obj, get_loader=False)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

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
            calc_div=args.calc_div
        )
        if args.wandb:
            wandb.log(anomaly_stats)
            wandb.finish()
