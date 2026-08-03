import logging

from datasets.custom_dataset import build_custom_dataloader
from datasets.explicit_dataset import build_explicit_dataloader
from datasets.samplewise_dataset import build_samplewise_dataloader
from datasets.multiclass_dataset import build_multiclass_dataloader

logger = logging.getLogger("global")


config_dict = {
    "realiad":  {
        "feature_dir": 'tmp/',
        "type": "explicit",
        "image_reader": {
            "type": "opencv",
            "kwargs": {
                "color_mode": "RGB",
            }
        },
        "input_size": (256, 256),
        "pixel_mean": (0.485, 0.456, 0.406),
        "pixel_std": (0.229, 0.224, 0.225),
        "train": {
            "hflip": False,
            "rebalance": False,
            "rotate": False,
            "vflip": False,
        },
        "type": "explicit",  # "feature",
        "workers": 4,
        "batch_size": 32,
    },
    "manta":  {
        "feature_dir": 'tmp/',
        "type": "explicit",
        "image_reader": {
            "type": "opencv",
            "kwargs": {
                "color_mode": "RGB",
            }
        },
        "input_size": (256, 256),
        "pixel_mean": (0.485, 0.456, 0.406),
        "pixel_std": (0.229, 0.224, 0.225),
        "train": {
            "hflip": False,
            "rebalance": False,
            "rotate": False,
            "vflip": False,
        },
        "type": "explicit",  # "feature",
        "workers": 4,
        "batch_size": 32,
    },
}


def build(cfg, training, get_loader):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    print(f"Dataset type: {dataset}")
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training)
    elif dataset == 'explicit':
        data_loader = build_explicit_dataloader(
            cfg, training, get_loader=get_loader)
    elif dataset == 'samplewise':
        data_loader = build_samplewise_dataloader(
            cfg, training, get_loader=get_loader)
    elif dataset == "multiclass":
        data_loader = build_multiclass_dataloader(
            cfg, training, get_loader=get_loader) if training else None
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, get_loader=True):

    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, get_loader=get_loader)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, get_loader=get_loader)

    logger.info("build dataset done")
    return train_loader, test_loader
