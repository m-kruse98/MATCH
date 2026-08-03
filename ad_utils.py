from typing import List, Dict, Optional

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
import timm
import transformers
from sklearn.metrics import precision_recall_curve

from models.unet import UNetModel
from models.rd_timestep import RDWrapper, RDWrapperSingleScale
from efficientnet_pytorch import EfficientNet

from datasets.mvtec import MVTecAD
from datasets.data_builder import build_dataloader, config_dict


class UNetWrapper(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(self, x, timesteps, extra=None):
        return self.model.forward(timesteps, x, y=extra)

    def forward(self, x, t, extra=None):
        return self.model.forward(t, x, y=extra)


def get_model(args, device):
    if args.rd_model:
        if args.encoder == "dinov3":
            # DINOv3
            model = RDWrapperSingleScale(num_feature_maps=3, feature_channels=384,
                                         width_per_group=args.width_per_group, model_channels=1024, args=args).to(device)
        else:
            model = RDWrapper(width_per_group=args.width_per_group,
                              model_channels=1024, args=args).to(device)
        args.encoder = f"{args.encoder}_rd"
    else:
        dim = {
            "stable-diffusion": (4, args.img_size // 8, args.img_size // 8),
            "identity": (3, args.img_size, args.img_size),
            "effnet": (304, 24, 24),
            # for 512x512 images: (512, 64, 64), # for 256x256 images: (512, 32, 32),
            "wideresnet": (512, 28, 28),
            "dinov3": (384, 16, 16),  # (1024, 16, 16),
        }[args.encoder]
        n_channels = {
            "stable-diffusion": 64,
            "identity": 32,
            "effnet": 128,
            "wideresnet": 512,
            "dinov3": 512,
        }[args.encoder]
        model = UNetWrapper(model=UNetModel(
            dim=dim,
            num_channels=n_channels,
            num_res_blocks=3,  # 3,
            use_pos_enc=args.pos_enc,
            class_cond=args.cond,
            num_classes=5 if args.cond else None,
            cross_attention=args.samplewise)).to(device)

    return model


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def get_datasets(dataset_name, args, data_base_path):

    if dataset_name in ["realiad", "manta"]:
        config_obj = config_dict[dataset_name]

        config_obj["image_reader"]["kwargs"]["image_dir"] = os.path.join(
            data_base_path, "classes")
        config_obj["train"]["meta_file"] = os.path.join(
            data_base_path, "jsons", "realiad_jsons")
        config_obj["test"] = {"meta_file": os.path.join(
            data_base_path, "jsons", "realiad_jsons")}

        config_obj["train"]["meta_file"] = os.path.join(
            config_obj['train']['meta_file'], f"{args.data_path}.json")
        config_obj["test"]["meta_file"] = os.path.join(
            config_obj['test']['meta_file'], f"{args.data_path}.json")
        config_obj["classname"] = args.data_path
        config_obj["input_size"] = (args.img_size, args.img_size)
        if args.samplewise:
            config_obj["type"] = "samplewise"
        if args.multiclass:
            config_obj["type"] = "multiclass"
        dataset_train, dataset_test = build_dataloader(
            config_obj, get_loader=False)

    elif dataset_name == "mvtec2d":
        # define transforms
        transform = v2.Compose(
            [v2.Resize((args.img_size, args.img_size)), v2.ToTensor()])

        def target_transform(x): return x != 0

        dataset_train = MVTecAD(
            root=data_base_path,
            subset_name=args.data_path,
            image_size=(args.img_size, args.img_size),
            train=True,
            transform=transform,
            target_transform=target_transform,
            mask_transform=transform
        )
        dataset_test = MVTecAD(
            root=data_base_path,
            subset_name=args.data_path,
            image_size=(args.img_size, args.img_size),
            train=False,
            transform=transform,
            target_transform=target_transform,
            mask_transform=transform
        )
    else:
        return ValueError(f"No valid data set given with {dataset_name}. Choose: [realiad, manta, mvtec2d]")

    return dataset_train, dataset_test


manta_categories = {
    'maize': "agriculture", 'paddy': "agriculture", 'soybean': "agriculture", 'wheat': "agriculture",
    'block_inductor': "electronics", 'copper_standoff': "electronics", 'flat_nut': "electronics", 'led_pad': "electronics",
    'led': "electronics", 'long_button': "electronics", 'power_inductor': "electronics", 'short_button': "electronics",
    'thin_resistor': "electronics", 'type_c': "electronics", 'wafer_resistor': "electronics",
    'coffee_beans': "groceries", 'goji_berries': "groceries", 'pistachios': "groceries",
    'button': "mechanics", 'gear': "mechanics", 'nut_cap': "mechanics", 'nut': "mechanics", 'red_washer': "mechanics",
    'round_button_cap': "mechanics", 'screw': "mechanics", 'square_button_cap': "mechanics", 'terminal': "mechanics",
    'wire_cap': "mechanics", 'yellow_green_washer': "mechanics", 'capsule': "medicine", 'coated_tablet': "medicine",
    'embossed_tablet': "medicine", 'lettered_tablet': "medicine", 'oblong_tablet': "medicine", 'pink_tablet': "medicine",
    'red_tablet': "medicine", 'white_tablet': "medicine", 'yellow_tablet': "medicine"
}


def model_size_info(model):
    if model is None:
        num_params = 0
        model_size_mb = 0
    else:
        # Get the number of parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Get the size of the model in MB
        model_size_mb = sum(p.element_size() * p.numel()
                            for p in model.parameters()) / (1024 * 1024)

    # Format the output string
    output = f"**Model Size Info**\n"
    output += f"  * Number of Parameters: {num_params:,}\n"
    output += f"  * Model Size (MB): {model_size_mb:.2f} MB"

    return output


class FeatureDissimilarityCalculator:
    def __init__(self, model_name: str = 'wide_resnet50_2', target_layers: Optional[List[str]] = None, device: str = 'cuda'):
        self.device = device
        self.model = timm.create_model(
            model_name, pretrained=True, features_only=False)
        self.model = self.model.to(device)
        self.model.eval()

        if target_layers is None:
            if 'wide_resnet' in model_name:
                self.target_layers = ['layer1', 'layer2', 'layer3']
                # self.target_layers = ['layer2', 'layer3'] # 'layer1',
            elif 'efficientnet' in model_name:
                self.target_layers = ['blocks.2', 'blocks.4', 'blocks.6']
            else:
                raise ValueError(
                    f"Default layers not defined for {model_name}")
        else:
            self.target_layers = target_layers

        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):

        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.features.clear()

        with torch.no_grad():
            _ = self.model(x)

        return dict(self.features)

    def compute_dissimilarity(self, reconst_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        reconst_images = reconst_images.to(self.device)
        orig_images = orig_images.to(self.device)

        reconst_features = self._extract_features(reconst_images)
        orig_features = self._extract_features(orig_images)

        total_dissimilarity = None

        for layer_name in self.target_layers:
            reconst_feat = reconst_features[layer_name]
            orig_feat = orig_features[layer_name]

            reconst_feat_norm = F.normalize(reconst_feat, p=2, dim=1)
            orig_feat_norm = F.normalize(orig_feat, p=2, dim=1)

            cosine_sim = torch.sum(reconst_feat_norm *
                                   orig_feat_norm, dim=1, keepdim=True)
            cosine_dissim = 1 - cosine_sim

            if total_dissimilarity is None:
                total_dissimilarity = cosine_dissim
            else:
                cosine_dissim_resized = F.interpolate(cosine_dissim, size=total_dissimilarity.shape[2:],
                                                      mode='bilinear', align_corners=False)
                total_dissimilarity = total_dissimilarity + cosine_dissim_resized

        return total_dissimilarity.squeeze(1)

    def __del__(self):
        for hook in self.hooks:
            hook.remove()


class FeatureExtractor(torch.nn.Module):
    def __init__(self, layer_idx=35):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained(
            'efficientnet-b5')
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.feature_extractor._swish(
            self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / \
                    len(self.feature_extractor._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.layer_idx:
                return x


class LatentWrapper(torch.nn.Module):

    def __init__(self, model_type, device, args):
        super().__init__()

        self.type = model_type
        self.device = device
        if model_type == "identity":
            self.encode = self.encode_identity
            self.decode = self.decode_identity
            self.can_decode = True
            self.model = None
        elif model_type == "effnet":
            self.encode = self.encode_effnet
            self.model = FeatureExtractor().to(device)
            self.can_decode = False
        elif model_type == "wideresnet":
            self.encode = self.encode_wideresnet
            self.model_helper = FeatureDissimilarityCalculator(
                "wide_resnet50_2", target_layers=["layer2"])
            self.model = self.model_helper.model
            self.can_decode = False
        elif model_type == "wideresnet_rd":
            self.encode = self.encode_wideresnet_rd
            self.model_helper = FeatureDissimilarityCalculator(
                "wide_resnet50_2", target_layers=["layer1", "layer2", "layer3"])
            self.model = self.model_helper.model
            self.can_decode = False
        elif model_type == "dinov3":
            self.encode = self.encode_dinov3
            self.model_helper = transformers.pipeline(
                model="facebook/dinov3-vits16-pretrain-lvd1689m",
                task="image-feature-extraction",
                batch_size=args.batch_size,
            )
            self.model = self.model_helper.model
            self.can_decode = False
        elif model_type == "dino":
            pass
        elif model_type == "dinov3_rd":
            # NOTE: DINOv3
            self.encode = self.encode_dinov3
            self.model_helper = transformers.pipeline(
                # "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                model="facebook/dinov3-vits16-pretrain-lvd1689m",
                task="image-feature-extraction",
                batch_size=args.batch_size
            )
            self.model = self.model_helper.model
            self.can_decode = False
        else:
            raise ValueError("No supported latent model was given!")

    def encode(self, x):
        raise NotImplementedError(
            f"No valid encoding method given for type {self.type}")

    def decode(self, x):
        raise NotImplementedError(
            f"No valid decoding method given for type {self.type}")

    def encode_dinov3(self, x):

        all_h = self.model(x, output_hidden_states=True)
        h = all_h["hidden_states"]
        feat_a = torch.transpose(h[1][:, 5:, :], 1, 2).reshape(
            x.shape[0], 384, 16, 16)
        feat_b = torch.transpose(h[6][:, 5:, :], 1, 2).reshape(
            x.shape[0], 384, 16, 16)
        feat_c = torch.transpose(h[11][:, 5:, :], 1, 2).reshape(
            x.shape[0], 384, 16, 16)
        return [feat_a, feat_b, feat_c]

    def encode_wideresnet(self, x):
        h = self.model_helper._extract_features(x)["layer2"]
        return h

    def encode_wideresnet_rd(self, x):
        h = self.model_helper._extract_features(x)
        return [h["layer1"], h["layer2"], h["layer3"]]

    def encode_effnet(self, x):
        h = self.model(x)
        return h

    def encode_stable_diffusion(self, x):
        h = self.model.encode(x).latent_dist.sample()
        return h

    def decode_stable_diffusion(self, x):
        h = self.model.decode(x).sample
        return h

    def encode_identity(self, x):
        return x

    def decode_identity(self, x):
        return x
