import torch
from torch import nn
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
import math
from typing import Callable
from functools import partial

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class OCBE(nn.Module):
    """One-Class Bottleneck Embedding module with timestep conditioning.

    This module implements a bottleneck layer that compresses multi-scale features into a
    compact representation, conditioned on timestep embeddings.

    Args:
        block (Bottleneck | BasicBlock): Block type that determines expansion factor.
        layers (int): Number of OCE layers to create after multi-scale feature fusion.
        emb_channels (int): Number of timestep embedding channels.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        width_per_group (int, optional): Number of channels in each layer. Defaults to 64.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization layer. Defaults to None.
        use_scale_shift_norm (bool, optional): Use scale-shift normalization. Defaults to False.
    """

    def __init__(
        self,
        block: Bottleneck | BasicBlock,
        layers: int,
        emb_channels: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32) # nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        # print("BOTTLENECK IN OCBE")
        # print(self.bn_layer)
        
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)
        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        # Timestep embedding layers for each processing stage
        bottleneck_channels = 512 * block.expansion
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * bottleneck_channels if use_scale_shift_norm else bottleneck_channels,
            ),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[Bottleneck | BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        # First is input layer
        layers = []
        layers.append(
            block(
                self.inplanes * 3, # Input
                planes,            # 
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        )
        self.inplanes = planes * block.expansion
        # n = 3 - 1 bottleneck layers
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor], emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Args:
            features (list[torch.Tensor]): List of 3 feature tensors from different scales.
            emb (torch.Tensor): Timestep embeddings of shape (B, emb_channels).

        Returns:
            torch.Tensor: Compressed bottleneck representation conditioned on timesteps.
        """
        # Process features as before
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        
        # print(feature0.shape, feature1.shape, features[2].shape, feature_cat.shape)
        
        # Apply bottleneck layers
        h = self.bn_layer(feature_cat)
        
        # Apply timestep conditioning
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        return h.contiguous()


class TimestepConditionedDecoderBottleneck(nn.Module):
    """Bottleneck block for the decoder network with timestep conditioning."""
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        emb_channels: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32) # nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Convolution layers
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(
                width,
                width,
                kernel_size=2,
                stride=stride,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        
        # Timestep embedding layers
        out_channels = planes * self.expansion
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )

    def forward(self, batch: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep conditioning."""
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        
        # Apply timestep conditioning before final batch norm
        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            out_norm = self.bn3(out)
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = out_norm * (1 + scale) + shift
        else:
            out = self.bn3(out + emb_out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        return self.relu(out)


class ResNetDecoder(nn.Module):
    """Decoder ResNet model with timestep conditioning."""

    def __init__(
        self,
        block: type[TimestepConditionedDecoderBottleneck],
        layers: list[int],
        emb_channels: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32) # nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.emb_channels = emb_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, TimestepConditionedDecoderBottleneck):
                    nn.init.constant_(module.bn3.weight, 0)

    def _make_layer(
        self,
        block: type[TimestepConditionedDecoderBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=2,
                    stride=stride,
                    groups=self.groups,
                    bias=False,
                    dilation=self.dilation,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, 
                planes, 
                self.emb_channels,
                stride, 
                upsample, 
                self.groups, 
                self.base_width, 
                previous_dilation, 
                norm_layer,
                self.use_scale_shift_norm
            ),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    self.emb_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor, emb: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the timestep-conditioned decoder ResNet.

        Args:
            batch (torch.Tensor): Input tensor of shape (B, C, H, W).
            emb (torch.Tensor): Timestep embeddings of shape (B, emb_channels).

        Returns:
            list[torch.Tensor]: List of feature tensors at different scales.
        """
        # Pass timestep embeddings through each layer
        feature_a = self._forward_layer(self.layer1, batch, emb)
        feature_b = self._forward_layer(self.layer2, feature_a, emb) 
        feature_c = self._forward_layer(self.layer3, feature_b, emb)

        return [feature_c, feature_b, feature_a]
    
    def _forward_layer(self, layer: nn.Sequential, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward through a layer with timestep conditioning."""
        for block in layer:
            x = block(x, emb)
        return x


def _resnet(block, layers: list[int], emb_channels: int, **kwargs) -> ResNetDecoder:
    return ResNetDecoder(block, layers, emb_channels, **kwargs)


class OCBESingleScale(nn.Module):
    """OCBE for N same-scale input feature maps."""
    
    def __init__(
        self,
        block: Bottleneck | BasicBlock,
        layers: int,
        num_input_maps: int,
        input_channels: int,
        emb_channels: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.use_scale_shift_norm = use_scale_shift_norm
        
        total_in_channels = num_input_maps * input_channels  # 13 * 256 = 3328
        
        # Project to intermediate size, then to bottleneck
        intermediate_channels = 512 * block.expansion  # 2048
        self.input_proj = nn.Sequential(
            conv3x3(total_in_channels, intermediate_channels, stride=1),
            norm_layer(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        
        self.inplanes = intermediate_channels
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        
        bottleneck_channels = 512 * block.expansion
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * bottleneck_channels if use_scale_shift_norm else bottleneck_channels,
            ),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                       self.groups, self.base_width, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        layers.extend([
            block(self.inplanes, planes, groups=self.groups, 
                  base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer)
            for _ in range(1, blocks)
        ])
        return nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor], emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat(features, dim=1)  # [B, 3328, 16, 16]
        x = self.input_proj(x)          # [B, 2048, 16, 16]
        h = self.bn_layer(x)            # [B, 2048, 8, 8]
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        return h.contiguous()


class ResNetDecoderSingleScale(nn.Module):
    """Decoder that outputs N same-scale feature maps."""

    def __init__(
        self,
        block: type[TimestepConditionedDecoderBottleneck],
        layers: list[int],
        num_output_maps: int,
        output_channels: int,
        emb_channels: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)
        self._norm_layer = norm_layer
        self.emb_channels = emb_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_output_maps = num_output_maps
        self.output_channels = output_channels

        self.inplanes = 512 * block.expansion  # 2048
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Single upsample layer: 8x8 -> 16x16
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        
        # Final projection to N output maps
        total_out_channels = num_output_maps * output_channels  # 13 * 256 = 3328
        self.output_proj = nn.Sequential(
            conv3x3(self.inplanes, total_out_channels),
            norm_layer(total_out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=2, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, self.emb_channels, stride, upsample,
                       self.groups, self.base_width, self.dilation, norm_layer,
                       self.use_scale_shift_norm)]
        self.inplanes = planes * block.expansion
        layers.extend([
            block(self.inplanes, planes, self.emb_channels,
                  groups=self.groups, base_width=self.base_width,
                  dilation=self.dilation, norm_layer=norm_layer,
                  use_scale_shift_norm=self.use_scale_shift_norm)
            for _ in range(1, blocks)
        ])
        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor, emb: torch.Tensor) -> list[torch.Tensor]:
        # batch: [B, 2048, 8, 8]
        x = self._forward_layer(self.layer1, batch, emb)  # [B, 1024, 16, 16]
        x = self.output_proj(x)  # [B, 3328, 16, 16]
        
        # Split into list of N feature maps
        return list(x.chunk(self.num_output_maps, dim=1))
    
    def _forward_layer(self, layer, x, emb):
        for block in layer:
            x = block(x, emb)
        return x


class RDWrapperSingleScale(torch.nn.Module):
    """Wrapper for single-scale multi-map reconstruction."""
    
    def __init__(
        self, 
        num_feature_maps: int,
        feature_channels: int,
        model_channels: int, 
        args,
        width_per_group: int = 128,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.args = args
        self.model_channels = model_channels
        
        timestep_channels = model_channels
        emb_channels = timestep_channels if not self.args.cond else timestep_channels * 2
            
        self.ocbe = OCBESingleScale(
            Bottleneck, 
            layers=3,
            num_input_maps=num_feature_maps,
            input_channels=feature_channels,
            emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            width_per_group=width_per_group // 2,
        )
        
        self.decoder = ResNetDecoderSingleScale(
            TimestepConditionedDecoderBottleneck,
            layers=[3],  # Single layer for 8x8 -> 16x16
            num_output_maps=num_feature_maps,
            output_channels=feature_channels,
            emb_channels=emb_channels,
            width_per_group=width_per_group,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, timestep_channels),
            nn.SiLU(),
            nn.Linear(timestep_channels, timestep_channels),
        )
        
        if self.args.cond:
            self.view_embed = nn.Sequential(
                nn.Embedding(5, model_channels),
                nn.Linear(model_channels, timestep_channels),
                nn.SiLU(),
                nn.Linear(timestep_channels, timestep_channels),
            )
        
    def forward(self, x: list[torch.Tensor], t: torch.Tensor, extra=None) -> list[torch.Tensor]:
        """
        Args:
            x: List of N tensors, each [B, C, H, W] (e.g., 13 x [B, 256, 16, 16])
            t: Timesteps [B] or scalar
            extra: Optional view conditioning
            
        Returns:
            List of N tensors matching input shapes
        """
        if len(t.shape) == 0 and extra is not None:
            t = torch.repeat_interleave(t[None], extra.shape[0], 0)
        
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        if extra is not None:
            view_emb = self.view_embed(extra)
            emb = torch.concat((emb, view_emb), dim=1)
        
        h = self.ocbe(x, emb)
        return self.decoder(h, emb)


class RDWrapper(torch.nn.Module):
    def __init__(self, width_per_group: int, model_channels: int, args, use_scale_shift_norm: bool = False):
        super().__init__()
        
        # NOTE: width_per_group was originally 128
        self.args = args
        timestep_channels = model_channels
        emb_channels = timestep_channels if not self.args.cond else timestep_channels * 2
        
        self.model_channels = model_channels
        
        self.ocbe = OCBE(
            Bottleneck, 
            3, 
            emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            width_per_group=width_per_group // 2, # 64
        )
        self.model = _resnet(
            TimestepConditionedDecoderBottleneck, 
            [3, 4, 6, 3], 
            emb_channels=emb_channels,
            width_per_group=width_per_group, # 128,
            use_scale_shift_norm=use_scale_shift_norm
        )
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, timestep_channels),
            nn.SiLU(),
            nn.Linear(timestep_channels, timestep_channels),
        )
        
        if self.args.cond:
            self.view_embed = nn.Sequential(
                nn.Embedding(5, model_channels),
                nn.Linear(model_channels, timestep_channels),
                nn.SiLU(),
                nn.Linear(timestep_channels, timestep_channels),
            )
        
    def forward(self, x, timesteps, extra=None):
        if len(t.shape) == 0 and extra is not None:
            t = torch.repeat_interleave(t[None], extra.shape[0], 0)
            
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if extra is not None:
            view_emb = self.view_embed(extra)
            emb = torch.concat((emb, view_emb), dim=1)
        
        
        h = self.ocbe(x, emb)
        return self.model(h, emb)
    
    def forward(self, x, t, extra=None):
        
        if len(t.shape) == 0 and extra is not None:
            t = torch.repeat_interleave(t[None], extra.shape[0], 0)
        
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        if extra is not None:
            view_emb = self.view_embed(extra)
            emb = torch.concat((emb, view_emb), dim=1)
        
        h = self.ocbe(x, emb)
        return self.model(h, emb)