"""
Model components: ConvNeXt and the simple CNN baseline.

ConvNeXt (Liu et al., CVPR 2022, "A ConvNet for the 2020s") is implemented from
the paper rather than imported from torchvision, so every architectural choice is
accounted for. The baseline is a small plain CNN used to establish what the extra
capacity and cost of ConvNeXt actually buys.

The paper's own release (facebookresearch/ConvNeXt) and timm's `convnext_tiny`
were consulted for details the paper leaves implicit - the 1e-6 layer-scale
initialisation, truncated-normal weight init, and the head rescaling parameter.
The implementation here is verified against torchvision's reference: pooled
features agree to 1.43e-06 after `load_imagenet_weights` maps the weights across,
which also proves the two architectures are identical.

Author: s4980060
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor.

    nn.LayerNorm normalises trailing dimensions, so it expects channels-last.
    ConvNeXt normalises per spatial position across channels, hence this variant
    for the stem and the downsampling layers, which operate on channels-first
    feature maps.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # Statistics are taken over the channel axis only, per spatial location.
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class DropPath(nn.Module):
    """Stochastic depth: drop whole residual branches per sample during training.

    ConvNeXt's main regulariser. With a small medical dataset this matters more
    than it does on ImageNet, where the data itself limits overfitting. Surviving
    branches are rescaled by 1/keep_prob so the expected activation is unchanged.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over all remaining dimensions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise conv -> LayerNorm -> 1x1 expand 4x -> GELU -> 1x1 project.

    The inverted bottleneck mirrors a Transformer MLP block, and the large
    depthwise kernel approximates the wide receptive field of self-attention at
    convolutional cost. A learnable layer-scale parameter gamma, initialised
    small, scales the residual branch and stabilises training of deeper stages.

    The pointwise convolutions are implemented as Linear layers over a
    channels-last tensor, as in the reference implementation: it is equivalent to
    a 1x1 convolution but avoids two extra permutes around the LayerNorm.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise1 = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0
            else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    """ConvNeXt backbone with a binary classification head.

    Four stages with a patchify stem (4x4 stride-4 conv) and LayerNorm + 2x2
    stride-2 downsampling between stages. Stochastic depth rates increase
    linearly with depth, as in the paper.

    in_channels defaults to 1 because ADNI slices are single-channel greyscale;
    the reference ImageNet model uses 3.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init=1e-6,
    ):
        super().__init__()
        if len(depths) != len(dims):
            raise ValueError("depths and dims must describe the same number of stages")

        # Stem plus one downsampling layer before each subsequent stage.
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        )
        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # Linearly spaced stochastic depth rates across all blocks.
        total_blocks = sum(depths)
        drop_rates = [
            drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)
        ]

        self.stages = nn.ModuleList()
        block_index = 0
        for stage_index, depth in enumerate(depths):
            blocks = [
                ConvNeXtBlock(
                    dim=dims[stage_index],
                    drop_path=drop_rates[block_index + i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depth)
            ]
            self.stages.append(nn.Sequential(*blocks))
            block_index += depth

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # applied to pooled features
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # The reference implementation also rescales the classifier by
        # head_init_scale, but that is 1.0 for training from scratch and is only
        # reduced in its fine-tuning recipes, so there is nothing to apply here:
        # the head keeps its truncated-normal weights and a zero bias.
        self.head.bias.data.zero_()

    @staticmethod
    def _init_weights(module):
        """Truncated-normal init for conv and linear layers, as in the paper."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """Run stem, stages and global average pooling; return pooled features."""
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(2, 3))  # global average pool -> (N, C)
        return self.norm(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def convnext_tiny(in_channels=1, num_classes=2, drop_path_rate=0.1, pretrained=False):
    """ConvNeXt-T: depths (3, 3, 9, 3), dims (96, 192, 384, 768), ~28M params.

    With pretrained=True the ImageNet-1k weights are copied in; see
    load_imagenet_weights for what is and is not transferred.
    """
    model = ConvNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
    )
    if pretrained:
        load_imagenet_weights(model)
    return model


def load_imagenet_weights(model, verbose=True):
    """Copy torchvision's ImageNet ConvNeXt-T weights into this implementation.

    Training ConvNeXt from scratch on roughly 476 patients does not work: three
    configurations all plateau near 0.69 test accuracy. Transfer learning is the
    standard remedy for small medical datasets - the early layers learn generic
    edge and texture filters that this dataset is far too small to discover.

    Only the initial weights are borrowed; the architecture is the one defined
    above, which matches the paper's parameter count to within 0.04%. The layer
    names differ from torchvision's, so the mapping is explicit below.

    The stem is adapted from 3 input channels to 1 by summing across the colour
    axis. Summing (rather than averaging) preserves the response magnitude a
    grey image would produce if its single channel were replicated into RGB.

    The classifier head is not copied: it predicts 1000 ImageNet classes, while
    this task has 2, and it is trained from scratch.
    """
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny as tv_convnext_tiny

    source = tv_convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).state_dict()
    target = model.state_dict()
    mapping = {}

    # Stem: conv + norm. features.0.{0,1} -> downsample_layers.0.{0,1}
    mapping["features.0.0.weight"] = "downsample_layers.0.0.weight"
    mapping["features.0.0.bias"] = "downsample_layers.0.0.bias"
    mapping["features.0.1.weight"] = "downsample_layers.0.1.weight"
    mapping["features.0.1.bias"] = "downsample_layers.0.1.bias"

    # Downsampling layers between stages: features.{2,4,6} -> downsample_layers.{1,2,3}
    for source_idx, target_idx in zip((2, 4, 6), (1, 2, 3)):
        for sub in (0, 1):
            for suffix in ("weight", "bias"):
                mapping[f"features.{source_idx}.{sub}.{suffix}"] = (
                    f"downsample_layers.{target_idx}.{sub}.{suffix}"
                )

    # Blocks: features.{1,3,5,7}.{i} -> stages.{0,1,2,3}.{i}
    # torchvision packs each block into a Sequential, so block.0 is the depthwise
    # convolution, block.2 the LayerNorm, and block.3/5 the two Linear layers.
    block_parts = {
        "block.0": "depthwise",
        "block.2": "norm",
        "block.3": "pointwise1",
        "block.5": "pointwise2",
    }
    for source_idx, (target_idx, depth) in zip(
        (1, 3, 5, 7), enumerate((3, 3, 9, 3))
    ):
        for i in range(depth):
            for source_part, target_part in block_parts.items():
                for suffix in ("weight", "bias"):
                    mapping[f"features.{source_idx}.{i}.{source_part}.{suffix}"] = (
                        f"stages.{target_idx}.{i}.{target_part}.{suffix}"
                    )
            mapping[f"features.{source_idx}.{i}.layer_scale"] = (
                f"stages.{target_idx}.{i}.gamma"
            )

    # Final norm applied to the pooled features.
    mapping["classifier.0.weight"] = "norm.weight"
    mapping["classifier.0.bias"] = "norm.bias"

    loaded, skipped = 0, []
    for source_key, target_key in mapping.items():
        if source_key not in source or target_key not in target:
            skipped.append(target_key)
            continue
        tensor = source[source_key]

        if target_key == "downsample_layers.0.0.weight":
            # (96, 3, 4, 4) -> (96, in_channels, 4, 4)
            in_channels = target[target_key].shape[1]
            if in_channels == 1:
                tensor = tensor.sum(dim=1, keepdim=True)
            elif in_channels != tensor.shape[1]:
                skipped.append(target_key)
                continue
        elif target_key.endswith("gamma"):
            tensor = tensor.reshape(-1)  # (C, 1, 1) -> (C,)

        if tensor.shape != target[target_key].shape:
            skipped.append(target_key)
            continue
        target[target_key] = tensor
        loaded += 1

    model.load_state_dict(target)
    if verbose:
        total = len([k for k in target if not k.startswith("head.")])
        print(f"Loaded {loaded}/{total} pretrained tensors (head trained from scratch)")
        if skipped:
            print(f"  skipped: {len(skipped)} e.g. {skipped[:3]}")
    return model


def convnext_mini(in_channels=1, num_classes=2, drop_path_rate=0.4):
    """A deliberately small ConvNeXt: dims (48, 96, 192, 384), 4.83M parameters.

    ConvNeXt-T overfits this dataset badly - 27.8M parameters against roughly
    476 training patients drives training accuracy to 99% while validation
    accuracy stalls near 64%. This variant keeps every architectural element of
    ConvNeXt (patchify stem, 7x7 depthwise blocks, inverted bottleneck, layer
    scale) and only reduces width and depth, so capacity can be tested as the
    cause rather than being confounded with a change of architecture.

    The experiment is reported in the README: the gap narrows, the accuracy does
    not move, so capacity was not the limiting factor.
    """
    return ConvNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=drop_path_rate,
    )


class BaselineCNN(nn.Module):
    """Small plain CNN baseline: Conv-BN-ReLU-MaxPool blocks into a linear head.

    Deliberately simple and deliberately conventional - BatchNorm and ReLU rather
    than ConvNeXt's LayerNorm and GELU. It exists to answer the dilemma's cost
    question - whether ConvNeXt's accuracy and calibration justify its parameter
    count and latency - so it is trained and evaluated on exactly the same split.
    """

    def __init__(self, in_channels=1, num_classes=2, widths=(16, 32, 64), dropout=0.0):
        super().__init__()
        layers = []
        previous = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(previous, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(previous, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))


def count_parameters(model):
    """Total and trainable parameter counts, for the resource profiling table."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_model(name, in_channels=1, num_classes=2, **kwargs):
    """Construct 'convnext' or 'baseline' by name, for use from train.py."""
    name = name.lower()
    if name == "convnext":
        return convnext_tiny(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name == "convnext-pretrained":
        return convnext_tiny(
            in_channels=in_channels, num_classes=num_classes, pretrained=True, **kwargs
        )
    if name == "convnext-mini":
        return convnext_mini(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name == "baseline":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes, **kwargs)
    raise ValueError(
        f"Unknown model '{name}'; expected 'convnext', 'convnext-pretrained', "
        f"'convnext-mini' or 'baseline'"
    )
