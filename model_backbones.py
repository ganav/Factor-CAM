from __future__ import print_function

import keras
from keras import layers, models

# These 3 should exist in Keras 2.1.6
try:
    from keras.applications import ResNet50, DenseNet121, MobileNet
except ImportError:
    # If your exact version uses different module names, adjust here.
    from keras.applications.resnet50 import ResNet50
    from keras.applications.densenet import DenseNet121
    from keras.applications.mobilenet import MobileNet

# ---------------------------------------------------------------------------
# Helpers: MBConv-style block for EfficientNetB0-like backbone
# ---------------------------------------------------------------------------

def mbconv_block(x,
                 in_channels,
                 out_channels,
                 expansion=1,
                 kernel_size=3,
                 stride=1,
                 se_ratio=0.25,
                 block_name="mbconv"):
    """
    Minimal MBConv block (EfficientNet-style, but simplified).
    - expansion: expansion factor for hidden channels
    - depthwise conv
    - squeeze-and-excitation
    - projection conv
    - residual connection if stride=1 and in/out channels match
    """
    shortcut = x
    bn_axis = 3  # channels_last

    # Expand
    hidden_dim = in_channels * expansion
    if expansion != 1:
        x = layers.Conv2D(hidden_dim, 1, padding="same",
                          use_bias=False, name=block_name + "_expand_conv")(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=block_name + "_expand_bn")(x)
        x = layers.Activation("relu", name=block_name + "_expand_relu")(x)
    else:
        hidden_dim = in_channels

    # Depthwise conv
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same",
                               use_bias=False, name=block_name + "_dwconv")(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  name=block_name + "_dw_bn")(x)
    x = layers.Activation("relu", name=block_name + "_dw_relu")(x)

    # Squeeze-and-Excitation
    if se_ratio is not None and 0 < se_ratio <= 1.0:
        se_channels = max(1, int(in_channels * se_ratio))
        se = layers.GlobalAveragePooling2D(name=block_name + "_se_squeeze")(x)
        se = layers.Reshape((1, 1, hidden_dim),
                            name=block_name + "_se_reshape")(se)
        se = layers.Conv2D(se_channels, 1, activation="relu",
                           name=block_name + "_se_reduce")(se)
        se = layers.Conv2D(hidden_dim, 1, activation="sigmoid",
                           name=block_name + "_se_expand")(se)
        x = layers.Multiply(name=block_name + "_se_excite")([x, se])

    # Project
    x = layers.Conv2D(out_channels, 1, padding="same",
                      use_bias=False, name=block_name + "_project_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  name=block_name + "_project_bn")(x)

    # Residual
    if stride == 1 and in_channels == out_channels:
        x = layers.Add(name=block_name + "_add")([shortcut, x])

    return x


def build_efficientnetb0_like_backbone(input_shape=(224, 224, 3)):
    """
    A small EfficientNet-B0-like backbone using MBConv blocks.
    This is not an exact copy of the official EfficientNet-B0,
    but follows a similar pattern of inverted residual blocks
    with squeeze-and-excitation.
    """
    img_input = keras.Input(shape=input_shape, name="effb0_like_input")
    bn_axis = 3

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding="same",
                      use_bias=False, name="stem_conv")(img_input)
    x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # Stage config (roughly inspired by EfficientNet-B0):
    # (expansion, out_channels, num_blocks, kernel_size, stride)
    stages = [
        # exp, out, blocks, k, s
        (1, 16, 1, 3, 1),
        (6, 24, 2, 3, 2),
        (6, 40, 2, 5, 2),
        (6, 80, 3, 3, 2),
        (6, 112, 3, 5, 1),
        (6, 192, 4, 5, 2),
        (6, 320, 1, 3, 1),
    ]

    in_channels = 32
    block_id = 0
    for (exp, out_ch, num_blocks, k, s) in stages:
        for i in range(num_blocks):
            block_stride = s if i == 0 else 1
            x = mbconv_block(
                x,
                in_channels=in_channels,
                out_channels=out_ch,
                expansion=exp,
                kernel_size=k,
                stride=block_stride,
                se_ratio=0.25,
                block_name="b{}_{}".format(block_id, i),
            )
            in_channels = out_ch
        block_id += 1

    # This x is the final conv feature map (backbone output)
    backbone = models.Model(img_input, x, name="efficientnetb0_like_backbone")
    return backbone


# ---------------------------------------------------------------------------
# Backbones using keras.applications for ResNet, DenseNet, MobileNet
# ---------------------------------------------------------------------------

def build_resnet50_backbone(input_shape=(224, 224, 3)):
    base = ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling=None,
    )
    return base  # last conv feature map is base.output


def build_densenet121_backbone(input_shape=(224, 224, 3)):
    base = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling=None,
    )
    return base


def build_mobilenet_backbone(input_shape=(224, 224, 3)):
    # MobileNet v1
    base = MobileNet(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling=None,
    )
    return base


# ---------------------------------------------------------------------------
# Unified builder: backbone + GAP + softmax for classification (CAM-ready)
# ---------------------------------------------------------------------------

def build_backbone(
    backbone_name,
    num_classes,
    input_shape=(224, 224, 3),
):
    """
    Builds:
      - class_model:   image -> class probabilities [N,num_classes]
      - backbone_model:image -> conv feature map [N,H',W',C]
    for one of:
      - "resnet50"
      - "densenet121"
      - "mobilenet"
      - "efficientnetb0_like"

    These are CAM-ready: last conv feature map + GAP + Dense.
    """
    name = backbone_name.lower()

    if name == "resnet50":
        base = build_resnet50_backbone(input_shape)
        backbone_name_clean = "resnet50"

    elif name == "densenet121":
        base = build_densenet121_backbone(input_shape)
        backbone_name_clean = "densenet121"

    elif name == "mobilenet":
        base = build_mobilenet_backbone(input_shape)
        backbone_name_clean = "mobilenet"

    elif name in ("efficientnetb0", "efficientnetb0_like", "effb0"):
        base = build_efficientnetb0_like_backbone(input_shape)
        backbone_name_clean = "efficientnetb0_like"

    else:
        raise ValueError("Unknown backbone_name: {}".format(backbone_name))

    Fmap = base.output  # last conv feature map
    gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
    class_out = layers.Dense(
        num_classes,
        activation="softmax",
        name=backbone_name_clean + "_pred",
    )(gap)

    class_model = models.Model(
        base.input,
        class_out,
        name=backbone_name_clean + "_classifier",
    )
    backbone_model = models.Model(
        base.input,
        Fmap,
        name=backbone_name_clean + "_backbone",
    )

    return class_model, backbone_model
