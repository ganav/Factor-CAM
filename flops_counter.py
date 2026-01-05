import numpy as np
import keras
from keras.layers import (
    Conv2D, Dense, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation,
    MaxPooling2D, AveragePooling2D,
    Flatten, GlobalAveragePooling2D
)


def count_flops(model, input_shape, include_bn=False):
    """
    Count FLOPs for Keras model (TF 1.x compatible)

    Args:
        model: keras Model
        input_shape: (H, W, C) or (N,) for vector input
        include_bn: whether to include BN FLOPs

    Returns:
        total_flops (int)
    """

    total_flops = 0
    cur_shape = input_shape

    for layer in model.layers:

        # ==============================
        # Conv2D
        # ==============================
        if isinstance(layer, Conv2D):
            kh, kw = layer.kernel_size
            sh, sw = layer.strides
            filters = layer.filters
            padding = layer.padding

            h, w, c = cur_shape

            if padding == 'same':
                out_h = int(np.ceil(float(h) / sh))
                out_w = int(np.ceil(float(w) / sw))
            else:  # valid
                out_h = int(np.ceil(float(h - kh + 1) / sh))
                out_w = int(np.ceil(float(w - kw + 1) / sw))

            flops = 2 * kh * kw * c * filters * out_h * out_w
            total_flops += flops

            cur_shape = (out_h, out_w, filters)

        # ==============================
        # Depthwise Conv
        # ==============================
        elif isinstance(layer, DepthwiseConv2D):
            kh, kw = layer.kernel_size
            sh, sw = layer.strides
            h, w, c = cur_shape

            if layer.padding == 'same':
                out_h = int(np.ceil(float(h) / sh))
                out_w = int(np.ceil(float(w) / sw))
            else:
                out_h = int(np.ceil(float(h - kh + 1) / sh))
                out_w = int(np.ceil(float(w - kw + 1) / sw))

            flops = 2 * kh * kw * c * out_h * out_w
            total_flops += flops

            cur_shape = (out_h, out_w, c)

        # ==============================
        # Separable Conv
        # ==============================
        elif isinstance(layer, SeparableConv2D):
            kh, kw = layer.kernel_size
            sh, sw = layer.strides
            filters = layer.filters
            h, w, c = cur_shape

            if layer.padding == 'same':
                out_h = int(np.ceil(float(h) / sh))
                out_w = int(np.ceil(float(w) / sw))
            else:
                out_h = int(np.ceil(float(h - kh + 1) / sh))
                out_w = int(np.ceil(float(w - kw + 1) / sw))

            # depthwise + pointwise
            flops = (
                2 * kh * kw * c * out_h * out_w +
                2 * c * filters * out_h * out_w
            )

            total_flops += flops
            cur_shape = (out_h, out_w, filters)

        # ==============================
        # Dense
        # ==============================
        elif isinstance(layer, Dense):
            units = layer.units
            in_units = cur_shape if isinstance(cur_shape, int) else np.prod(cur_shape)

            flops = 2 * in_units * units
            total_flops += flops

            cur_shape = units

        # ==============================
        # BatchNorm (optional)
        # ==============================
        elif isinstance(layer, BatchNormalization):
            if include_bn:
                flops = 2 * np.prod(cur_shape)
                total_flops += flops

        # ==============================
        # Pooling
        # ==============================
        elif isinstance(layer, (MaxPooling2D, AveragePooling2D)):
            sh, sw = layer.strides
            h, w, c = cur_shape
            cur_shape = (h // sh, w // sw, c)

        # ==============================
        # Global Avg Pool
        # ==============================
        elif isinstance(layer, GlobalAveragePooling2D):
            h, w, c = cur_shape
            total_flops += h * w * c
            cur_shape = c

        # ==============================
        # Flatten
        # ==============================
        elif isinstance(layer, Flatten):
            cur_shape = np.prod(cur_shape)

        # ==============================
        # Activation / ReLU
        # ==============================
        elif isinstance(layer, (Activation)):
            # negligible, skip
            pass

        # ==============================
        # Unknown layer
        # ==============================
        else:
            pass

    return int(total_flops)
