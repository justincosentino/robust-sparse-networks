"""Model"""

import numpy as np

import tensorflow as tf

from .registry import register
from .mask import MaskedDense


@register("dense-300-100")
def build_model(name="dense-300-100", kernels={}, masks={}, l1_reg=0.0):
    """
    Returns a sequential keras model of the following form:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    hidden_1 (Dense)             (None, 300)               235500
    _________________________________________________________________
    hidden_2 (Dense)             (None, 100)               30100
    _________________________________________________________________
    output (Dense)               (None, 10)                1000
    =================================================================
    Total params: 266,600
    Trainable params: 266,600
    Non-trainable params: 0
    _________________________________________________________________
    """
    # Convert provided kernels to initializers
    kernel_initializers = {
        layer: tf.constant_initializer(kernel) for layer, kernel in kernels.items()
    }

    mask_initializers = {
        layer: tf.constant_initializer(mask) for layer, mask in masks.items()
    }

    with tf.name_scope(name=name):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(784,), name="input"),
                MaskedDense(
                    300,
                    activation=tf.nn.relu,
                    kernel_initializer=kernel_initializers.get(
                        "hidden_1", "glorot_uniform"
                    ),
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=False,
                    name="hidden_1",
                    mask_initializer=mask_initializers.get("hidden_1", "ones"),
                ),
                MaskedDense(
                    100,
                    activation=tf.nn.relu,
                    kernel_initializer=kernel_initializers.get(
                        "hidden_2", "glorot_uniform"
                    ),
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=False,
                    name="hidden_2",
                    mask_initializer=mask_initializers.get("hidden_2", "ones"),
                ),
                MaskedDense(
                    10,
                    activation=tf.nn.softmax,
                    kernel_initializer=kernel_initializers.get(
                        "output", "glorot_uniform"
                    ),
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=False,
                    name="output",
                    mask_initializer=mask_initializers.get("output", "ones"),
                ),
            ]
        )
        model.summary()
        return model
