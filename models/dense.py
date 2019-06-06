"""Model"""

import numpy as np

import tensorflow as tf

from .registry import register
from .mask import MaskedDense


@register("dense-300-100")
def build_model(name="dense-300-100", masks={}, l1_reg=0.0):
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
    with tf.name_scope(name=name):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(784,), name="input"),
                MaskedDense(
                    300,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=True,
                    name="hidden_1",
                    mask=masks.get("hidden_1", None),
                ),
                MaskedDense(
                    100,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=True,
                    name="hidden_2",
                    mask=masks.get("hidden_2", None),
                ),
                tf.keras.layers.Dense(
                    10, activation=tf.nn.softmax, use_bias=False, name="output"
                ),
            ]
        )
        model.summary()
        return model
