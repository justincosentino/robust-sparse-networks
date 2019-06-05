"""Model"""

import tensorflow as tf
import numpy as np


def build_model(name="dense_model", l1_reg=0.0):
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
                tf.keras.layers.Dense(
                    300,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    bias_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=True,
                    name="hidden_1",
                ),
                tf.keras.layers.Dense(
                    100,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    bias_regularizer=tf.keras.regularizers.l1(l=l1_reg),
                    use_bias=True,
                    name="hidden_2",
                ),
                tf.keras.layers.Dense(
                    10, activation=tf.nn.softmax, use_bias=False, name="output"
                ),
            ]
        )
        model.summary()
        return model
