"""Loads preprocessed MNIST fashion from the keras dataset."""

import tensorflow as tf

from .registry import register
from .loader_utils import load_from_keras


@register("fashion")
def load_fashion(num_valid=10000, label_smoothing=0):
    """
    Returns preprocessed train, validation, and test sets for MNIST fashion.
    """
    return load_from_keras(
        tf.keras.datasets.fashion_mnist, num_valid, label_smoothing=label_smoothing
    )
