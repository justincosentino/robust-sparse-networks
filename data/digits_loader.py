"""Loads preprocessed MNIST digits from the keras dataset."""

import tensorflow as tf

from .registry import register
from .loader_utils import load_from_keras


@register("digits")
def load_digits(num_valid=10000, label_smoothing=0):
    """
    Returns preprocessed train, validation, and test sets for MNIST digits.
    """
    return load_from_keras(
        tf.keras.datasets.mnist, num_valid, label_smoothing=label_smoothing
    )
