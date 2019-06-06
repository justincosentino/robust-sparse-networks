"""
Utilities used to load and preprocess MNIST keras datasets.
"""

import numpy as np


def load_from_keras(keras_dataset, num_valid=10000, label_smoothing=0):
    """
    Loads images from a keras dataset, normalizes the images, and breaks them
    into train, validation, and test subsets.
    """
    # Load data from keras
    (x_train, y_train), (x_test, y_test) = keras_dataset.load_data()

    # Preprocess images and labels
    x_train, x_test = _preprocess_images(x_train), _preprocess_images(x_test)
    y_train, y_test = (
        _preprocess_labels(y_train, label_smoothing=label_smoothing),
        _preprocess_labels(y_test),
    )

    # Further break training data into train / validation sets
    if num_valid < 0:
        raise Exception(
            "invalid num_valid: must be >= 0, specified {}".format(num_valid)
        )
    (x_train, x_valid) = x_train[num_valid:], x_train[:num_valid]
    (y_train, y_valid) = y_train[num_valid:], y_train[:num_valid]

    return (x_train, y_train, x_valid, y_valid, x_test, y_test)


def _preprocess_images(images):
    """
    Converts MNIST images from (?, 28, 28, 1) to (?, 784) and normalizes pixel
    values from [0, 255] grayscale to a [0, 1] representation.
    """
    images = images / 255.0
    images = images.reshape(images.shape[0], 784)
    images = images.astype("float32")
    assert images.shape[1] == 784
    return images


def _preprocess_labels(labels, label_smoothing=0):
    """Converts labels to one-hot float32. Smooths labels if specified"""
    n_labels = np.max(labels) + 1
    labels = np.eye(n_labels)[labels]
    if label_smoothing:
        labels -= label_smoothing * (labels - 1.0 / n_labels)
    return labels.astype("float32")
