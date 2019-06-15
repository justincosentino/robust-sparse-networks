"""Experiment utilities."""
import os

import numpy as np

import tensorflow as tf

from ..models.mask import MaskedDense


def save_array(filename, array_dict):
    """
    Saves an numpy array.

    array_dict is a dictionary where each key is the name of a tensor and each
    value is a numpy array. filename is created as a directory and each item
    of array_dict is saved as a separate npy file within that directory.

    Args:
        filename: A directory in which the network should be saved.
        array_dict: A dictionary where each key is the name of a tensor and each
        value is a numpy array. This is the dictionary of values that is to be
        saved.
    """
    tf.gfile.MakeDirs(filename)

    for k, v in array_dict.items():
        with tf.gfile.GFile(os.path.join(filename, k + ".npy"), "w") as fp:
            np.save(fp, v)


def restore_array(filename):
    """
    Loads an array in the form stored by save_array.

    filename is the name of a directory containing many npy files. Each npy file
    is loaded into a numpy array and added to a dictionary. The dictionary key
    is the name of the file (without the .npy extension). This dictionary is
    returned.

    Args:
        filename: The name of the directory where the npy files are saved.

    Returns:
        A dictionary where each key is the name of a npy file within filename and
        each value is the corresponding numpy array stored in that file. This
        dictionary is of the same form as that passed to save_network.

    Raises:
        ValueError: If filename does not exist.
    """
    if not tf.gfile.Exists(filename):
        raise ValueError("Filename {} does not exist.".format(filename))

    array_dict = {}

    for basename in tf.gfile.ListDirectory(filename):
        name = basename.split(".")[0]
        with tf.gfile.GFile(os.path.join(filename, basename), "rb") as fp:
            array_dict[name] = np.load(fp)

    return array_dict


def get_masked_kernels(model):
    """Get all MaskedDense kernels"""
    kernels = {}
    for layer in model.layers:
        if not isinstance(layer, MaskedDense):
            continue
        # TODO: this seems like risky business
        kernels[layer.name] = layer.get_weights()[0]
    return kernels


def get_masks(model):
    """Get all MaskedDense masks"""
    masks = {}
    for layer in model.layers:
        if not isinstance(layer, MaskedDense):
            continue
        # TODO: this seems like risky business, will break with biases
        weights = layer.get_weights()
        if len(weights) == 1:
            # No masks applied
            continue
        # Using -1 because we might have biases
        masks[layer.name] = weights[-1]

    return masks


def apply_masks(kernels, masks):
    """Apply masks to their associated kernels"""
    applied_masks = {}
    for layer, kernel in kernels.items():
        applied_masks[layer] = np.multiply(kernel, masks[layer])
    return applied_masks
