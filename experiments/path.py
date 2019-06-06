"""Utilities for building paths."""
import os

import tensorflow as tf


def get_kernels_path(base_dir, trial, epoch):
    """Builds the kernel save path"""
    return os.path.join(
        base_dir, "trial_{:02d}".format(trial), "kernels", "epoch_{}".format(epoch)
    )


def get_masks_path(base_dir, trial, epoch):
    """Builds the mask save path"""
    return os.path.join(
        base_dir, "trial_{:02d}".format(trial), "masks", "epoch_{}".format(epoch)
    )


def get_training_log_path(base_dir, trial):
    """Builds the training log save path"""
    path = os.path.join(base_dir, "trial_{:02d}".format(trial))
    file_name = "training.log"
    tf.gfile.MakeDirs(path)
    return os.path.join(path, file_name)
