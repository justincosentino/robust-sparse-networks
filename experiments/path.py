"""Utilities for building paths."""
import os

import tensorflow as tf


def get_kernels_path(base_dir, trial, prune_iter, prefix=""):
    """Builds the kernel save path"""
    prefix = prefix = "{}_".format(prefix) if prefix else prefix
    return os.path.join(
        base_dir,
        "trial_{:02d}".format(trial),
        "prune_iter_{:02d}".format(prune_iter),
        "{}kernels".format(prefix),
    )


def get_masks_path(base_dir, trial, prune_iter):
    """Builds the mask save path"""
    return os.path.join(
        base_dir,
        "trial_{:02d}".format(trial),
        "prune_iter_{:02d}".format(prune_iter),
        "masks",
    )


def get_log_path(base_dir, trial, prune_iter, log_type):
    """Builds the training log save path"""
    path = os.path.join(
        base_dir, "trial_{:02d}".format(trial), "prune_iter_{:02d}".format(prune_iter)
    )
    file_name = "{}.csv".format(log_type)
    tf.gfile.MakeDirs(path)
    return os.path.join(path, file_name)
