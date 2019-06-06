"""Tensorflow keras callbacks used to write network state to files."""
import tensorflow as tf

from .path import get_kernels_path, get_masks_path
from .utils import get_masked_kernels, get_masks, save_array


class LogKernels(tf.keras.callbacks.Callback):
    """Saves kernels of masked layers as .npy files at the end of each epoch."""

    def __init__(self, base_dir, trial):
        super().__init__()
        self.base_dir = base_dir
        self.trial = trial

    def on_epoch_end(self, epoch, logs={}):
        save_array(
            get_kernels_path(self.base_dir, self.trial, epoch),
            get_masked_kernels(self.model),
        )


class LogMasks(tf.keras.callbacks.Callback):
    """Saves masks as .npy files at the end of each epoch."""

    def __init__(self, base_dir, trial):
        super().__init__()
        self.base_dir = base_dir
        self.trial = trial

    def on_epoch_end(self, epoch, logs={}):
        save_array(
            get_kernels_path(self.base_dir, self.trial, epoch), get_masks(self.model)
        )
