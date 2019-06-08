"""Tensorflow keras callbacks used to write network state to files."""
import pandas as pd
import tensorflow as tf

from .path import *
from .utils import *


# class LogKernels(tf.keras.callbacks.Callback):
#     """Saves kernels of masked layers as .npy files at the end of each epoch."""

#     def __init__(self, base_dir, trial):
#         super().__init__()
#         self.base_dir = base_dir
#         self.trial = trial

#     def on_epoch_end(self, epoch, logs={}):
#         save_array(
#             get_kernels_path(self.base_dir, self.trial, epoch),
#             get_masked_kernels(self.model),
#         )


# class LogMasks(tf.keras.callbacks.Callback):
#     """Saves masks as .npy files at the end of each epoch."""

#     def __init__(self, base_dir, trial):
#         super().__init__()
#         self.base_dir = base_dir
#         self.trial = trial

#     def on_epoch_end(self, epoch, logs={}):
#         save_array(
#             get_kernels_path(self.base_dir, self.trial, epoch), get_masks(self.model)
#         )


class EvalEvery(tf.keras.callbacks.Callback):
    """Logs train, validation, and test accuracy every N iterations."""

    def __init__(
        self, base_dir, trial, prune_iter, eval_every, x_valid, y_valid, x_test, y_test
    ):
        super().__init__()
        # Used to determine where to log
        self.base_dir = base_dir
        self.trial = trial
        self.prune_iter = prune_iter

        # Used to determine when to log
        self.eval_every = eval_every
        self.curr_iter = 0

        # Validation and test data
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

        # Lists to store logs
        self.train_logs = []
        self.valid_logs = []
        self.test_logs = []

    def on_train_batch_end(self, batch, logs={}):
        if batch % self.eval_every != 0:
            return

        # Handle training data
        train_log = {
            "batch": self.curr_iter,
            "loss": logs["loss"],
            "acc": logs["acc"],
            "adv_acc": logs["adv_acc"],
        }
        self.train_logs.append(logs)

        # Handle validation data
        loss, acc, adv_acc = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        valid_log = {
            "batch": self.curr_iter,
            "loss": loss,
            "acc": acc,
            "adv_acc": adv_acc,
        }
        self.valid_logs.append(valid_log)

        # Handle test data
        loss, acc, adv_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        test_log = {
            "batch": self.curr_iter,
            "loss": loss,
            "acc": acc,
            "adv_acc": adv_acc,
        }
        self.test_logs.append(test_log)

        # Increment this last because we start at batch 0
        self.curr_iter += self.eval_every

    def on_train_end(self, logs={}):
        # Write everything to files~~
        columns = ["batch", "loss", "acc", "adv_acc"]
        train_df = pd.DataFrame(self.train_logs, columns=columns)
        train_df.to_csv(
            get_log_path(self.base_dir, self.trial, self.prune_iter, "train"),
            index=None,
            header=True,
        )
        valid_df = pd.DataFrame(self.valid_logs, columns=columns)
        valid_df.to_csv(
            get_log_path(self.base_dir, self.trial, self.prune_iter, "valid"),
            index=None,
            header=True,
        )
        test_df = pd.DataFrame(self.test_logs, columns=columns)
        test_df.to_csv(
            get_log_path(self.base_dir, self.trial, self.prune_iter, "test"),
            index=None,
            header=True,
        )
