import pickle

import tensorflow as tf

from ..attacks import registry as attack_registry
from ..data import registry as data_registry
from ..models import registry as model_registry
from .callbacks import *
from .path import *
from .registry import register
from .utils import *


def run_trial(trial, dataset, model_name, hparams, prune_iter=0):
    # Init global state
    sess = init_experiment()

    # Load the specified dataset
    data_loader = data_registry.get_loader(dataset)
    (x_train, y_train, x_valid, y_valid, x_test, y_test) = data_loader(
        num_valid=10000, label_smoothing=0.1
    )

    # Build model
    model_builder = model_registry.get_builder(model_name)
    model = model_builder(l1_reg=hparams["l1_reg"])

    # Save initial kernels and masks
    save_array(
        get_kernels_path(hparams["base_dir"], trial, prune_iter, prefix="init"),
        get_masked_kernels(model),
    )
    save_array(get_masks_path(hparams["base_dir"], trial, prune_iter), get_masks(model))

    # Build attack method and get adv acc and loss
    attack_builder = attack_registry.get_builder(hparams["attack"])
    attack_method, adv_acc, adv_loss = attack_builder(model, sess)

    # Get loss function, depends on if we are using adversarial training
    loss_fn = adv_loss if hparams["adv_train"] else "categorical_crossentropy"

    # Compile the updated model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=hparams["learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", adv_acc],
    )

    # Build callbacks for logging state
    # log_kernels = LogKernels(hparams["base_dir"], trial)
    # log_masks = LogMasks(hparams["base_dir"], trial)
    log_data = EvalEvery(
        hparams["base_dir"],
        trial,
        prune_iter,
        hparams["eval_every"],
        x_valid,
        y_valid,
        x_test,
        y_test,
    )

    # Each iter is a batch of batch_size, so the number of epochs needed to
    # train for train_iters batches is
    epochs = int(hparams["train_iters"] / x_train.shape[0] * hparams["batch_size"])

    # Train the model and save kernels, masks, acc, loss, etc. in callbacks
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=hparams["batch_size"],
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        callbacks=[log_data],
    )

    # TODO save test eval results?
    loss, acc, adv_acc = model.evaluate(
        x_test, y_test, batch_size=hparams["batch_size"], verbose=0
    )
    print("loss: {}; legit acc: {}; adv acc: {}".format(loss, acc, adv_acc))

    # Save initial kernels and masks
    save_array(
        get_kernels_path(hparams["base_dir"], trial, prune_iter, prefix="post"),
        get_masked_kernels(model),
    )


@register("no_pruning")
def run(dataset, model_name, hparams):
    for trial in range(hparams["trials"]):
        run_trial(trial, dataset, model_name, hparams)
