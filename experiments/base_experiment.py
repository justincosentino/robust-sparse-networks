import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from ..attacks import registry as attack_registry
from ..data import registry as data_registry
from ..models import registry as model_registry
from .callbacks import *
from .path import *
from .registry import register
from .utils import *


def init_experiment():
    """Initialize seeds for reproducibility and get the current session"""
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    return sess


def train_once(
    trial, dataset, model_name, hparams, prune_iter, init_kernels={}, masks={}
):
    # Init global state
    sess = init_experiment()

    # Load the specified dataset
    data_loader = data_registry.get_loader(dataset)
    (x_train, y_train, x_valid, y_valid, x_test, y_test) = data_loader(
        num_valid=10000, label_smoothing=0.1
    )

    # Build inital model
    model_builder = model_registry.get_builder(model_name)
    model = model_builder(
        kernels=init_kernels,
        masks=masks,
        l1_reg=hparams["l1_reg"],
        show_summary=(prune_iter == 0),
    )

    # Save initial kernels and masks
    init_kernels = get_masked_kernels(model)
    masks = get_masks(model)
    save_array(
        get_kernels_path(hparams["base_dir"], trial, prune_iter, prefix="init"),
        init_kernels,
    )
    save_array(get_masks_path(hparams["base_dir"], trial, prune_iter), masks)

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

    # Build callback for logging state
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

    print(
        "Current model sparsity: {} / {}".format(
            int(sum([np.sum(masks[l]) for l in masks.keys()])),
            int(sum([masks[l].size for l in masks.keys()])),
        )
    )

    # Train the model and save acc, loss, etc. in callbacks
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=hparams["batch_size"],
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        callbacks=[log_data],
        verbose=2,
    )

    # TODO save test eval results?
    loss, acc, adv_acc = model.evaluate(
        x_test, y_test, batch_size=hparams["batch_size"], verbose=0
    )
    print("loss: {}; legit acc: {}; adv acc: {}".format(loss, acc, adv_acc))

    # Save initial kernels and masks
    post_kernels = get_masked_kernels(model)
    save_array(
        get_kernels_path(hparams["base_dir"], trial, prune_iter, prefix="post"),
        get_masked_kernels(model),
    )

    return post_kernels, masks
