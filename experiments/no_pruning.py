from .base_experiment import *


def run_trial(trial, dataset, model_name, hparams):
    train_once(trial, dataset, model_name, hparams, 0, masks={})


@register("no_pruning")
def run(dataset, model_name, hparams):
    for trial in range(hparams["trials"]):
        run_trial(trial, dataset, model_name, hparams)
