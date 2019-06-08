"""Always reinit the model with the original weights after pruning."""
import copy

from .base_experiment import *
from .pruning import *


def run_trial(trial, dataset, model_name, hparams):
    init_kernels, masks = train_once(
        trial, dataset, model_name, hparams, 0, init_kernels={}, masks={}
    )
    kernels = copy.deepcopy(init_kernels)
    for prune_iter in range(1, hparams["prune_iters"] + 1):

        masks = prune_by_percent(kernels, masks, percents=hparams["percents"])

        # Hold init kernels constant, use original
        kernels, masks = train_once(
            trial,
            dataset,
            model_name,
            hparams,
            prune_iter,
            init_kernels=init_kernels,
            masks=masks,
        )


@register("reinit_orig")
def run(dataset, model_name, hparams):
    for trial in range(hparams["trials"]):
        run_trial(trial, dataset, model_name, hparams)
