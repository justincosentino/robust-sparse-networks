import collections
import os


import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from ..experiments import path, utils
from ..models import registry as model_registry


sns.set_style(
    "ticks",
    {
        "axes.grid": True,
        "font.family": ["serif"],
        "text.usetex": True,
        "legend.frameon": False,
    },
)
sns.set_palette("deep")


YLIMS = {
    "pgd": {"digits": (0.25, 0.65), "fashion": (0.0, 0.40)},
    "fgsm": {"digits": (0.80, 0.99), "fashion": (0.70, 0.90)},
}


def run(hparams):
    exp_results = load_all_exp_results(hparams)
    plot_test_accuracies(hparams, exp_results)
    generate_weight_distributions(hparams, exp_results)


def generate_weight_distributions(
    hparams, exp_results, filter_ids=["03.6", "16.9", "51.3", "100"]
):

    # Get experimental results corresponding to the filter_ids sparsities
    filtered_results = {}
    fixed_trial = "trial_00"
    for experiment, results in exp_results.items():
        for result_id, result in results.items():
            if fixed_trial not in result_id:
                continue
            if result["sparsity"] not in filter_ids and (
                "prune_iter_00" not in result_id and "100" in filter_ids
            ):
                continue
            filtered_results[result_id] = result

        fig, axes = plt.subplots(1, len(filter_ids), figsize=(5 * len(filter_ids), 5))
        for i, item in enumerate(sorted(filtered_results.items())):
            result_id, result = item
            if result["init_kernels"] == {}:
                model_builder = model_registry.get_builder(hparams["model"])
                model = model_builder(kernels={}, show_summary=False)
                result["init_kernels"] = utils.get_masked_kernels(model)
            masked_kernels = utils.apply_masks(result["init_kernels"], result["masks"])
            for layer, kernel in sorted(result["init_kernels"].items()):
                mask = result["masks"][layer]
                active_kernel = kernel[mask == 1]
                plot = sns.distplot(
                    pd.DataFrame(active_kernel.flatten()),
                    hist=False,
                    ax=axes[i],
                    kde_kws={"linewidth": 3},
                )
                plot.set(xlabel="Initial Active Weights", ylabel="Density")
                plot.set_title("{}% Remaining".format(result["sparsity"]))
                plot.set(xlim=(-2.5, 2.5))

                if result["sparsity"] != "100.0":
                    plot.set(ylim=(0, 4.5))

        file_path = os.path.join(
            hparams["analysis_dir"], "dists_{}.svg".format(experiment)
        )
        fig.savefig(file_path, format="svg", bbox_inches="tight")
        plt.clf()
        print("Saving figure to ", file_path)


def get_masks_stats(masks):
    total_weights = sum([v.size for k, v in masks.items()])
    active_weights = sum([int(np.sum(v)) for k, v in masks.items()])
    return active_weights, total_weights


def load_all_exp_results(hparams):
    """
    Load the valid/test logs, init kernels, post kernels, and masks for each
    each pruning iteration of trial of each experiment.
    """
    all_results = {}
    for experiment, experiment_dir in hparams["base_dirs"].items():
        results = collections.defaultdict(lambda: collections.defaultdict(int))
        for trial_dir in trial_iterator(hparams, experiment):
            for prune_iter_dir in prune_iter_iterator(hparams, experiment, trial_dir):
                path = os.path.join(experiment_dir, trial_dir, prune_iter_dir)
                masks_path = os.path.join(path, "masks")
                init_kernels_path = os.path.join(path, "init_kernels")
                post_kernels_path = os.path.join(path, "post_kernels")

                # TODO: remove. try-catch here so that we can run on live exps
                try:
                    masks = utils.restore_array(masks_path)
                    init_kernels = utils.restore_array(init_kernels_path)
                    post_kernels = utils.restore_array(post_kernels_path)
                except:
                    continue

                active_weights, total_weights = get_masks_stats(masks)
                key = "{}/{}".format(trial_dir, prune_iter_dir)
                results[key]["sparsity"] = "{:>04.1f}".format(
                    active_weights / total_weights * 100
                )

                valid_acc_log = pd.read_csv(os.path.join(path, "valid.csv"))
                test_acc_log = pd.read_csv(os.path.join(path, "test.csv"))
                results[key]["valid_acc_log"] = valid_acc_log
                results[key]["test_acc_log"] = test_acc_log
                results[key]["masks"] = masks
                results[key]["init_kernels"] = init_kernels
                results[key]["post_kernels"] = post_kernels

        # for key, value in sorted(results.items()):
        #     print(
        #         "{}: {} -> {:6.3f} | {:6.3f}".format(
        #             key,
        #             value["sparsity"],
        #             value["test_acc_log"]["acc"].iloc[-1],
        #             value["test_acc_log"]["adv_acc"].iloc[-1],
        #         )
        #     )

        all_results[experiment] = results

    return all_results


def plot_test_accuracies(
    hparams,
    exp_results,
    metrics=[
        {"metric": "acc", "label": "Test Accuracy"},
        {"metric": "adv_acc", "label": "Adversarial Test Accuracy"},
    ],
    filter_ids=["01.8", "03.6", "08.7", "16.9", "51.3", "100"],
):
    # Average unpruned results
    unpruned_test_acc = []
    for experiment, results in exp_results.items():
        for trial in trial_iterator(hparams, experiment):
            unpruned_test_acc.append(
                results["{}/prune_iter_00".format(trial)]["test_acc_log"]
            )
            del results["{}/prune_iter_00".format(trial)]
    unpruned_test_acc = pd.concat(unpruned_test_acc).groupby(level=0).mean()

    reinit_label = " (random)"
    for metric in metrics:
        accs = {}
        for experiment, results in exp_results.items():
            for key, value in sorted(results.items()):
                if filter_ids is not None and value["sparsity"] not in filter_ids:
                    continue
                label = "{}{}".format(
                    value["sparsity"],
                    reinit_label if experiment == "reinit_rand" else "",
                )
                accs[label] = value["test_acc_log"][metric["metric"]]
        accs["100"] = unpruned_test_acc[metric["metric"]]

        current_palette = sns.color_palette()
        palette = {
            k: current_palette[filter_ids.index(k.strip(reinit_label))] for k in accs
        }
        dashes = {k: (1, 1) if (reinit_label in k or k == "100") else "" for k in accs}

        accs["iterations"] = value["test_acc_log"]["batch"]
        data_frame = pd.DataFrame.from_dict(accs).set_index("iterations")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        left = sns.lineplot(
            data=data_frame[filter_ids], ax=axes[0], dashes=dashes, palette=palette
        )
        left.set(xlim=(0, 30000))
        left.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]])
        left.set(xlabel="Training Iterations", ylabel=metric["label"])
        left.get_legend().remove()

        right = sns.lineplot(
            data=data_frame.loc[:, data_frame.columns != "100"],
            ax=axes[1],
            dashes=dashes,
            palette=palette,
        )
        right.set(xlim=(0, 30000))
        right.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]])
        right.set(xlabel="Training Iterations", ylabel=metric["label"])
        right.get_legend().remove()

        def parse_legend(x):
            return float(x[0].strip(reinit_label)) + (
                100 if reinit_label in x[0] else 0
            )

        left_handles, left_labels = left.get_legend_handles_labels()
        right_handles, right_labels = right.get_legend_handles_labels()
        handles = left_handles + right_handles
        labels = left_labels + right_labels
        by_label = collections.OrderedDict(
            sorted(zip(labels, handles), key=parse_legend)
        )
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0, 1.0, 1.0, 0.05),
            loc="lower center",
            ncol=11,
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
        )
        plt.tight_layout()

        file_name = "test_{}.svg".format(metric["metric"])
        file_path = os.path.join(hparams["analysis_dir"], file_name)

        fig.savefig(file_path, format="svg", bbox_inches="tight")
        plt.clf()
        print("Saving figure to ", file_path)


def trial_iterator(hparams, experiment):
    with os.scandir(hparams["base_dirs"][experiment]) as it:
        for entry in it:
            if entry.name.startswith(".") or entry.name == "analysis":
                continue
            if entry.is_dir():
                yield entry.name


def prune_iter_iterator(hparams, experiment, trial_dir):
    with os.scandir(os.path.join(hparams["base_dirs"][experiment], trial_dir)) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                yield entry.name
