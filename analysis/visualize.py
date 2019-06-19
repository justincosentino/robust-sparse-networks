import collections
import os


import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns


from ..experiments import path, utils


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
    "fgsm": {"digits": (0.80, 0.99), "fashion": (0.70, 0.9)},
}


def run(hparams):
    exp_results = load_all_exp_results(hparams)
    # plot_test_accuracies(hparams, exp_results)
    if hparams["table"]:
        produce_tables(hparams, exp_results)


def generate_weight_distributions():
    pass


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

        for key, value in sorted(results.items()):
            print(
                "{}: {} -> {:6.3f} | {:6.3f}".format(
                    key,
                    value["sparsity"],
                    value["test_acc_log"]["acc"].iloc[-1],
                    value["test_acc_log"]["adv_acc"].iloc[-1],
                )
            )

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

    reinit_label = " (reinit)"
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


def print_all_keys(dictionary, parent_keys=""):
    """
    Auxiliary function to facilitate understanding by recursively printing all keys
    in a dictionary.
    """
    for key in dictionary:
        print(parent_keys, key)
        if isinstance(dictionary[key], dict):
            print_all_keys(dictionary[key], "{} {}".format(parent_keys, key))


def produce_tables(
    hparams, exp_results, filter_ids=["01.8", "03.6", "08.7", "16.9", "51.3", "100.0"]
):
    # Create dictionary for latex tables to easily insert data by index
    table = {}
    for dataset in hparams["dataset"]:
        table[dataset] = {}
        for sparsity in filter_ids:
            table[dataset][sparsity] = {}
            for attack in ["normal", "fgsm", "pgd"]:
                table[dataset][sparsity][attack] = {}
                table[dataset][sparsity][attack] = {}
                table[dataset][sparsity][attack] = {}

    # Get data from experiment results and insert them in the table by index
    for experiment, results in exp_results.items():
        for key, value in sorted(results.items()):
            if filter_ids is not None and value["sparsity"] not in filter_ids:
                continue
            sparsity = value["sparsity"]
            iterations = value["test_acc_log"]["batch"]
            target_iteration = pd.Index(iterations).get_loc(hparams["target_iteration"])
            dataset, attack, adv_training = tuple(experiment.split("_"))
            if (
                value["test_acc_log"]["acc"][target_iteration] is None
                or value["test_acc_log"]["adv_acc"][target_iteration] is None
            ):
                continue

            table[dataset][sparsity]["normal"][adv_training] = round(
                value["test_acc_log"]["acc"][target_iteration] * 100, 2
            )
            table[dataset][sparsity][attack][adv_training] = round(
                value["test_acc_log"]["adv_acc"][target_iteration] * 100, 2
            )

    # Re-format dictionary into expected input for Pandas.DataFrame.from_dict
    # Then create DataFrame and produce the latex table
    for dataset in table:
        data_for_pandas = {}
        for sparsity in table[dataset]:
            column = table[dataset][sparsity]
            data_for_pandas[sparsity] = [
                f"{column['normal']['false']} / {column['normal']['true']}",
                f"{column['fgsm']['false']} / {column['fgsm']['true']}",
                f"{column['pgd']['false']} / {column['pgd']['true']}",
            ]

        df_for_latex = pd.DataFrame.from_dict(
            data_for_pandas, orient="index", columns=["Natural Images", "FGSM", "PGD"]
        )
        df_for_latex.insert(loc=0, column="Sparsity", value=df_for_latex.index)
        df_for_latex.index = df_for_latex.index.astype(float)
        df_for_latex = df_for_latex.sort_index(ascending=False)
        latex_table = df_for_latex.to_latex(index=False)

        latex_output = os.path.join(hparams["table_output"], "tables")
        print("Tables in", latex_output)
        with open(os.path.join(latex_output, f"{dataset}.txt"), "w") as f:
            f.write(latex_table)

