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
        "pgd": {
            "digits": {
                "acc": (0.25, 0.65),
                "adv_acc": (0.25, 0.65),
                "loss": (0.0, 1.5),
            },
            "fashion": {
                "acc": (0.0, 0.40),
                "adv_acc": (0.0, 0.40),
                "loss": (0.0, 1.5),
            },
        },
        "fgsm": {
            "digits": {
                "acc": (0.80, 0.99),
                "adv_acc": (0.80, 0.99),
                "loss": (0.0, 1.5),
            },
            "fashion": {
                "acc": (0.70, 0.90),
                "adv_acc": (0.70, 0.90),
                "loss": (0.0, 1.5),
            },
        },
    }



def run(hparams):
    exp_results = load_all_exp_results(hparams)
    plot_test_accuracies(hparams, exp_results)
    # generate_weight_distributions(hparams, exp_results)
    # plot_early_stoping(hparams, exp_results)
    if hparams["table"]:
        produce_tables(hparams, exp_results)


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
        {"metric": "loss", "label": "Test Loss"},
    ],
    filter_ids=["01.8", "03.6", "08.7", "16.9", "51.3", "100.0"],
):

    test_acc = []
    for experiment, results in exp_results.items():
        for key, value in sorted(results.items()):
            if filter_ids is not None and value["sparsity"] not in filter_ids:
                continue
            if value["sparsity"] == "100.0":
                if not "orig" in experiment:
                    continue
            label = "{} ({})".format(
                value["sparsity"],
                experiment,
            )
            data = value["test_acc_log"]
            data['Experiment'] = label
            test_acc.append(data)
    test_acc = pd.DataFrame(pd.concat(test_acc))

    labels = test_acc['Experiment'].unique()
    
    current_palette = sns.color_palette()
    palette = {
        k: current_palette[filter_ids.index(k.split(" ")[0])] for k in labels
    }
    dashes = {
        k: (1, 1) if ("rand" in k) else (2, 2) if ("none" in k) else "" for k in labels
    }
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        exp_filter = np.logical_or(test_acc['Experiment'].str.contains("orig"), test_acc['Experiment'].str.contains("100"))
        left_data = test_acc[exp_filter]
        left = sns.lineplot(
            x="batch", y=metric["metric"], data=left_data, ax=axes[0], hue="Experiment", style="Experiment", dashes=dashes, palette=palette
        )
        left.set(xlim=(0, 30000))
        left.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]][metric["metric"]])
        left.set(xlabel="Training Iterations", ylabel=metric["label"])
        left.get_legend().remove()

        exp_filter = np.logical_and(np.logical_not(test_acc['Experiment'].str.contains("none")),  np.logical_not(test_acc['Experiment'].str.contains("100")))
        right_data = test_acc[exp_filter]
        right = sns.lineplot(
            x="batch",
            y=metric["metric"],
            data=right_data,
            ax=axes[1],
            hue="Experiment",
            style="Experiment",
            dashes=dashes,
            palette=palette,
        )
        right.set(xlim=(0, 30000))
        right.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]][metric["metric"]])
        right.set(xlabel="Training Iterations", ylabel=metric["label"])
        right.get_legend().remove()

        exp_filter = np.logical_and(np.logical_not(test_acc['Experiment'].str.contains("rand")), np.logical_not(test_acc['Experiment'].str.contains("100")))
        third_data = test_acc[exp_filter]
        third = sns.lineplot(
            x="batch",
            y=metric["metric"],
            data=third_data,
            ax=axes[2],
            hue="Experiment",
            style="Experiment",
            dashes=dashes,
            palette=palette,
        )
        third.set(xlim=(0, 30000))
        third.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]][metric["metric"]])
        third.set(xlabel="Training Iterations", ylabel=metric["label"])
        third.get_legend().remove()
        
        
        def parse_legend(x):
            return float(x[0].split(" ")[0]) + (
                0 if "100" in x[0] else 100 if "orig" in x[0] else 200 if "rand" in x[0] else 300 if "none" in x[0] else 400
            )

        left_handles, left_labels = left.get_legend_handles_labels()
        right_handles, right_labels = right.get_legend_handles_labels()
        third_handles, third_labels = third.get_legend_handles_labels()
        handles = left_handles[1:] + right_handles[1:] + third_handles[1:]
        legend_labels = left_labels[1:] + right_labels[1:] + third_labels[1:]

        def update_label(legend_label):
            if "100" in legend_label:
                return "100"
            label = legend_label.split(" ")
            if "orig" in label[1]:
                label[1] = "(original)"
            elif "rand" in label[1]:
                label[1] = "(random)"
            elif "none" in label[1]:
                label[1] = "(continued)"
            legend_label = " ".join(label)
            return legend_label

        legend_labels = [update_label(label) for label in legend_labels]

        by_label = collections.OrderedDict(
            sorted(zip(legend_labels, handles), key=parse_legend)
        )
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0, 1.0, 1.0, 0.05),
            loc="lower center",
            ncol=8,
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

    nested_dict = lambda: collections.defaultdict(nested_dict)
    table = nested_dict()

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
            
            if adv_training not in table[dataset][sparsity][attack]:
                table[dataset][sparsity][attack][adv_training]["normal"] = []
                table[dataset][sparsity][attack][adv_training]["adv"] = []          

            table[dataset][sparsity][attack][adv_training]["normal"].append(
                value["test_acc_log"]["acc"][target_iteration]
            )
            table[dataset][sparsity][attack][adv_training]["adv"].append(
                value["test_acc_log"]["adv_acc"][target_iteration]
            )

    for dataset in table:
        for sparsity in table[dataset]:
            for attack in table[dataset][sparsity]:
                for adv_training in table[dataset][sparsity][attack]:
                    for test in table[dataset][sparsity][attack][adv_training]:
                        trials = len(table[dataset][sparsity][attack][adv_training][test])
                        mean = np.mean(table[dataset][sparsity][attack][adv_training][test]) * 100
                        std = np.std(table[dataset][sparsity][attack][adv_training][test]) * 100
                        table[dataset][sparsity][attack][adv_training][test] = (mean, std, trials)

    # Re-format dictionary into expected input for Pandas.DataFrame.from_dict
    # Then create DataFrame and produce the latex table
    for dataset in table:
        data_for_pandas = {}
        for sparsity in table[dataset]:
            column = table[dataset][sparsity]
            data_for_pandas[sparsity] = [
                f"{column['fgsm']['false']['normal'][0]:05.2f} $\pm$ {column['fgsm']['false']['normal'][1]:05.2f} ({column['fgsm']['false']['normal'][2]}) / {column['fgsm']['true']['normal'][0]:05.2f} $\pm$ {column['fgsm']['true']['normal'][1]:05.2f} ({column['fgsm']['true']['normal'][2]})",
                f"{column['fgsm']['false']['adv'][0]:05.2f} $\pm$ {column['fgsm']['false']['adv'][1]:05.2f} ({column['fgsm']['false']['adv'][2]}) / {column['fgsm']['true']['adv'][0]:05.2f} $\pm$ {column['fgsm']['true']['adv'][1]:05.2f} ({column['fgsm']['true']['adv'][2]})",
                f"{column['pgd']['false']['normal'][0]:05.2f} $\pm$ {column['pgd']['false']['normal'][1]:05.2f} ({column['pgd']['false']['normal'][2]}) / {column['pgd']['true']['normal'][0]:05.2f} $\pm$ {column['pgd']['true']['normal'][1]:05.2f} ({column['pgd']['true']['normal'][2]})",
                f"{column['pgd']['false']['adv'][0]:05.2f} $\pm$ {column['pgd']['false']['adv'][1]:05.2f} ({column['pgd']['false']['adv'][2]}) / {column['pgd']['true']['adv'][0]:05.2f} $\pm$ {column['pgd']['true']['adv'][1]:05.2f} ({column['pgd']['true']['adv'][2]})",
            ]

        mux = pd.MultiIndex.from_product([["FGSM", "PGD"], ["Natural", "Attack"]])
        df_for_latex = pd.DataFrame.from_dict(
            data_for_pandas, orient="index", columns=mux
        )
        df_for_latex.insert(loc=0, column="Sparsity", value=df_for_latex.index)
        df_for_latex.index = df_for_latex.index.astype(float)
        df_for_latex = df_for_latex.sort_index(ascending=False)
        latex_table = df_for_latex.to_latex(index=False, escape=False, column_format='ccccc', multicolumn_format='c')

        latex_output = os.path.join(hparams["table_output"], "tables")
        print("Tables in", latex_output)
        with open(os.path.join(latex_output, f"{dataset}.txt"), "w") as f:
            f.write(latex_table)


def get_early_stop_iteration(value):
    losses = value["valid_acc_log"]["loss"]
    target_iteration = pd.Series(losses).idxmin()
    return target_iteration


def plot_early_stoping(
    hparams, exp_results, filter_ids=["01.8", "03.6", "08.7", "16.9", "51.3", "100.0"]
):

    # Builds a list of tuples (sparsity, experiment, early_stop_iteration)
    # for creating the DataFrame
    early_stop_iter = []
    for experiment, results in exp_results.items():
        for key, value in sorted(results.items()):
            if filter_ids is not None and value["sparsity"] not in filter_ids:
                continue
            label = experiment
            sparsity = value["sparsity"]
            target_iteration = get_early_stop_iteration(value)
            early_stop_iteration = value["valid_acc_log"]["batch"][target_iteration]
            early_stop_adv_acc = value["valid_acc_log"]["adv_acc"][target_iteration]
            early_stop_acc = value["valid_acc_log"]["acc"][target_iteration]
            early_stop_iter.append((sparsity, label, early_stop_iteration, early_stop_adv_acc, early_stop_acc))

    data_frame = pd.DataFrame(
        early_stop_iter, columns=["Sparsity", "Experiment", "Iteration", "Adversarial_Accuracy", "Test_Accuracy"]
    )
    sorted_index = pd.Series.argsort(data_frame["Sparsity"].astype(float))[::-1]
    data_frame = data_frame.iloc[sorted_index]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    left = sns.lineplot(
        x="Sparsity",
        y="Iteration",
        hue="Experiment",
        ax=axes[0],
        ci="sd",
        sort=False,
        data=data_frame,
    )
    left.set(
        xlabel="Percent of Weights Remaining", ylabel="Early Stop Iteration (Val.)"
    )
    left.get_legend().remove()
    
    right = sns.lineplot(
        x="Sparsity",
        y="Adversarial_Accuracy",
        hue="Experiment",
        ax=axes[1],
        ci="sd",
        sort=False,
        data=data_frame,
    )
    right.set(
        xlabel="Percent of Weights Remaining", ylabel="Early Stop Adv. Acc. (Val.)"
    )
    right.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]]["adv_acc"])
    right.get_legend().remove()

    third = sns.lineplot(
        x="Sparsity",
        y="Test_Accuracy",
        hue="Experiment",
        ax=axes[2],
        ci="sd",
        sort=False,
        data=data_frame,
    )
    third.set(
        xlabel="Percent of Weights Remaining", ylabel="Early Stop Test Acc. (Val.)"
    )
    third.set(ylim=YLIMS[hparams["attack"]][hparams["dataset"]]["adv_acc"])
    third.get_legend().remove()

    left_handles, left_labels = left.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    third_handles, third_labels = third.get_legend_handles_labels()
    handles = left_handles[1:] + right_handles[1:] + third_handles[1:]
    legend_labels = left_labels[1:] + right_labels[1:] + third_labels[1:]

    def update_label(legend_label):
        if "orig" in legend_label:
            legend_label = "original"
        elif "rand" in legend_label:
            legend_label = "random"
        elif "none" in legend_label:
            legend_label = "continued"
        return legend_label

    legend_labels = [update_label(label) for label in legend_labels]

    by_label = dict(
        zip(legend_labels, handles)
    )
    
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0, 1.0, 1.0, 0.05),
        loc="lower center",
        ncol=3,
        borderaxespad=0.0,
        frameon=False,
    )

    plt.tight_layout()

    file_name = "test_{}.svg".format("early_stop_iteration")
    file_path = os.path.join(hparams["analysis_dir"], file_name)

    fig.savefig(file_path, format="svg", bbox_inches="tight")
    plt.clf()
    print("Saving figure to ", file_path)
