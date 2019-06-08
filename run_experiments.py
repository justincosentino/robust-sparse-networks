"""Run experiments with CLI"""
import argparse
import os
import shutil

from .experiments import registry


def init_flags():
    """Init command line flags used for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Runs experiments to find robust, sparse networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials",
        metavar="trials",
        type=int,
        nargs=1,
        default=[10],
        help="number trials per experiment",
    )
    parser.add_argument(
        "--train_iters",
        metavar="train_iters",
        type=int,
        nargs=1,
        default=[50000],
        help="number of training iterations",
    )
    parser.add_argument(
        "--prune_iters",
        metavar="prune_iters",
        type=int,
        nargs=1,
        default=[20],
        help="number of pruning iterations",
    )
    parser.add_argument(
        "--eval_every",
        metavar="eval_every",
        type=int,
        nargs=1,
        default=[100],
        help="number of iterations to eval on validation set",
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        type=int,
        nargs=1,
        default=[60],
        help="batch size",
    )
    parser.add_argument(
        "--valid_size",
        metavar="valid_size",
        type=int,
        nargs=1,
        default=[10000],
        help="validation set size",
    )
    parser.add_argument(
        "--dataset",
        metavar="dataset",
        type=str,
        nargs=1,
        default=["digits"],
        choices=["digits", "fashion"],
        help="source dataset",
    )
    parser.add_argument(
        "--model",
        metavar="model",
        type=str,
        nargs=1,
        default=["dense-300-100"],
        choices=["dense-300-100"],
        help="model type",
    )
    parser.add_argument(
        "--experiment",
        metavar="experiment",
        type=str,
        nargs=1,
        default=["no_pruning"],
        choices=["no_pruning", "reinit_rand", "reinit_orig", "no_reinit"],
        help="the experiment to run",
    )
    base_dir_default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output"
    )
    parser.add_argument(
        "--base_dir",
        metavar="base_dir",
        type=str,
        nargs=1,
        default=[base_dir_default],
        help="base output directory for results and checkpoints",
    )
    parser.add_argument(
        "--attack",
        metavar="attack",
        type=str,
        nargs=1,
        default=["fgsm"],
        choices=["fgsm", "pgd"],
        help="adversarial attack used for training and evaluation",
    )
    parser.add_argument(
        "--adv_train",
        action="store_true",
        default=False,
        help="use adversarial training for the given attack method",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        metavar="learning_rate",
        type=float,
        nargs=1,
        default=[0.0012],
        help="model's learning rate",
    )
    parser.add_argument(
        "-l1",
        "--l1_reg",
        metavar="l1_reg",
        type=float,
        nargs=1,
        default=[0.0],
        help="l1 regularization penalty",
    )
    parser.add_argument(
        "--devices",
        metavar="devices",
        type=str,
        nargs=1,
        default=["0,1,2,3"],
        help="gpu devices",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force train, deleting old experiment dirs if existing.",
    )
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    hparams = {
        "trials": args.trials[0],
        "train_iters": args.train_iters[0],
        "prune_iters": args.prune_iters[0],
        "eval_every": args.eval_every[0],
        "batch_size": args.batch_size[0],
        "valid_size": args.valid_size[0],
        "dataset": args.dataset[0],
        "model": args.model[0],
        "experiment": args.experiment[0],
        "attack": args.attack[0],
        "adv_train": args.adv_train,
        "base_dir": os.path.join(
            args.base_dir[0],
            args.dataset[0],
            args.model[0],
            args.experiment[0],
            args.attack[0],
        ),
        "learning_rate": args.learning_rate[0],
        "l1_reg": args.l1_reg[0],
        "devices": args.devices[0],
        "force": args.force,
    }
    exp_dir = "lr-{}_l1-{}_advtrain-{}".format(
        hparams["learning_rate"], hparams["l1_reg"], str(hparams["adv_train"]).lower()
    )
    hparams["base_dir"] = os.path.join(hparams["base_dir"], exp_dir)
    hparams["percents"] = {"hidden_1": 0.2, "hidden_2": 0.2, "output": 0.1}
    print("-" * 40, "hparams", "-" * 40)
    print("Beginning experiments using the following configuration:\n")
    for param, value in hparams.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 89)
    return hparams


def main():
    """Parses command line arguments and runs the specified experiment."""

    # Init hparams
    hparams = parse_args(init_flags())
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams["devices"]

    # Check if base_dir already exists, fail or create as specified
    if os.path.exists(hparams["base_dir"]) and not hparams["force"]:
        raise Exception(
            "directory '{} already exists. ".format(hparams["base_dir"])
            + "Run with --force to overwrite."
        )
    if os.path.exists(hparams["base_dir"]):
        shutil.rmtree(hparams["base_dir"])
    os.makedirs(hparams["base_dir"])

    # Fetch experiment function
    run_fn = registry.get_experiment_fn(hparams["experiment"])

    # Run experiments
    run_fn(hparams["dataset"], hparams["model"], hparams)


if __name__ == "__main__":
    main()
