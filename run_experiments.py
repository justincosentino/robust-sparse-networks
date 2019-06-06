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
        "--epochs",
        metavar="epochs",
        type=int,
        nargs=1,
        default=[10],
        help="number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        type=int,
        nargs=1,
        default=[128],
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
        choices=["no_pruning"],
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
        "-lr",
        "--learning_rate",
        metavar="learning_rate",
        type=float,
        nargs=1,
        default=[1e-3],
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
        "epochs": args.epochs[0],
        "batch_size": args.batch_size[0],
        "valid_size": args.valid_size[0],
        "dataset": args.dataset[0],
        "model": args.model[0],
        "experiment": args.experiment[0],
        "base_dir": os.path.join(args.base_dir[0], args.dataset[0], args.model[0]),
        "learning_rate": args.learning_rate[0],
        "l1_reg": args.l1_reg[0],
        "devices": args.devices[0],
        "force": args.force,
    }
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