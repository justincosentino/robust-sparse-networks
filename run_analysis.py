"""Run anlysis with CLI"""
import argparse
import os
import shutil

from .analysis import visualize

EXPERIMENTS = ["reinit_rand", "reinit_orig"]


def init_flags():
    """Init command line flags used for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Runs analysis on results generated by run_experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="adversarial attack to analyze",
    )
    parser.add_argument(
        "--adv_train",
        action="store_true",
        default=False,
        help="whether or not adversarial training was used for the given attack method",
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
        "--force",
        action="store_true",
        default=False,
        help="force analysis, deleting old anlysis dirs if existing.",
    )
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    hparams = {
        "dataset": args.dataset[0],
        "model": args.model[0],
        "attack": args.attack[0],
        "adv_train": args.adv_train,
        "base_dirs": {},
        "learning_rate": args.learning_rate[0],
        "l1_reg": args.l1_reg[0],
        "force": args.force,
        "experiments": EXPERIMENTS,
    }
    exp_dir = "lr-{}_l1-{}_advtrain-{}".format(
        hparams["learning_rate"], hparams["l1_reg"], str(hparams["adv_train"]).lower()
    )
    for experiment in hparams["experiments"]:
        hparams["base_dirs"][experiment] = os.path.join(
            args.base_dir[0],
            args.dataset[0],
            args.model[0],
            experiment,
            args.attack[0],
            exp_dir,
        )

    hparams["analysis_dir"] = os.path.join(
        args.base_dir[0],
        args.dataset[0],
        args.model[0],
        "analysis",
        args.attack[0],
        exp_dir,
    )
    print("-" * 40, "hparams", "-" * 40)
    print("Beginning anlysis for the following experiments:\n")
    for param, value in hparams.items():
        if param == "base_dirs":
            print("\t{:>13}:".format(param))
            for exp, exp_dir in value.items():
                print("\t\t{:>13}: {}".format(exp, exp_dir))
        else:
            print("\t{:>13}: {}".format(param, value))

    print()
    print("-" * 89)
    return hparams


def main():
    """Parses command line arguments and runs the specified analysis."""

    # Init hparams
    hparams = parse_args(init_flags())

    # Check if base_dir already exists, fail if not
    for experiment in hparams["experiments"]:
        if not os.path.exists(hparams["base_dirs"][experiment]):
            raise Exception(
                "directory '{} does not exist. ".format(
                    hparams["base_dirs"][experiment]
                )
            )
    if os.path.exists(hparams["analysis_dir"]) and not hparams["force"]:
        raise Exception(
            "directory '{} already exists. ".format(hparams["analysis_dir"])
            + "Run with --force to overwrite."
        )
    if os.path.exists(hparams["analysis_dir"]):
        shutil.rmtree(hparams["analysis_dir"])
    os.makedirs(hparams["analysis_dir"])

    visualize.run(hparams)

    # TODO: we need to run per-trial anlysis for network structure (ie weight magnitudes, etc. )


if __name__ == "__main__":
    main()