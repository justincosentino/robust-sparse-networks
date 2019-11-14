# robust-sparse-networks

Empirically evaluating basic robustness properties of the winning "lottery tickets".

This repository provides the code required to run experiments from "The Search for Sparse, Robust Neural Networks" by {Justin Cosentino, Federico Zaiter}, {Dan Pei and Jun Zhu}. We presented this work at the Safety and Robustness in Decision Making Workshop at the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), in Vancouver, Canada.

## Table of Contents

- [robust-sparse-networks](#robust-sparse-networks)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
    - [Directory Structure](#directory-structure)
    - [Datasets](#datasets)
    - [Flags](#flags)
  - [Experiments](#experiments)

## Abstract

Recent work on deep neural network pruning has shown there exist sparse subnetworks that achieve equal or improved accuracy, training time, and loss using fewer network parameters when compared to their dense counterparts. Orthogonal to pruning literature, deep neural networks are known to be susceptible to adversarial examples, which may pose risks in security- or safety-critical applications. Intuition suggests that there is an inherent trade-off between sparsity and robustness such that these characteristics could not co-exist. We perform an extensive empirical evaluation and analysis testing the Lottery Ticket Hypothesis with adversarial training and show this approach enables us to find sparse, robust neural networks.

## Requirements

This implementation assumes `python==3.6.9` and `tensorflow-gpu==1.13.1`. All requirements are listed in `requirements.txt`. If you wish to run without gpu support, specify `tensorflow==1.13.1` in `requirements.txt` before continuing.

We recommend creating a new virtual environment using [conda](https://docs.conda.io/en/latest/):

```[bash]
conda create --name "robust_sparse_networks" python=3.6
conda activate robust_sparse_networks
```

To install, run:

```[bash]
pip install -r requirements.txt
```

## Quick Start

Experiments and analyses are each run with their own script. To use the experiment script, navigate to the module's parent directory and run

```[bash]
$ python -m network-pruning.train
```

A unique `experiment_id` is either specified by the user or generated using a subset of flag parameters (`dataset`, `epoch`, `learning_rate`, etc.). Checkpoints and results will be written to the given `base_dir` directory.

### Directory Structure

The code is organized as follows:

```[txt]
.
├── LICENSE
├── README.md
├── __init__.py
├── analysis
│   ├── __init__.py
│   └── visualize.py        # generates graphs and tables
├── attacks
│   ├── __init__.py
│   ├── fgsm.py             # build adversarial loss and accuracy for fgsm
│   ├── pgd.py              # build adversarial loss and accuracy for pgd
│   ├── registry.py
│   └── utils.py
├── data
│   ├── __init__.py
│   ├── digits_loader.py    # data loader for mnist digits
│   ├── fashion_loader.py   # data loader for mnist fashion
│   ├── loader_utils.py     # data loading and preprocessing utils
│   └── registry.py
├── experiments
│   ├── __init__.py
│   ├── base_experiment.py  # base experimental setup
│   ├── callbacks.py        # callbacks used to log train, validation and test accuracy
│   ├── no_pruning.py       # run trials without pruning
│   ├── path.py             # path building utilities
│   ├── pruning.py          # pruning utilities
│   ├── registry.py
│   ├── reinit_none.py      # experiments do not reinit weights after pruning
│   ├── reinit_orig.py      # experiments reinit weights to original after pruning
│   ├── reinit_rand.py      # experiments reinit weights to random after pruning
│   └── utils.py
├── models
│   ├── __init__.py
│   ├── dense.py            # the dense-300-100 model
│   ├── mask.py             # a custom masked dense layer
│   └── registry.py
├── requirements.txt
├── run_analysis.py         # performs analysis on a given experimental run
├── run_experiments.py      # runs experiments for a given config
└── run_sh/                 # example experimental runs
```

### Datasets

Experiments can be run on either the [MNIST Digits](http://yann.lecun.com/exdb/mnist/) or [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist) datasets using the `--dataset={digits|fashion}` flag.

### Flags

Flags are provided to easily control experiments. For example,

```[bash]
python -m robust-sparse-networks.run_experiments --trials=1 --learning_rate=0.001  --attack=pgd
```

A full list of flags can be found here:

```[bash]
$ python -m robust-sparse-networks.run_experiments --help
> usage: run_experiments.py [-h] [--trials trials] [--train_iters train_iters]
>                           [--prune_iters prune_iters]
>                           [--eval_every eval_every] [--batch_size batch_size]
>                           [--valid_size valid_size] [--dataset dataset]
>                           [--model model] [--experiment experiment]
>                           [--base_dir base_dir] [--attack attack]
>                           [--adv_train] [-lr learning_rate] [-l1 l1_reg]
>                           [--devices devices] [--force]
>
> Runs experiments to find robust, sparse networks.
>
> optional arguments:
>   -h, --help            show this help message and exit
>   --trials trials       number trials per experiment (default: [10])
>   --train_iters train_iters
>                         number of training iterations (default: [50000])
>   --prune_iters prune_iters
>                         number of pruning iterations (default: [20])
>   --eval_every eval_every
>                         number of iterations to eval on validation set
>                         (default: [500])
>   --batch_size batch_size
>                         batch size (default: [60])
>   --valid_size valid_size
>                         validation set size (default: [10000])
>   --dataset dataset     source dataset (default: ['digits'])
>   --model model         model type (default: ['dense-300-100'])
>   --experiment experiment
>                         the experiment to run (default: ['reinit_orig'])
>   --base_dir base_dir   base output directory for results and checkpoints
>                         (default: ['/home/justin/gpu1.back/robust-sparse-
>                         networks/output'])
>   --attack attack       adversarial attack used for training and evaluation
>                         (default: ['fgsm'])
>   --adv_train           use adversarial training for the given attack method
>                         (default: False)
>   -lr learning_rate, --learning_rate learning_rate
>                         model's learning rate (default: [0.0012])
>   -l1 l1_reg, --l1_reg l1_reg
>                         l1 regularization penalty (default: [0.0])
>   --devices devices     gpu devices (default: ['0,1,2,3'])
>   --force               force train, deleting old experiment dirs if existing.
>                         (default: False)
```

## Experiments

Experiments were run on a 300-100-10 `MaskedDense` network:

```[text]
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
hidden_1 (MaskedDense)       (None, 300)               470400
_________________________________________________________________
hidden_2 (MaskedDense)       (None, 100)               60000
_________________________________________________________________
output (MaskedDense)         (None, 10)                2000
=================================================================
Total params: 532,400
Trainable params: 266,200
Non-trainable params: 266,200
_________________________________________________________________
```

The `MaskedDense` layer is a custom layer wrapping the standard `Dense` layer. It supports masking, allowing us to zero-out specific weights.

Each hidden layer used a ReLU activation function while the final output layer used a softmax activation function. We do not use biases. Models are trained using the Adam optimizer and a learning rate of 1.2e-3. All models train for 50,000 iterations per pruning iteration and use a batch size of 60. The first two layers have a pruning rate of 20%, while the output layer has a pruning rate of 10%.

We complete three separate pruning experiments:

- Iterative pruning with resetting: in this strategy, we reset the network to its original parameters after each training and pruning cycle
- Iterative pruning with random reinitialization: in this strategy, we reinitialize the
network to random parameters after each training and pruning cycle
- Iterative pruning with continued training: in this strategy, we never reset the network to random parameters, continuing to train the current parameter set after each training and pruning cycle

For each pruning strategy, we train the model with and without adversarial training using either the FGSM or PGD attack for 20 pruning iterations. A single pruning iteration consists of initializing the current iteration’s parameters according to the pruning strategy, training for 50,000 iterations, and pruning some percent of the model to get an updated mask. We evaluate the model on both natural and adversarial examples from the entire validation and test set every 500 training iterations. Experimental results, unless otherwise noted, are averaged over five trials of each experiment. Any error metrics denote standard deviation.

Normal training consists of minimizing the categorical cross entropy loss function on natural examples. Adversarial training minimizes a combination of categorical cross entropy loss on natural examples and adversarial examples with a 50/50 split. We craft FGSM attacks in a l∞ ball of ε = 0.3. We craft PGD attacks with a step size of 0.05 for 10 iterations in a l∞ ball of ε = 0.3. All attacks are clipped to be within [0, 1].

All experiments were run on both the MNIST Digits and MNIST Fashion datasets.
