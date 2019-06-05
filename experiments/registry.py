"""Basic registry for experiments."""

_EXPERIMENTS = dict()


def register(name):
    """Registers a new experiment function under the given experiment name."""

    def add_to_dict(func):
        _EXPERIMENTS[name] = func
        return func

    return add_to_dict


def get_experiment_fn(experiment_name):
    """Fetches the experiment function associated with the given experiment name"""
    return _EXPERIMENTS[experiment_name]
