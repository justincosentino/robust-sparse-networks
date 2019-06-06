"""Basic registry for data loaders."""

_LOADERS = dict()


def register(name):
    """Registers a new data loader function under the given name."""

    def add_to_dict(func):
        _LOADERS[name] = func
        return func

    return add_to_dict


def get_loader(data_src):
    """Fetches the data loader function associated with the given data src"""
    return _LOADERS[data_src]
