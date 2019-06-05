"""Basic registry for model builders."""

BUILDERS = dict()


def register(name):
    """Registers a new model builder function under the given model name."""

    def add_to_dict(func):
        BUILDERS[name] = func
        return func

    return add_to_dict


def get_builder(model_name):
    """Fetches the model builder function associated with the given model name"""
    return BUILDERS[model_name]
