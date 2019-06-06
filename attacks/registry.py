"""Basic registry for attack builders."""

ATTACKS = dict()


def register(name):
    """Registers a new attack builder function under the given attack name."""

    def add_to_dict(func):
        ATTACKS[name] = func
        return func

    return add_to_dict


def get_builder(attack_name):
    """Fetches the attack builder function associated with the given attack name"""
    return ATTACKS[attack_name]
