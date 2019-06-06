from .registry import register


@register("pdg")
def build_attack():
    raise NotImplementedError()
