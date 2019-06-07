"""Builds the PGD attack."""
import tensorflow as tf
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper

from .registry import register
from .utils import *


@register("pgd")
def build_attack(model, sess, eps=0.3, clip_min=0.0, clip_max=1.0):
    # Wrap model with cleverhans and init the attack method
    wrapped_model = KerasModelWrapper(model)
    pgd = ProjectedGradientDescent(wrapped_model, sess=sess)

    # Build acc and loss
    pgd_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
    adv_acc_metric = get_adversarial_acc_metric(model, pgd, pgd_params)
    adv_loss = get_adversarial_loss(model, pgd, pgd_params)
    return pgd, adv_acc_metric, adv_loss

