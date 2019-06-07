"""Builds the FGSM attack."""
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from .registry import register
from .utils import *


@register("fgsm")
def build_attack(model, sess, eps=0.3, clip_min=0.0, clip_max=1.0):
    # Wrap model with cleverhans and init the attack method
    wrapped_model = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrapped_model, sess=sess)

    # Build acc and loss
    fgsm_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
    adv_acc_metric = get_adversarial_acc_metric(model, fgsm, fgsm_params)
    adv_loss = get_adversarial_loss(model, fgsm, fgsm_params)
    return fgsm, adv_acc_metric, adv_loss
