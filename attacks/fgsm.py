"""Builds the FGSM attack."""
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from .registry import register


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return tf.keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = tf.keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = tf.keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss


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
