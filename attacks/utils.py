import tensorflow as tf


def get_adversarial_acc_metric(model, attack, attack_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = attack.generate(model.input, **attack_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return tf.keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, attack, attack_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = tf.keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = attack.generate(model.input, **attack_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = tf.keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss
