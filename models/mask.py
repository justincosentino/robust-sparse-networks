"""MaskedDense layer"""

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import common_shapes, ops, tensor_shape
from tensorflow.python.ops import gen_math_ops, nn


class MaskedDense(tf.keras.layers.Dense):
    """Just your regular densely-connected NN layer with masking.
    `MaskedDense` implements the operation:
    `output = activation(dot(input, kernel*mask) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `mask` is a binary-mask over `kernel` weights, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    Example:
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(MaskedDense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(MaskedDense(32))
    ```
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        mask=None,
        **kwargs
    ):

        super(MaskedDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.mask_init = mask

    def build(self, input_shape):
        super(MaskedDense, self).build(input_shape)

        if self.mask_init is None:
            return

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        mask_init = np.array(self.mask_init)
        if mask_init.shape != self.kernel.get_shape():
            raise ValueError(
                "the provided mask dimenions do not match the kernel: got {}, expected {}".format(
                    mask_init.shape, self.kernel.get_shape()
                )
            )

        self.mask = self.add_weight(
            "mask",
            shape=[last_dim, self.units],
            initializer=tf.constant_initializer(mask_init),
            dtype=self.dtype,
            trainable=False,
        )

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            kernel = self.kernel
            if self.mask_init is not None:
                kernel = tf.multiply(self.kernel, self.mask)
            outputs = gen_math_ops.mat_mul(inputs, kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {"self.mask_init": self.mask_init}
        base_config = super(MaskedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
