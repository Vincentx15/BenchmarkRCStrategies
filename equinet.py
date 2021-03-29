import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

import keras
from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
import keras.activations as ka

import numpy as np


class RegToRegConv(Layer):
    """
    Mapping from one reg layer to another
    """

    def __init__(self, reg_in, reg_out, kernel_size,
                 padding='valid',
                 kernel_initializer='glorot_uniform'):
        super(RegToRegConv, self).__init__()
        self.reg_in = reg_in
        self.reg_out = reg_out
        self.input_dim = 2 * reg_in
        self.filters = 2 * reg_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding

        self.left_kernel = self.add_weight(shape=(self.kernel_size // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size % 2 == 1:
            self.half_center = self.add_weight(shape=(1, 2 * self.reg_in, 2 * self.reg_out),
                                               initializer=self.kernel_initializer,
                                               name='center_kernel_half')

    def call(self, inputs, **kwargs):
        # Build the right part of the kernel from the left one
        #

        right_kernel = self.left_kernel[::-1, ::-1, ::-1]

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing everything.
        if self.kernel_size % 2 == 1:
            other_half = self.half_center[:, ::-1, ::-1]
            center_kernel = (other_half + self.half_center) / 2
            kernel = K.concatenate((self.left_kernel, center_kernel, right_kernel), axis=0)
        else:
            kernel = K.concatenate((self.left_kernel, right_kernel), axis=0)
        outputs = K.conv1d(inputs,
                           kernel,
                           padding=self.padding)
        return outputs

    def get_config(self):
        config = {'reg_in': self.reg_in,
                  'reg_out': self.reg_out}
        base_config = super(RegToRegConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if self.padding == 'valid':
            return None, length - self.kernel_size + 1, self.filters
        if self.padding == 'same':
            return None, length, self.filters


class RegToIrrepConv(Layer):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_in, a_out, b_out, kernel_size,
                 padding='valid',
                 kernel_initializer='glorot_uniform'):
        super(RegToIrrepConv, self).__init__()
        self.reg_in = reg_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = 2 * reg_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding

        self.left_kernel = self.add_weight(shape=(self.kernel_size // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size % 2 == 1:
            if self.a_out > 0:
                self.top_left = self.add_weight(shape=(1, self.reg_in, self.a_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_tl')
            if self.b_out > 0:
                self.bottom_right = self.add_weight(shape=(1, self.reg_in, self.b_out),
                                                    initializer=self.kernel_initializer,
                                                    name='center_kernel_br')

    def call(self, inputs, **kwargs):
        # Build the right part of the kernel from the left one
        # Columns are transposed, the b lines are flipped
        if self.a_out == 0:
            right_kernel = -self.left_kernel[::-1, ::-1, self.a_out:]
        elif self.b_out == 0:
            right_kernel = self.left_kernel[::-1, ::-1, :self.a_out]
        else:
            right_top = self.left_kernel[::-1, ::-1, :self.a_out]
            right_bottom = -self.left_kernel[::-1, ::-1, self.a_out:]
            right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing and flipping the sign of b_parts.
        if self.kernel_size % 2 == 1:
            if self.a_out == 0:
                bottom_left = -self.bottom_right[:, ::-1, :]
                bottom = K.concatenate((bottom_left, self.bottom_right), axis=1)
                kernel = K.concatenate((self.left_kernel, bottom, right_kernel), axis=0)
            elif self.b_out == 0:
                top_right = self.top_left[:, ::-1, :]
                top = K.concatenate((self.top_left, top_right), axis=1)
                kernel = K.concatenate((self.left_kernel, top, right_kernel), axis=0)
            else:
                top_right = self.top_left[:, ::-1, :]
                bottom_left = -self.bottom_right[:, ::-1, :]
                left = K.concatenate((self.top_left, bottom_left), axis=2)
                right = K.concatenate((top_right, self.bottom_right), axis=2)
                center_kernel = K.concatenate((left, right), axis=1)
                kernel = K.concatenate((self.left_kernel, center_kernel, right_kernel), axis=0)
        else:
            kernel = K.concatenate((self.left_kernel, right_kernel), axis=0)
        outputs = K.conv1d(inputs,
                           kernel,
                           padding=self.padding)
        return outputs

    def get_config(self):
        config = {'reg_in': self.reg_in,
                  'a_out': self.a_out,
                  'b_out': self.b_out}
        base_config = super(RegToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if self.padding == 'valid':
            return None, length - self.kernel_size + 1, self.filters
        if self.padding == 'same':
            return None, length, self.filters


class IrrepToIrrepConv(Layer):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, a_in, a_out, b_in, b_out, kernel_size,
                 padding='valid',
                 kernel_initializer='glorot_uniform'):
        super(IrrepToIrrepConv, self).__init__()

        self.a_in = a_in
        self.b_in = b_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = a_in + b_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding

        self.left_kernel = self.add_weight(shape=(self.kernel_size // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')
        # odd size
        if self.kernel_size % 2 == 1:
            # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts
            # Here we initialize the non zero blocks
            if self.a_out > 0 and self.a_in > 0:
                self.top_left = self.add_weight(shape=(1, self.a_in, self.a_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_tl')
            if self.b_out > 0 and self.b_in > 0:
                self.bottom_right = self.add_weight(shape=(1, self.b_in, self.b_out),
                                                    initializer=self.kernel_initializer,
                                                    name='center_kernel_br')

    def call(self, inputs):
        # Build the right part of the kernel from the left one
        # Here being on a 'b' part means flipping so the block diagonal is flipped

        # going from as ->
        if self.b_in == 0:
            # going from as -> bs
            if self.a_out == 0:
                right_kernel = - self.left_kernel[::-1, :, :]
            # going from as -> as
            elif self.b_out == 0:
                right_kernel = self.left_kernel[::-1, :, :]
            # going from as -> abs
            else:
                right_top = self.left_kernel[::-1, :, :self.a_out]
                right_bottom = - self.left_kernel[::-1, :, self.a_out:]
                right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # going from bs ->
        elif self.a_in == 0:
            # going from bs -> bs
            if self.a_out == 0:
                right_kernel = self.left_kernel[::-1, :, :]
            # going from bs -> as
            elif self.b_out == 0:
                right_kernel = - self.left_kernel[::-1, :, :]
            # going from bs -> abs
            else:
                right_top = -self.left_kernel[::-1, :, :self.a_out]
                right_bottom = self.left_kernel[::-1, :, self.a_out:]
                right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # going to -> bs
        elif self.a_out == 0:
            # going from bs -> bs
            if self.a_in == 0:
                right_kernel = self.left_kernel[::-1, :, :]
            # going from as -> bs
            elif self.b_in == 0:
                right_kernel = - self.left_kernel[::-1, :, :]
            # going from abs -> bs
            else:
                right_left = - self.left_kernel[::-1, :self.a_in, :]
                right_right = self.left_kernel[::-1, self.a_in:, :]
                right_kernel = K.concatenate((right_left, right_right), axis=1)

        # going to -> as
        elif self.b_out == 0:
            # going from bs -> as
            if self.a_in == 0:
                right_kernel = - self.left_kernel[::-1, :, :]
            # going from as -> as
            elif self.b_in == 0:
                right_kernel = self.left_kernel[::-1, :, :]
            # going from abs -> as
            else:
                right_left = self.left_kernel[::-1, :self.a_in, :]
                right_right = -self.left_kernel[::-1, self.a_in:, :]
                right_kernel = K.concatenate((right_left, right_right), axis=1)

        else:
            right_top_left = self.left_kernel[::-1, :self.a_in, :self.a_out]
            right_top_right = -self.left_kernel[::-1, self.a_in:, :self.a_out]
            right_bottom_left = -self.left_kernel[::-1, :self.a_in, self.a_out:]
            right_bottom_right = self.left_kernel[::-1, self.a_in:, self.a_out:]
            right_left = K.concatenate((right_top_left, right_bottom_left), axis=2)
            right_right = K.concatenate((right_top_right, right_bottom_right), axis=2)
            right_kernel = K.concatenate((right_left, right_right), axis=1)

        # Extra steps are needed for building the middle part when using the odd size
        if self.kernel_size % 2 == 1:

            # We only have the left part
            # going from as ->
            if self.b_in == 0:
                # going from as -> bs
                if self.a_out == 0:
                    center_kernel = K.zeros(shape=(1, self.a_in, self.b_out))
                # going from as -> as
                elif self.b_out == 0:
                    center_kernel = self.top_left
                # going from as -> abs
                else:
                    bottom_left = K.zeros(shape=(1, self.a_in, self.b_out))
                    center_kernel = K.concatenate((self.top_left, bottom_left), axis=2)

            # We only have the right part
            # going from bs -> 
            elif self.a_in == 0:
                # going from bs -> bs
                if self.a_out == 0:
                    center_kernel = self.bottom_right
                # going from bs -> as
                elif self.b_out == 0:
                    center_kernel = K.zeros(shape=(1, self.b_in, self.a_out))
                # going from bs -> abs
                else:
                    top_right = K.zeros(shape=(1, self.b_in, self.a_out))
                    center_kernel = K.concatenate((top_right, self.bottom_right), axis=2)

            # in <=> left/right, out <=> top/bottom

            # We only have the bottom
            # going to -> bs
            elif self.a_out == 0:
                # going from bs -> bs
                if self.a_in == 0:
                    center_kernel = self.bottom_right
                # going from as -> bs
                elif self.b_in == 0:
                    center_kernel = K.zeros(shape=(1, self.a_in, self.b_out))
                # going from abs -> bs
                else:
                    bottom_left = K.zeros(shape=(1, self.a_in, self.b_out))
                    center_kernel = K.concatenate((bottom_left, self.bottom_right), axis=1)

            # We only have the top
            # going to -> as
            elif self.b_out == 0:
                # going from bs -> as
                if self.a_in == 0:
                    center_kernel = K.zeros(shape=(1, self.b_in, self.a_out))
                # going from as -> as
                elif self.b_in == 0:
                    center_kernel = self.top_left
                # going from abs -> as
                else:
                    top_right = K.zeros(shape=(1, self.b_in, self.a_out))
                    center_kernel = K.concatenate((self.top_left, top_right), axis=1)

            else:
                # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts:
                bottom_left = K.zeros(shape=(1, self.a_in, self.b_out))
                top_right = K.zeros(shape=(1, self.b_in, self.a_out))
                left = K.concatenate((self.top_left, bottom_left), axis=2)
                right = K.concatenate((top_right, self.bottom_right), axis=2)
                center_kernel = K.concatenate((left, right), axis=1)
            kernel = K.concatenate((self.left_kernel, center_kernel, right_kernel), axis=0)

        else:
            kernel = K.concatenate((self.left_kernel, right_kernel), axis=0)

        outputs = K.conv1d(inputs,
                           kernel,
                           padding=self.padding)
        return outputs

    def get_config(self):
        config = {'a_in': self.a_in,
                  'a_out': self.a_out,
                  'b_in': self.b_in,
                  'b_out': self.b_out}
        base_config = super(IrrepToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if self.padding == 'valid':
            return None, length - self.kernel_size + 1, self.filters
        if self.padding == 'same':
            return None, length, self.filters


class IrrepActivationLayer(Layer):
    """
    BN layer for a_n, b_n feature map
    """

    def __init__(self, a, b, placeholder=False):
        super(IrrepActivationLayer, self).__init__()
        self.a = a
        self.b = b
        self.placeholder = placeholder

    def call(self, inputs):
        if self.placeholder:
            return inputs
        a_outputs = None
        if self.a > 0:
            a_inputs = inputs[:, :, :self.a]
            a_outputs = kl.ReLU()(a_inputs)
        if self.b > 0:
            b_inputs = inputs[:, :, self.a:]
            b_outputs = tf.tanh(b_inputs)
            if a_outputs is not None:
                return K.concatenate((a_outputs, b_outputs), axis=-1)
            else:
                return b_outputs
        return a_outputs


class RegBatchNorm(Layer):
    """
    BN layer for regular layers
    """

    def __init__(self, reg_dim, momentum=0.99, use_momentum=True, placeholder=False):
        super(RegBatchNorm, self).__init__()
        self.reg_dim = reg_dim
        self.placeholder = placeholder
        self.momentum = momentum
        self.use_momentum = use_momentum

        if not placeholder:
            self.passed = tf.Variable(initial_value=tf.constant(0, dtype=tf.float32), dtype=tf.float32, trainable=False)
            self.mu = self.add_weight(shape=([reg_dim]),
                                      initializer="zeros",
                                      name='mu')
            self.sigma = self.add_weight(shape=([reg_dim]),
                                         initializer="ones",
                                         name='sigma')
            self.running_mu = tf.Variable(initial_value=tf.zeros([reg_dim]), trainable=False)
            self.running_sigma = tf.Variable(initial_value=tf.ones([reg_dim]), trainable=False)

    def call(self, inputs, **kwargs):
        if self.placeholder:
            return inputs
        a = tf.shape(inputs)
        batch_size = a[0]
        length = a[1]
        division_over = batch_size * length
        division_over = tf.cast(division_over, tf.float32)

        modified_inputs = K.concatenate(
            tensors=[inputs[:, :, :self.reg_dim],
                     inputs[:, :, self.reg_dim:][:, :, ::-1]],
            axis=1)
        mu_batch = tf.reduce_mean(modified_inputs, axis=(0, 1))
        sigma_batch = tf.math.reduce_std(modified_inputs, axis=(0, 1)) + 0.0001
        train_normed_inputs = (modified_inputs - mu_batch) / sigma_batch * self.sigma + self.mu

        # Only update the running means in train mode
        if self.use_momentum:
            train_running_mu = self.running_mu * self.momentum + mu_batch * (1 - self.momentum)
            train_running_sigma = self.running_sigma * self.momentum + sigma_batch * (1 - self.momentum)
        else:
            train_running_mu = (self.running_mu * self.passed + division_over * mu_batch) / (
                    self.passed + division_over)
            train_running_sigma = (self.running_sigma * self.passed + division_over * sigma_batch) / (
                    self.passed + division_over)
            self.passed = K.in_train_phase(self.passed + division_over, self.passed)
        self.running_mu = K.in_train_phase(train_running_mu, self.running_mu)
        self.running_sigma = K.in_train_phase(train_running_sigma, self.running_sigma)

        train_true_normed_inputs = K.concatenate(
            tensors=[train_normed_inputs[:, :length, :],
                     train_normed_inputs[:, length:, :][:, :, ::-1]],
            axis=2)

        # Test mode
        test_normed_inputs = (modified_inputs - self.running_mu) / self.running_sigma * self.sigma + self.mu
        test_true_normed_inputs = K.concatenate(
            tensors=[test_normed_inputs[:, :length, :],
                     test_normed_inputs[:, length:, :][:, :, ::-1]],
            axis=2)

        out = K.in_train_phase(train_true_normed_inputs, test_true_normed_inputs)
        # out = K.in_train_phase(train_true_normed_inputs, 0 * test_true_normed_inputs)
        # out = K.in_train_phase(0 * test_true_normed_inputs, train_true_normed_inputs)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


class IrrepBatchNorm(Layer):
    """
    BN layer for a_n, b_n feature map
    """

    def __init__(self, a, b, placeholder=False, use_momentum=True, momentum=0.99, **kwargs):
        super(IrrepBatchNorm, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.placeholder = placeholder
        self.momentum = momentum
        self.use_momentum = use_momentum

        if not placeholder:
            self.passed = tf.Variable(initial_value=tf.constant(0, dtype=tf.float32), dtype=tf.float32, trainable=False)
            if a > 0:
                self.mu_a = self.add_weight(shape=([a]),
                                            initializer="zeros",
                                            name='mu_a')
                self.sigma_a = self.add_weight(shape=([a]),
                                               initializer="ones",
                                               name='sigma_a')
                self.running_mu_a = tf.Variable(initial_value=tf.zeros([a]), trainable=False)
                self.running_sigma_a = tf.Variable(initial_value=tf.ones([a]), trainable=False)

            if b > 0:
                self.sigma_b = self.add_weight(shape=([b]),
                                               initializer="ones",
                                               name='sigma_a')
                self.running_sigma_b = tf.Variable(initial_value=tf.ones([b]), trainable=False)

    def call(self, inputs, **kwargs):
        if self.placeholder:
            return inputs
        a = tf.shape(inputs)
        batch_size = a[0]
        length = a[1]
        division_over = batch_size * length
        division_over = tf.cast(division_over, tf.float32)

        # ============== Compute training values ===================
        # We have to compute statistics and update the running means if in train
        train_a_outputs = None
        if self.a > 0:
            a_inputs = inputs[:, :, :self.a]
            mu_a_batch = tf.reduce_mean(a_inputs, axis=(0, 1))
            sigma_a_batch = tf.math.reduce_std(a_inputs, axis=(0, 1)) + 0.0001

            # print('inbatch mu', mu_a_batch.numpy())
            # print('inbatch sigma', sigma_a_batch.numpy())
            train_a_outputs = (a_inputs - mu_a_batch) / sigma_a_batch * self.sigma_a + self.mu_a

            # Momentum version :
            if self.use_momentum:
                train_running_mu_a = self.running_mu_a * self.momentum + mu_a_batch * (1 - self.momentum)
                train_running_sigma_a = self.running_sigma_a * self.momentum + sigma_a_batch * (1 - self.momentum)
            else:
                train_running_mu_a = (self.running_mu_a * self.passed + division_over * mu_a_batch) / (
                        self.passed + division_over)
                train_running_sigma_a = (self.running_sigma_a * self.passed + division_over * sigma_a_batch) / (
                        self.passed + division_over)
            self.running_mu_a = K.in_train_phase(train_running_mu_a, self.running_mu_a)
            self.running_sigma_a = K.in_train_phase(train_running_sigma_a, self.running_sigma_a)

        # For b_dims, the problem is that we cannot compute a std from the mean as we include as a prior
        # that the mean is zero
        # We compute some kind of averaged over group action mean/std : std with a mean of zero.
        if self.b > 0:
            b_inputs = inputs[:, :, self.a:]
            numerator = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(b_inputs), axis=(0, 1)))
            sigma_b_batch = numerator / tf.math.sqrt(division_over) + 0.0001
            train_b_outputs = b_inputs / sigma_b_batch * self.sigma_b

            # Uncomment and call with RC batch to see that the mean is actually zero
            # and the estimated std is the right one
            # mu_b_batch = tf.reduce_mean(b_inputs, axis=(0, 1))
            # sigma_b_batch_emp = tf.math.reduce_std(b_inputs, axis=(0, 1)) + 0.0001
            # print('inbatch mu', mu_b_batch.numpy())
            # print('inbatch sigma', sigma_b_batch_emp.numpy())
            # print('computed sigman', sigma_b_batch.numpy())

            # Momentum version
            if self.use_momentum:
                train_running_sigma_b = self.running_sigma_b * self.momentum + sigma_b_batch * (1 - self.momentum)
            else:
                train_running_sigma_b = (self.running_sigma_b * self.passed + division_over * sigma_b_batch) / (
                        self.passed + division_over)
            self.running_sigma_b = K.in_train_phase(train_running_sigma_b, self.running_sigma_b)

            if train_a_outputs is not None:
                train_outputs = K.concatenate((train_a_outputs, train_b_outputs), axis=-1)
            else:
                train_outputs = train_b_outputs

        else:
            train_outputs = train_a_outputs
        self.passed = K.in_train_phase(self.passed + division_over, self.passed)

        # ============== Compute test values ====================
        test_a_outputs = None
        if self.a > 0:
            a_inputs = inputs[:, :, :self.a]
            test_a_outputs = (a_inputs - self.running_mu_a) / self.running_sigma_a * self.sigma_a + self.mu_a

        # For b_dims, we compute some kind of averaged over group action mean/std
        if self.b > 0:
            b_inputs = inputs[:, :, self.a:]
            test_b_outputs = b_inputs / self.running_sigma_b * self.sigma_b
            if test_a_outputs is not None:
                test_outputs = K.concatenate((test_a_outputs, test_b_outputs), axis=-1)
            else:
                test_outputs = test_b_outputs
        else:
            test_outputs = test_a_outputs

        # By default, this value returns the test value for the eager mode.
        # out = K.in_train_phase(0*train_outputs, test_outputs)
        # out = K.in_train_phase(train_outputs, 0*test_outputs)
        out = K.in_train_phase(train_outputs, test_outputs)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


class IrrepConcatLayer(Layer):
    """
    Concatenation layer to average both strands outputs
    """

    def __init__(self, a, b):
        super(IrrepConcatLayer, self).__init__()
        self.a = a
        self.b = b

    def call(self, inputs):
        a_outputs = None
        if self.a > 0:
            a_outputs = (inputs[:, :, :self.a] + inputs[:, ::-1, :self.a]) / 2
        if self.b > 0:
            b_outputs = (inputs[:, :, self.a:] - inputs[:, ::-1, self.a:]) / 2
            if a_outputs is not None:
                return K.concatenate((a_outputs, b_outputs), axis=-1)
            else:
                return b_outputs
        return a_outputs


class RegConcatLayer(Layer):
    """
    Concatenation layer to average both strands outputs
    """

    def __init__(self, reg):
        super(RegConcatLayer, self).__init__()
        self.reg = reg

    def call(self, inputs):
        outputs = (inputs[:, :, :self.reg] + inputs[:, :, self.reg:][:, ::-1, ::-1]) / 2
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2] / 2))


class EquiNetBinary:

    def __init__(self,
                 filters=((2, 2), (2, 2), (2, 2), (1, 1)),
                 kernel_sizes=(5, 5, 7, 7),
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """

        # assert len(filters) == len(kernel_sizes)
        # self.input_dense = 1000
        # successive_shrinking = (i - 1 for i in kernel_sizes)
        # self.input_dense = 1000 - sum(successive_shrinking)

        # First mapping goes from the input to an irrep feature space
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        self.reg_irrep = RegToIrrepConv(reg_in=2,
                                        a_out=first_a,
                                        b_out=first_b,
                                        kernel_size=first_kernel_size)
        self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.irrep_layers = []
        self.bn_layers = []
        self.activation_layers = []
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a,
                b_in=prev_b,
                a_out=next_a,
                b_out=next_b,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            # Don't add activation if it's the last layer
            placeholder = (i == len(filters) - 1)
            self.activation_layers.append(IrrepActivationLayer(a=next_a,
                                                               b=next_b,
                                                               placeholder=placeholder))

        self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
        self.flattener = kl.Flatten()
        self.dense = kl.Dense(out_size, activation='sigmoid')

    def func_api_model(self):
        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
        x = self.reg_irrep(inputs)
        x = self.first_bn(x)
        x = self.first_act(x)

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = IrrepConcatLayer(a=self.last_a, b=self.last_b)(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model

    def eager_call(self, inputs):
        rcinputs = inputs[:, ::-1, ::-1]

        x = self.reg_irrep(inputs)
        x = self.first_bn(x)
        x = self.first_act(x)

        rcx = self.reg_irrep(rcinputs)
        rcx = self.first_bn(rcx)
        rcx = self.first_act(rcx)

        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx[:, ::-1, :].numpy()[0, :5, :])
        # print()

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

            rcx = irrep_layer(rcx)
            rcx = bn_layer(rcx)
            rcx = activation_layer(rcx)

        # Print the beginning of both strands to see it adds up in concat
        # print(x.shape)
        # print(x.numpy()[0, :5, :])
        # print('end')
        # print(rcx.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx[:, ::-1, :].numpy()[0, :5, :])
        # print()

        # Average two strands predictions
        x = IrrepConcatLayer(a=self.last_a, b=self.last_b)(x)
        rcx = IrrepConcatLayer(a=self.last_a, b=self.last_b)(rcx)

        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx.numpy()[0, :5, :])
        # print()

        x = self.pool(x)
        rcx = self.pool(rcx)

        # print(x.shape)
        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx.numpy()[0, :5, :])
        # print()

        x = self.flattener(x)
        rcx = self.flattener(rcx)

        # print(x.shape)
        # print(x.numpy()[0, :5])
        # print('reversed')
        # print(rcx.numpy()[0, :5])
        # print()

        outputs = self.dense(x)
        rcout = self.dense(rcx)

        # print(outputs.shape)
        # print(outputs.numpy()[0, :5])
        # print('reversed')
        # print(rcout.numpy()[0, :5])
        # print()
        return outputs


class RCNetBinary:

    def __init__(self,
                 filters=[2, 2, 2],
                 kernel_sizes=[5, 5, 7],
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """

        # First mapping goes from the input to an irrep feature space
        first_kernel_size = kernel_sizes[0]
        first_reg = filters[0]
        self.last_reg = filters[-1]

        self.first_conv = RegToRegConv(reg_in=2,
                                       reg_out=first_reg,
                                       kernel_size=first_kernel_size)
        self.first_bn = RegBatchNorm(reg_dim=first_reg, placeholder=placeholder_bn)
        self.first_act = kl.core.Activation("relu")

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.irrep_layers = []
        self.bn_layers = []
        self.activation_layers = []
        for i in range(1, len(filters)):
            prev = filters[i - 1]
            next = filters[i]
            self.irrep_layers.append(RegToRegConv(
                reg_in=prev,
                reg_out=next,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(RegBatchNorm(reg_dim=next, placeholder=placeholder_bn))
            self.activation_layers.append(kl.core.Activation("relu"))

        self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
        self.flattener = kl.Flatten()
        self.dense = kl.Dense(out_size, activation='sigmoid')

    def func_api_model(self):
        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
        x = self.first_conv(inputs)
        x = self.first_bn(x)
        x = self.first_act(x)

        for conv_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = RegConcatLayer(reg=self.last_reg)(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model

    def eager_call(self, inputs):
        rcinputs = inputs[:, ::-1, ::-1]

        x = self.first_conv(inputs)
        x = self.first_bn(x)
        x = self.first_act(x)

        rcx = self.first_conv(rcinputs)
        rcx = self.first_bn(rcx)
        rcx = self.first_act(rcx)

        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx[:, ::-1, :].numpy()[0, :5, :])
        # print()

        for conv_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

            rcx = conv_layer(rcx)
            rcx = bn_layer(rcx)
            rcx = activation_layer(rcx)

        # Print the beginning of both strands to see it adds up in concat
        # print(x.shape)
        # print(x.numpy()[0, :5, :])
        # print('end')
        # print(rcx.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx[:, ::-1, :].numpy()[0, :5, :])

        # Average two strands predictions
        x = RegConcatLayer(reg=self.last_reg)(x)
        rcx = RegConcatLayer(reg=self.last_reg)(rcx)

        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx.numpy()[0, :5, :])
        # print()

        x = self.pool(x)
        rcx = self.pool(rcx)

        # print(x.shape)
        # print(x.numpy()[0, :5, :])
        # print('reversed')
        # print(rcx.numpy()[0, :5, :])
        # print()

        x = self.flattener(x)
        rcx = self.flattener(rcx)

        # print(x.numpy()[0, :5])
        # print('reversed')
        # print(rcx.numpy()[0, :5])
        # print()

        outputs = self.dense(x)
        rcout = self.dense(rcx)

        # print(outputs.shape)
        # print(outputs.numpy()[0, :5])
        # print('reversed')
        # print(rcout.numpy()[0, :5])
        # print()
        return outputs


if __name__ == '__main__':
    pass
    import tensorflow as tf
    from keras.utils import Sequence

    eager = False

    if eager:
        tf.enable_eager_execution()

    curr_seed = 42
    np.random.seed(curr_seed)
    tf.set_random_seed(curr_seed)


    def random_one_hot(size=(2, 100), return_tf=True):
        bs, len = size
        numel = bs * len
        randints_np = np.random.randint(0, 3, size=numel)
        one_hot_np = np.eye(4)[randints_np]
        one_hot_np = np.reshape(one_hot_np, newshape=(bs, len, 4))
        if return_tf:
            tf_one_hot = tf.convert_to_tensor(one_hot_np, dtype=float)
            return tf_one_hot
        return one_hot_np


    class Generator(Sequence):
        def __init__(self, eager=False, inlen=1000, outlen=1000, infeat=4, outfeat=1, bs=1, binary=False, one_hot=True):
            self.eager = eager
            self.inlen = inlen
            self.outlen = outlen
            self.infeat = infeat
            self.outfeat = outfeat
            self.bs = bs
            self.binary = binary
            self.one_hot = one_hot

        def __getitem__(self, item):
            if self.eager:
                if self.one_hot:
                    inputs = random_one_hot((self.bs, self.inlen))
                else:
                    inputs = tf.ones(shape=(self.bs, self.inlen, self.infeat))
                if self.binary:
                    targets = tf.ones(shape=(1))
                else:
                    targets = tf.ones(shape=(self.bs, self.outlen, self.outfeat))
            else:
                if self.one_hot:
                    inputs = random_one_hot((self.bs, self.inlen), return_tf=False)
                else:
                    inputs = np.ones(shape=(self.bs, self.inlen, self.infeat))
                if self.binary:
                    targets = np.ones(shape=(self.bs, 1))
                else:
                    targets = np.ones(shape=(self.bs, self.outlen, self.outfeat))
            return inputs, targets

        def __len__(self):
            return 10

        def __iter__(self):
            for item in (self[i] for i in range(len(self))):
                yield item


    # Now create the layers objects

    a_1 = 2
    b_1 = 0
    reg_out = 10
    a_2 = 0
    b_2 = 2

    # generator = Generator(outfeat=4, outlen=986, eager=eager)
    # val_generator = Generator(outfeat=4, outlen=986, eager=eager)
    generator = Generator(binary=True)
    val_generator = Generator(binary=True)
    # inputs, targets = generator[2]
    # print(inputs.shape)
    # print(targets.shape)

    reg_reg = RegToRegConv(reg_in=2,
                           reg_out=reg_out,
                           kernel_size=8)

    reg_irrep = RegToIrrepConv(reg_in=reg_out,
                               a_out=a_1,
                               b_out=b_1,
                               kernel_size=8)

    irrep_irrep = IrrepToIrrepConv(a_in=a_1,
                                   b_in=b_1,
                                   a_out=a_2,
                                   b_out=b_2,
                                   kernel_size=8)
    bn_irrep = IrrepBatchNorm(a=a_1,
                              b=b_1)

    # Now use these layers to build models : either use directly in eager mode or use the functional API for Keras use

    # model = reg_irrep
    # model = irrep_irrep
    # model = reg_reg
    # model = EquiNet()

    # Keras Style
    if not eager:
        pass
        # CHECK EQUIVARIANCE of the rcps : to me it should not be equivariant
        #   because of the maxpooling that is called too soon

        # parameters = {
        #     'filters': 16,
        #     'input_length': 1000,
        #     'pool_size': 40,
        #     'strides': 20
        # }
        # model = BA.get_rc_model(parameters=parameters, is_weighted_sum=False)
        #
        # x = np.random.uniform(size=(5, 1000, 4))
        # rcx = x[:, ::-1, ::-1]
        # out1 = model.predict(x)
        # print(out1)
        # out2 = model.predict(rcx)
        # print(out2)

        generator = Generator(outlen=1000 - 7 - 7 - 7, outfeat=2)
        val_generator = Generator(outlen=1000 - 7 - 7 - 7, outfeat=2)

        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
        outputs = reg_reg(inputs)
        outputs = RegBatchNorm(reg_dim=reg_out)(outputs)

        outputs = reg_irrep(outputs)
        outputs = IrrepBatchNorm(a_1, b_1)(outputs)
        outputs = irrep_irrep(outputs)

        # outputs = IrrepActivationLayer(a_1, b_1)(outputs)
        model = keras.Model(inputs, outputs)
        # model.summary()
        # model = EquiNetBinary(placeholder_bn=False).func_api_model()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="mse", metrics=["mse"])
        model.fit_generator(generator,
                            validation_data=val_generator,
                            epochs=25)

        # loss_history = model.fit_generator(generator,
        #                                    validation_data=(valid_data.X, valid_data.Y),
        #                                    epochs=5,
        #                                    steps_per_epoch=10,)

    if eager:
        pass


        # inputs = tf.random.uniform((2, 1000, 4))
        # inputs = tf.ones((2, 1000, 4))
        # x2 = x[:, ::-1, ::-1]
        # print('without BN')
        # out1 = RCNetBinary(placeholder_bn=False).eager_call(inputs)
        # out1 = EquiNetBinary(placeholder_bn=False).eager_call(inputs)
        # outputs = reg_irrep(inputs)
        # outputs = IrrepBatchNorm(a_1, b_1)(outputs)

        # outputs = reg_reg(x)
        # outputs = RegBatchNorm(reg_dim=reg_out)(inputs)
        #
        # outputs = reg_irrep(inputs)
        # outputs = IrrepBatchNorm(a_1, b_1)(outputs)
        # outputs = irrep_irrep(outputs)

        # print(outputs[0, :5, :].numpy())
        # outputs = IrrepActivationLayer(a_1, b_1)(outputs)

        # print()
        # print('with BN')
        # out2 = RCNetBinary(placeholder_bn=False).eager_call(inputs)
        # out2 = EquiNetBinary(placeholder_bn=False).eager_call(inputs)
        # print(x)
        # x = reg_irrep(inputs)
        # x = bn_irrep(x)
        # x = tf.math.reduce_mean(x, axis=(1,2))
        # out1 = reg_irrep(x)
        # out1 = bn_irrep(out1)
        # out2 = reg_irrep(x2)
        # out2 = bn_irrep(out2)
        # print(out1[0, :5, :].numpy())
        # out1 = ActivationLayer(a_1, b_1)(out1)
        # out2 = ActivationLayer(a_1, b_1)(out2)
        #
        # print(out1[0, :5, :].numpy())
        # print('reversed')
        # print(out2[0, -5:, :].numpy()[::-1])
        # #
        # x = tf.random.uniform((1, 1000, 4))
        # x2 = x[:, ::-1, ::-1]
        # out1 = model(x)
        # out2 = model(x2)
        #
        # print(out1.numpy())
        # print('reversed')
        # print(out2.numpy()[::-1])

        class testmodel(tf.keras.Model):

            def __init__(self):
                super(testmodel, self).__init__()

                a_1 = 0
                b_1 = 2
                a_2 = 2
                b_2 = 0

                self.reg_irrep = RegToIrrepConv(reg_in=2,
                                                a_out=a_1,
                                                b_out=b_1,
                                                kernel_size=8)
                self.irrep_bn = IrrepBatchNorm(a_1, b_1)
                self.irrep_irrep = IrrepToIrrepConv(a_in=a_1,
                                                    b_in=b_1,
                                                    a_out=a_2,
                                                    b_out=b_2,
                                                    kernel_size=8)

            def call(self, inputs, **kwargs):
                outputs = self.reg_irrep(inputs)
                outputs = self.irrep_bn(outputs)
                outputs = self.irrep_irrep(outputs)
                return outputs


        model = testmodel()
        optimizer = tf.keras.optimizers.Adam()
        generator = Generator(outlen=1000 - 7 - 7, outfeat=2, eager=eager)
        val_generator = Generator(outlen=1000 - 7 - 7, outfeat=2, eager=eager)

        print('Training phase')
        K.set_learning_phase(1)
        for epoch in range(10):
            for batch_idx, (batch_in, batch_out) in enumerate(generator):
                # Make RC batch for testing
                # rc_batch = tf.concat((batch_in, batch_in[:, ::-1, ::-1]), axis=0)
                # pred = model(rc_batch)

                with tf.GradientTape() as tape:
                    pred = model(batch_in)
                    loss = tf.reduce_mean(tf.keras.losses.MSE(batch_out, pred))
                    grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print('running sigma', model.irrep_bn.running_sigma_b.numpy())
                print('sigmab', model.irrep_bn.sigma_b.numpy())
                # print('running mu', model.irrep_bn.running_mu_a.numpy())
                # print('running sigma', model.irrep_bn.running_sigma_a.numpy())
                # print('mua', model.irrep_bn.mu_a.numpy())
                # print('sigmaa', model.irrep_bn.sigma_a.numpy())
                print()
            print(loss.numpy())

        # print('Prediction phase')
        # K.set_learning_phase(0)
        # for epoch in range(10):
        #     for batch_idx, (batch_in, batch_out) in enumerate(val_generator):
        #         pred = model(batch_in)
        #         loss = tf.reduce_mean(tf.keras.losses.MSE(batch_out, pred))
        #     print(loss.numpy())
