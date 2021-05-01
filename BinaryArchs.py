import keras
import keras.layers as kl
from keras.layers.core import Flatten

from equinet import RegToIrrepConv, IrrepToIrrepConv, RegToRegConv
from equinet import ToKmerLayer, IrrepActivationLayer, IrrepBatchNorm, RegBatchNorm, IrrepConcatLayer, RegConcatLayer


def get_reg_model(parameters):
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution1D(
        input_shape=(1000, 4), nb_filter=16, filter_length=15))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.core.Activation("relu"))
    model.add(keras.layers.convolutional.Convolution1D(
        nb_filter=16, filter_length=14))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.convolutional.Convolution1D(
        nb_filter=16, filter_length=14))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                      strides=parameters['strides']))
    model.add(Flatten())
    model.add(keras.layers.core.Dense(output_dim=1, trainable=True,
                                      init="glorot_uniform"))
    model.add(keras.layers.core.Activation("sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


class EquiNetBinary:

    def __init__(self,
                 filters=((16, 16), (16, 16), (16, 16)),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """

        # assert len(filters) == len(kernel_sizes)
        # self.input_dense = 1000
        # successive_shrinking = (i - 1 for i in kernel_sizes)
        # self.input_dense = 1000 - sum(successive_shrinking)

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        reg_in = self.to_kmer.features // 2

        # First mapping goes from the input to an irrep feature space
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        self.reg_irrep = RegToIrrepConv(reg_in=reg_in,
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
            # placeholder = (i == len(filters) - 1)
            # placeholder = True
            self.activation_layers.append(IrrepActivationLayer(a=next_a,
                                                               b=next_b))

        self.concat = IrrepConcatLayer(a=self.last_a, b=self.last_b)
        self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
        self.flattener = kl.Flatten()
        self.dense = kl.Dense(out_size, activation='sigmoid')

    def func_api_model(self):
        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")

        x = self.to_kmer(inputs)
        x = self.reg_irrep(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model

    def eager_call(self, inputs):
        rcinputs = inputs[:, ::-1, ::-1]

        x = self.to_kmer(inputs)
        x = self.reg_irrep(x)
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
        x = self.concat(x)
        rcx = self.concat(rcx)

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


class CustomRCPS:

    def __init__(self,
                 filters=(16, 16, 16),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        reg_in = self.to_kmer.features // 2
        filters = [reg_in] + list(filters)

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.reg_layers = []
        self.bn_layers = []
        self.activation_layers = []
        for i in range(len(filters) - 1):
            prev_reg = filters[i]
            next_reg = filters[i + 1]
            self.reg_layers.append(RegToRegConv(
                reg_in=prev_reg,
                reg_out=next_reg,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(RegBatchNorm(reg_dim=next_reg, placeholder=placeholder_bn))
            # Don't add activation if it's the last layer
            placeholder = (i == len(filters) - 1)
            self.activation_layers.append(kl.core.Activation("relu"))

        self.concat = RegConcatLayer(reg=filters[-1])
        self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
        self.flattener = kl.Flatten()
        self.dense = kl.Dense(out_size, activation='sigmoid')

    def func_api_model(self):
        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")

        x = self.to_kmer(inputs)

        for reg_layer, bn_layer, activation_layer in zip(self.reg_layers, self.bn_layers, self.activation_layers):
            x = reg_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model

    def eager_call(self, inputs):
        x = self.to_kmer(inputs)
        for reg_layer, bn_layer, activation_layer in zip(self.reg_layers, self.bn_layers, self.activation_layers):
            x = reg_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model


if __name__ == '__main__':

    import tensorflow as tf
    from equinet import Generator

    eager = False

    if eager:
        tf.enable_eager_execution()
    generator = Generator(binary=True, eager=eager)
    val_generator = Generator(binary=True, eager=eager)
    model = EquiNetBinary(placeholder_bn=False, kmers=3).func_api_model()
    # model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="mse", metrics=["accuracy"])
    model.fit_generator(generator,
                        validation_data=val_generator,
                        validation_steps=10,
                        epochs=3)

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
"""


import tensorflow as tf
import numpy as np
import os
import functools

import keras_genomics
from keras import backend as K
from keras.layers.core import Dropout
from keras.engine import Layer
from keras.models import Sequential
from keras.engine.base_layer import InputSpec
from keras import initializers

from keras.layers import Input
from keras.models import Model


# Used to preserve RC symmetry
class RevCompSumPool(Layer):
    def __init__(self, **kwargs):
        super(RevCompSumPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_input_chan = input_shape[2]
        super(RevCompSumPool, self).build(input_shape)

    def call(self, inputs):
        inputs = (inputs[:, :, :int(self.num_input_chan / 2)] + inputs[:, :, int(self.num_input_chan / 2):][:, ::-1,
                                                                ::-1])
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2] / 2))


class WeightedSum1D(Layer):
    '''Learns a weight for each position, for each channel, and sums
    lengthwise.
    # Arguments
        symmetric: if want weights to be symmetric along length, set to True
        input_is_revcomp_conv: if the input is [RevCompConv1D], set to True for
            added weight sharing between reverse-complement pairs
        smoothness_penalty: penalty to be applied to absolute difference
            of adjacent weights in the length dimension
        bias: whether or not to have bias parameters
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''

    def __init__(self, symmetric, input_is_revcomp_conv,
                 smoothness_penalty=None, bias=False,
                 **kwargs):
        super(WeightedSum1D, self).__init__(**kwargs)
        self.symmetric = symmetric
        self.input_is_revcomp_conv = input_is_revcomp_conv
        self.smoothness_penalty = smoothness_penalty
        self.bias = bias
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        # input_shape[0] is the batch index
        # input_shape[1] is length of input
        # input_shape[2] is number of filters

        # Equivalent to 'fanintimesfanouttimestwo' from the paper
        limit = np.sqrt(6.0 / (input_shape[1] * input_shape[2] * 2))
        self.init = initializers.uniform(-1 * limit, limit)

        if (self.symmetric == False):
            W_length = input_shape[1]
        else:
            self.odd_input_length = input_shape[1] % 2.0 == 1
            # +0.5 below turns floor into ceil
            W_length = int(input_shape[1] / 2.0 + 0.5)

        if (self.input_is_revcomp_conv == False):
            W_chan = input_shape[2]
        else:
            assert input_shape[2] % 2 == 0, \
                "if input is revcomp conv, # incoming channels would be even"
            W_chan = int(input_shape[2] / 2)

        self.W_shape = (W_length, W_chan)
        self.b_shape = (W_chan,)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=(None if self.smoothness_penalty is None else
                                              regularizers.SmoothnessRegularizer(
                                                  self.smoothness_penalty)))
        if (self.bias):
            assert False, "No bias was specified in original experiments"

        self.built = True

    # 3D input -> 2D output (loses length dimension)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if (self.symmetric == False):
            W = self.W
        else:
            W = K.concatenate(
                tensors=[self.W,
                         # reverse along length, concat along length
                         self.W[::-1][(1 if self.odd_input_length else 0):]],
                axis=0)
        if (self.bias):
            b = self.b
        if (self.input_is_revcomp_conv):
            # reverse along both length and channel dims, concat along chan
            # if symmetric=True, reversal along length here makes no diff
            W = K.concatenate(tensors=[W, W[::-1, ::-1]], axis=1)
            if (self.bias):
                b = K.concatenate(tensors=[b, b[::-1]], axis=0)
        output = K.sum(x * K.expand_dims(W, 0), axis=1)
        if (self.bias):
            output = output + K.expand_dims(b, 0)
        return output

    def get_config(self):
        config = {'symmetric': self.symmetric,
                  'input_is_revcomp_conv': self.input_is_revcomp_conv,
                  'smoothness_penalty': self.smoothness_penalty,
                  'bias': self.bias}
        base_config = super(WeightedSum1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_rc_model(parameters, is_weighted_sum, use_bias=False):
    rc_model = keras.models.Sequential()
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        input_shape=(1000, 4), nb_filter=16, filter_length=15))
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        nb_filter=16, filter_length=14))
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        nb_filter=16, filter_length=14))
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras.layers.convolutional.MaxPooling1D(
        pool_length=parameters['pool_size'], strides=parameters['strides']))
    if is_weighted_sum:
        rc_model.add(WeightedSum1D(
            symmetric=False, input_is_revcomp_conv=True))
        rc_model.add(kl.Dense(output_dim=1, trainable=False,
                              init="ones"))
    else:
        rc_model.add(RevCompSumPool())
        rc_model.add(Flatten())
        rc_model.add(keras.layers.core.Dense(output_dim=1, trainable=True,
                                             init="glorot_uniform", use_bias=use_bias))
    rc_model.add(kl.core.Activation("sigmoid"))
    rc_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                     loss="binary_crossentropy", metrics=["accuracy"])
    return rc_model

def get_siamese_model(parameters):
    main_input = Input(shape=(1000, 4))
    rev_input = kl.Lambda(lambda x: x[:, ::-1, ::-1])(main_input)

    s_model = Sequential([
        keras.layers.Convolution1D(
            input_shape=(1000, 4), nb_filter=16, filter_length=15),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                strides=parameters['strides']),
        Flatten(),
        keras.layers.core.Dense(output_dim=1, trainable=True,
                                init="glorot_uniform"),
    ], name="shared_layers")

    main_output = s_model(main_input)
    rev_output = s_model(rev_input)

    avg = kl.average([main_output, rev_output])

    final_out = kl.core.Activation("sigmoid")(avg)
    siamese_model = Model(inputs=main_input, outputs=final_out)
    siamese_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
    return siamese_model

# The difference between this siamese model and the one above is when the averaging takes place
def get_new_siamese_model(parameters):
    main_input = Input(shape=(1000, 4))
    rev_input = kl.Lambda(lambda x: x[:, ::-1, ::-1])(main_input)

    s_model = Sequential([
        keras.layers.Convolution1D(
            input_shape=(1000, 4), nb_filter=16, filter_length=15),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                strides=parameters['strides']),
        Flatten(),
        keras.layers.core.Dense(output_dim=1, trainable=True,
                                init="glorot_uniform"),
    ], name="shared_layers")

    main_output = s_model(main_input)
    rev_output = s_model(rev_input)

    final_out_main = kl.core.Activation("sigmoid")(main_output)
    final_out_rev = kl.core.Activation("sigmoid")(rev_output)

    avg = kl.average([final_out_main, final_out_rev])

    siamese_model = Model(inputs=main_input, outputs=avg)
    siamese_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
    return siamese_model
"""
