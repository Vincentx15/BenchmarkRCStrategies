import keras
from keras import backend as K
from keras.layers import Layer, Dense
import keras.layers as kl

import tensorflow as tf


class RegToRegConv(Layer):
    """
    Mapping from one reg layer to another
    """

    def __init__(self, reg_in, reg_out, kernel_size, kernel_initializer='glorot_uniform'):
        super(RegToRegConv, self).__init__()
        self.reg_in = reg_in
        self.reg_out = reg_out
        self.input_dim = 2 * reg_in
        self.filters = 2 * reg_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

        self.left_kernel = self.add_weight(shape=(self.kernel_size // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        # TODO : fixme ?
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
        # TODO : fixme
        if self.kernel_size % 2 == 1:
            other_half = self.half_center[:, ::-1, ::-1]
            center_kernel = (other_half + self.half_center) / 2
            kernel = K.concatenate((self.left_kernel, center_kernel, right_kernel), axis=0)
        else:
            kernel = K.concatenate((self.left_kernel, right_kernel), axis=0)
        outputs = K.conv1d(inputs,
                           kernel,
                           padding='valid')
        return outputs

    def get_config(self):
        config = {'reg_in': self.reg_in,
                  'a_out': self.a_out,
                  'b_out': self.b_out}
        base_config = super(RegToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegToIrrepConv(Layer):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_in, a_out, b_out, kernel_size, kernel_initializer='glorot_uniform'):
        super(RegToIrrepConv, self).__init__()
        self.reg_in = reg_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = 2 * reg_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

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
                           padding='valid')
        return outputs

    def get_config(self):
        config = {'reg_in': self.reg_in,
                  'a_out': self.a_out,
                  'b_out': self.b_out}
        base_config = super(RegToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IrrepToIrrepConv(Layer):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, a_in, a_out, b_in, b_out, kernel_size, kernel_initializer='glorot_uniform'):
        super(IrrepToIrrepConv, self).__init__()

        self.a_in = a_in
        self.b_in = b_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = a_in + b_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

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
                           padding='valid')
        return outputs

    def get_config(self):
        config = {'a_in': self.a_in,
                  'a_out': self.a_out,
                  'b_in': self.b_in,
                  'b_out': self.b_out}
        base_config = super(IrrepToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EquiNet(Layer):

    def __init__(self, filters=[(2, 2), (2, 2), (2, 2), (1, 0)], kernel_sizes=[5, 5, 7, 7], out_size=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        filters
        """
        super(EquiNet, self).__init__()

        assert len(filters) == len(kernel_sizes)
        # self.input_dense = 1000
        successive_shrinking = (i - 1 for i in kernel_sizes)
        self.input_dense = 1000 - sum(successive_shrinking)

        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.reg_irrep = RegToIrrepConv(reg_in=2,
                                        a_out=first_a,
                                        b_out=first_b,
                                        kernel_size=first_kernel_size)
        self.irrep_layers = []
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

        self.dense = Dense(out_size, input_dim=self.input_dense)

    def call(self, inputs):
        x = self.reg_irrep(inputs)
        # rcinputs = inputs[:, ::-1, ::-1]
        # rcx = self.reg_irrep(rcinputs)

        for irrep_layer in self.irrep_layers:
            x = irrep_layer(x)

            # rcx = irrep_layer(rcx)
            # print('first')
            # print(x[0, :5, :].numpy())
            # print('reversed')
            # print(rcx[0, -5:, :].numpy()[::-1])

        # Average two strands predictions
        x = x + x[:, ::-1, :]

        # Fla
        bs = tf.shape(x)[0]
        x = tf.reshape(x, (bs, -1))
        length = tf.shape(x)[1].numpy().item()
        assert length == self.input_dense
        x = self.dense(x)
        return x


def training_step(model, input, target):
    with tf.GradientTape() as tape:
        pred = model(input)
        loss = loss_fn(target, pred)
        grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


if __name__ == '__main__':
    pass
    import tensorflow as tf
    from keras.utils import Sequence

    tf.enable_eager_execution()


    # inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
    # outputs = EquiNet(inputs)
    # model = keras.Model(inputs, outputs)
    # model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    class gen(Sequence):
        def __getitem__(self, item):
            input = tf.random.uniform((1, 1000, 4))
            target = tf.random.uniform((1, 1000, 3))
            return input, target

        def __len__(self):
            return 10

        def __iter__(self):
            for item in (self[i] for i in range(len(self))):
                yield item


    a_1 = 1
    b_1 = 1
    a_2 = 1
    b_2 = 1

    reg_reg = RegToRegConv(reg_in=2,
                           reg_out=3,
                           kernel_size=7)

    reg_irrep = RegToIrrepConv(reg_in=2,
                               a_out=a_1,
                               b_out=b_1,
                               kernel_size=8)

    irrep_irrep = IrrepToIrrepConv(a_in=a_1,
                                   b_in=b_1,
                                   a_out=a_2,
                                   b_out=b_2,
                                   kernel_size=15)

    whole = EquiNet()

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # model = reg_irrep
    # model = irrep_irrep
    # model = reg_reg
    model = whole
    from keras_genomics.layers import RevCompConv1D

    # model = RevCompConv1D(3,10)

    # x = tf.random.uniform((1, 1000, 4))
    # x2 = x[:, ::-1, ::-1]
    # out1 = reg_irrep(x)
    # out1 = irrep_irrep(out1)
    # out2 = reg_irrep(x2)
    # out2 = irrep_irrep(out2)
    # print(out1[0, :5, :].numpy())
    # print('reversed')
    # print(out2[0, -5:, :].numpy()[::-1])

    # x = tf.random.uniform((1, 1000, 4))
    # x2 = x[:, ::-1, ::-1]
    # out1 = model(x)
    # out2 = model(x2)
    #
    # print(out1.numpy())
    # print('reversed')
    # print(out2.numpy()[::-1])

    import sys

    sys.exit()

    gen = gen()
    # input, target = gen.__getitem__(1)
    # for i in range(10):
    #     training_step(model, input, target)
    #
    # gen = iter([i for i in range(10)])
    cal = lambda: gen
    train_dataset = tf.data.Dataset.from_generator(cal, (tf.float32, tf.float32))
    for batch in train_dataset:
        print(type(batch[0]))
        model(batch[0])
