import keras
from keras import backend as K
from keras.layers import Layer


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
            self.top_left = self.add_weight(shape=(1, self.reg_in, self.a_out),
                                            initializer=self.kernel_initializer,
                                            name='center_kernel_tl')
            self.bottom_right = self.add_weight(shape=(1, self.reg_in, self.b_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_br')

    def call(self, inputs, **kwargs):
        # Build the right part of the kernel from the left one
        # Columns are transposed, the b lines are flipped

        right_top = self.left_kernel[::-1, ::-1, :self.a_out]
        right_bottom = -self.left_kernel[::-1, ::-1, self.a_out:]
        right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing and flipping the sign of b_parts.
        if self.kernel_size % 2 == 1:
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
                           padding='same')
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
            self.top_left = self.add_weight(shape=(1, self.a_in, self.a_out),
                                            initializer=self.kernel_initializer,
                                            name='center_kernel_tl')
            self.bottom_right = self.add_weight(shape=(1, self.b_in, self.b_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_br')

    def call(self, inputs):
        # Build the right part of the kernel from the left one
        # Here being on a 'b' part means flipping so the block diagonal is flipped
        right_top_left = self.left_kernel[::-1, :self.a_in, :self.a_out]
        right_top_right = -self.left_kernel[::-1, self.a_in:, :self.a_out]
        right_bottom_left = -self.left_kernel[::-1, :self.a_in, self.a_out:]
        right_bottom_right = self.left_kernel[::-1, self.a_in:, self.a_out:]
        right_left = K.concatenate((right_top_left, right_bottom_left), axis=2)
        right_right = K.concatenate((right_top_right, right_bottom_right), axis=2)
        right_kernel = K.concatenate((right_left, right_right), axis=1)

        # Extra steps are needed for building the middle part when using the odd size
        if self.kernel_size % 2 == 1:
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
                           padding='same')
        return outputs

    def get_config(self):
        config = {'a_in': self.a_in,
                  'a_out': self.a_out,
                  'b_in': self.b_in,
                  'b_out': self.b_out}
        base_config = super(IrrepToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EquiNet(Layer):

    def __init__(self, filters=[(2, 2), (2, 2), (2, 0)], kernel_sizes=[5, 5, 10, 10]):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        filters
        """
        super(EquiNet, self).__init__()

        assert len(filters) + 1 == len(kernel_sizes)

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

    def call(self, inputs):
        x = self.reg_irrep(inputs)
        for irrep_layer in self.irrep_layers:
            x = irrep_layer(x)
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

    reg_irrep = RegToIrrepConv(reg_in=2,
                               a_out=3,
                               b_out=0,
                               kernel_size=15)

    irrep_irrep = IrrepToIrrepConv(a_in=3,
                                   b_in=1,
                                   a_out=3,
                                   b_out=0,
                                   kernel_size=15)

    whole = EquiNet()


    # inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
    # outputs = reg_irrep(inputs)
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


    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = reg_irrep
    # model = irrep_irrep
    # model = whole

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
