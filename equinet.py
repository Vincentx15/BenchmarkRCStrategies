from keras.engine.base_layer import InputSpec
from keras.layers.convolutional import Conv1D
from keras.layers import Layer
from keras import backend as K


class RegToIrrepConv(Layer):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_in, a_out, b_out, kernel_size, kernel_initializer='glorot_uniform', **kwargs):
        super(RegToIrrepConv, self).__init__()
        self.reg_in = reg_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = 2 * reg_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

        self.left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size[0] % 2 == 1:
            self.top_left = self.add_weight(shape=(1, self.reg_in, self.a_out),
                                            initializer=self.kernel_initializer,
                                            name='center_kernel_tl')
            self.bottom_right = self.add_weight(shape=(1, self.reg_in, self.b_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_br')

    # input_shape=(1000, 4),
    #                                      reg_in=2,
    #                                      a_out=3,
    #                                      b_out=0,
    #                                      filter_length=15,
    #                                      use_bias=False)

    # def build(self, input_shape):
    #     """
    #     Overrides the kernel construction to build a constrained one
    #     """
    #
    #     if self.data_format == 'channels_first':
    #         channel_axis = 1
    #     else:
    #         channel_axis = -1
    #     if input_shape[channel_axis] is None:
    #         raise ValueError('The channel dimension of the inputs '
    #                          'should be defined. Found `None`.')
    #     input_dim = input_shape[channel_axis]
    #
    #     # kernel_shape = self.kernel_size[0] + (input_dim, self.filters)
    #     # Filters is a_out + b_out
    #
    #     assert 2 * self.reg_in == input_dim
    #
    #
    #     self.left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, input_dim, self.filters),
    #                                        initializer=self.kernel_initializer,
    #                                        name='left_kernel')
    #
    #     # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
    #     if self.kernel_size[0] % 2 == 1:
    #         self.top_left = self.add_weight(shape=(1, self.reg_in, self.a_out),
    #                                         initializer=self.kernel_initializer,
    #                                         name='center_kernel_tl')
    #         self.bottom_right = self.add_weight(shape=(1, self.reg_in, self.b_out),
    #                                             initializer=self.kernel_initializer,
    #                                             name='center_kernel_br')
    #
    #     # For now, let's not use bias. It can be added on the a_n invariant dimensions, but not so easy to implement
    #     # in Keras
    #     if self.use_bias:
    #         raise NotImplementedError
    #     self.built = True

    def call(self, inputs):
        # Build the right part of the kernel from the left one
        # Columns are transposed, the b lines are flipped

        right_top = self.left_kernel[::-1, ::-1, :self.a_out]
        right_bottom = -self.left_kernel[::-1, ::-1, self.a_out:]
        right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing and flipping the sign of b_parts.
        if self.kernel_size[0] % 2 == 1:
            top_right = self.top_left[:, ::-1, :]
            bottom_left = -self.bottom_right[:, ::-1, :]
            left = K.concatenate((self.top_left, bottom_left), axis=2)
            right = K.concatenate((top_right, self.bottom_right), axis=2)
            center_kernel = K.concatenate((left, right), axis=1)
            kernel = K.concatenate((self.left_kernel, center_kernel, right_kernel), axis=0)
        else:
            kernel = K.concatenate((self.left_kernel, right_kernel), axis=0)
        outputs = K.conv1d(inputs,
                           kernel)
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

    def __init__(self, a_in, a_out, b_in, b_out, **kwargs):
        self.a_in = a_in
        self.a_out = a_out
        self.b_in = b_in
        self.b_out = b_out
        self.input_dim = a_in + b_in
        self.filters = a_out + b_out
        # super(IrrepToIrrepConv, self).__init__(filters=filters, **kwargs)

        self.left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, self.input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')
        # odd size
        if self.kernel_size[0] % 2 == 1:
            # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts
            # Here we initialize the non zero blocks
            self.top_left = self.add_weight(shape=(1, self.a_in, self.a_out),
                                            initializer=self.kernel_initializer,
                                            name='center_kernel_tl')
            self.bottom_right = self.add_weight(shape=(1, self.b_in, self.b_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_br')

    def build(self, input_shape):
        """
        Overrides the kernel construction to build a constrained one
        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        # kernel_shape : self.kernel_size[0], input_dim, self.filters)
        assert self.a_in + self.b_in == input_dim

        self.left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, input_dim, self.filters),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel')
        # odd size
        if self.kernel_size[0] % 2 == 1:
            # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts
            # Here we initialize the non zero blocks
            self.top_left = self.add_weight(shape=(1, self.a_in, self.a_out),
                                            initializer=self.kernel_initializer,
                                            name='center_kernel_tl')
            self.bottom_right = self.add_weight(shape=(1, self.b_in, self.b_out),
                                                initializer=self.kernel_initializer,
                                                name='center_kernel_br')

        # For now, let's not use bias. It can be added on the a_n invariant dimensions, but not so easy to implement
        # in Keras
        if self.use_bias:
            raise NotImplementedError

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

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
        if self.kernel_size[0] % 2 == 1:
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
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])
        return outputs

    def get_config(self):
        config = {'a_in': self.a_in,
                  'a_out': self.a_out,
                  'b_in': self.b_in,
                  'b_out': self.b_out}
        base_config = super(IrrepToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
