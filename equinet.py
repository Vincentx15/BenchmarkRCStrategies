from keras.engine.base_layer import InputSpec
from keras.layers.convolutional import Conv1D
from keras import backend as K


class RegToIrrepConv(Conv1D):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_in, a_out, b_out, **kwargs):
        self.reg_in = reg_in
        self.a_out = a_out
        self.b_out = b_out
        filters = a_out + b_out
        super(RegToIrrepConv, self).__init__(filters=filters, **kwargs)

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
        # kernel_shape = self.kernel_size[0] + (input_dim, self.filters)
        assert 2 * self.reg_in == input_dim

        # Filters is a_out + b_out
        # columns are transposed, the b lines are flipped

        left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, input_dim, self.filters),
                                      initializer=self.kernel_initializer,
                                      name='left_kernel')

        right_top = left_kernel[::-1, ::-1, :self.a_out]
        right_bottom = -left_kernel[::-1, ::-1, self.a_out:]
        right_kernel = K.concatenate((right_top, right_bottom), axis=2)

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size[0] % 2 == 1:
            top_left = self.add_weight(shape=(1, self.reg_in, self.a_out),
                                       initializer=self.kernel_initializer,
                                       name='center_kernel_tl')
            bottom_right = self.add_weight(shape=(1, self.reg_in, self.b_out),
                                           initializer=self.kernel_initializer,
                                           name='center_kernel_br')
            top_right = top_left[:, ::-1, :]
            bottom_left = -bottom_right[:, ::-1, :]
            left = K.concatenate((top_left, bottom_left), axis=2)
            right = K.concatenate((top_right, bottom_right), axis=2)
            center_kernel = K.concatenate((left, right), axis=1)
            self.kernel = K.concatenate((left_kernel, center_kernel, right_kernel), axis=0)
        else:
            self.kernel = K.concatenate((left_kernel, right_kernel), axis=0)

        # For now, let's not use bias. It can be added on the a_n invariant dimensions, but not so easy to implement
        # in Keras
        if self.use_bias:
            raise NotImplementedError
            # self.bias = self.add_weight(shape=(self.filters,),
            #                             initializer=self.bias_initializer,
            #                             name='bias',
            #                             regularizer=self.bias_regularizer,
            #                             constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        # self.input_spec = InputSpec(ndim=self.rank + 2,
        #                             axes={channel_axis: input_dim})
        self.built = True

    # def call(self, inputs):
    #     outputs = K.conv1d(inputs, self.kernel,
    #                        strides=self.strides[0],
    #                        padding=self.padding,
    #                        data_format=self.data_format,
    #                        dilation_rate=self.dilation_rate[0])
    #     return outputs

    def get_config(self):
        config = {'reg_in': self.reg_in,
                  'a_out': self.a_out,
                  'b_out': self.b_out}
        base_config = super(RegToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IrrepToIrrepConv(Conv1D):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, a_in, a_out, b_in, b_out, **kwargs):
        self.a_in = a_in
        self.a_out = a_out
        self.b_in = b_in
        self.b_out = b_out
        filters = a_out + b_out
        super(IrrepToIrrepConv, self).__init__(filters=filters, **kwargs)

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
        # kernel_shape = self.kernel_size[0] + (input_dim, self.filters)
        assert self.a_in + self.b_in == input_dim

        left_kernel = self.add_weight(shape=(self.kernel_size[0] // 2, input_dim, self.filters),
                                      initializer=self.kernel_initializer,
                                      name='left_kernel')
        # right_kernel = left_kernel[::-1, :, :]
        # right_kernel[:, -self.b_in:, :] = -right_kernel[:, -self.b_in:, :]
        # right_kernel[:, :, -self.b_out:] = -right_kernel[:, :, -self.b_out:]

        right_top_left = left_kernel[::-1, :self.a_in, :self.a_out]
        right_top_right = -left_kernel[::-1, self.a_in:, :self.a_out]
        right_bottom_left = -left_kernel[::-1, :self.a_in, self.a_out:]
        right_bottom_right = left_kernel[::-1, self.a_in:, self.a_out:]
        right_left = K.concatenate((right_top_left, right_bottom_left), axis=2)
        right_right = K.concatenate((right_top_right, right_bottom_right), axis=2)
        right_kernel = K.concatenate((right_left, right_right), axis=1)

        # odd size
        if self.kernel_size[0] % 2 == 1:
            # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts:
            top_left = self.add_weight(shape=(1, self.a_in, self.a_out),
                                       initializer=self.kernel_initializer,
                                       name='center_kernel_tl')
            bottom_right = self.add_weight(shape=(1, self.b_in, self.b_out),
                                           initializer=self.kernel_initializer,
                                           name='center_kernel_br')
            bottom_left = K.zeros(shape=(1, self.a_in, self.b_out))
            top_right = K.zeros(shape=(1, self.b_in, self.a_out))
            left = K.concatenate((top_left, bottom_left), axis=2)
            right = K.concatenate((top_right, bottom_right), axis=2)
            center_kernel = K.concatenate((left, right), axis=1)
            self.kernel = K.concatenate((left_kernel, center_kernel, right_kernel), axis=0)

        else:
            self.kernel = K.concatenate((left_kernel, right_kernel), axis=0)

        # For now, let's not use bias. It can be added on the a_n invariant dimensions, but not so easy to implement
        # in Keras
        if self.use_bias:
            raise NotImplementedError
            # self.bias = self.add_weight(shape=(self.filters,),
            #                             initializer=self.bias_initializer,
            #                             name='bias',
            #                             regularizer=self.bias_regularizer,
            #                             constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    # def call(self, inputs):
    #     outputs = K.conv1d(inputs, self.kernel,
    #                        strides=self.strides[0],
    #                        padding=self.padding,
    #                        data_format=self.data_format,
    #                        dilation_rate=self.dilation_rate[0])
    #     return outputs

    def get_config(self):
        config = {'a_in': self.a_in,
                  'a_out': self.a_out,
                  'b_in': self.b_in,
                  'b_out': self.b_out}
        base_config = super(IrrepToIrrepConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
