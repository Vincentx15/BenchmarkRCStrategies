from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
from keras import backend as K

import tensorflow as tf

from equinet import RegToIrrepConv, IrrepToIrrepConv, RegToRegConv, IrrepToRegConv
from equinet import IrrepActivationLayer, ToKmerLayer


# Loss Function
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tf.compat.v1.distributions.Multinomial(total_count=counts_per_example,
                                                  logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) /
            tf.cast((tf.shape(true_counts)[0]), tf.float32))


class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}


class AbstractProfileModel(object):
    def __init__(self, dataset,
                 input_seq_len=1346,
                 c_task_weight=0,
                 p_task_weight=1,
                 filters=64,
                 n_dil_layers=6,
                 conv1_kernel_size=21,
                 dil_kernel_size=3,
                 outconv_kernel_size=75,
                 is_add=True,
                 optimizer='Adam',
                 weight_decay=0.01,
                 lr=0.001,
                 kernel_initializer="glorot_uniform"):
        self.dataset = dataset
        self.input_seq_len = input_seq_len
        self.c_task_weight = c_task_weight
        self.p_task_weight = p_task_weight
        self.filters = filters
        self.n_dil_layers = n_dil_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dil_kernel_size = dil_kernel_size
        self.outconv_kernel_size = outconv_kernel_size
        self.is_add = is_add
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr = lr
        self.kernel_initializer = kernel_initializer

    def get_embedding_len(self):
        embedding_len = self.input_seq_len
        embedding_len -= (self.conv1_kernel_size - 1)
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = (2 ** i)
            embedding_len -= dilation_rate * (self.dil_kernel_size - 1)
        return embedding_len

    def get_output_profile_len(self):
        embedding_len = self.get_embedding_len()
        out_profile_len = embedding_len - (self.outconv_kernel_size - 1)
        return out_profile_len

    def trim_flanks_of_conv_layer(self, conv_layer, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            lambda x: x[:,
                      int(0.5 * (width_to_trim)):-(width_to_trim - int(0.5 * (width_to_trim)))],
            output_shape=(output_len, filters))(conv_layer)
        return layer

    def get_inputs(self):
        out_pred_len = self.get_output_profile_len()

        inp = kl.Input(shape=(self.input_seq_len, 4), name='sequence')
        if self.dataset == "SPI1":
            bias_counts_input = kl.Input(shape=(1,), name="control_logcount")
            bias_profile_input = kl.Input(shape=(out_pred_len, 2),
                                          name="control_profile")
        else:
            bias_counts_input = kl.Input(shape=(2,), name="patchcap.logcount")
            # if working with raw counts, go from logcount->count
            bias_profile_input = kl.Input(shape=(out_pred_len, 2),
                                          name="patchcap.profile")
        return inp, bias_counts_input, bias_profile_input

    def get_names(self):
        if self.dataset == "SPI1":
            countouttaskname = "task0_logcount"
            profileouttaskname = "task0_profile"
        elif self.dataset == 'NANOG':
            countouttaskname = "CHIPNexus.NANOG.logcount"
            profileouttaskname = "CHIPNexus.NANOG.profile"
        elif self.dataset == "OCT4":
            countouttaskname = "CHIPNexus.OCT4.logcount"
            profileouttaskname = "CHIPNexus.OCT4.profile"
        elif self.dataset == "KLF4":
            countouttaskname = "CHIPNexus.KLF4.logcount"
            profileouttaskname = "CHIPNexus.KLF4.profile"
        elif self.dataset == "SOX2":
            countouttaskname = "CHIPNexus.SOX2.logcount"
            profileouttaskname = "CHIPNexus.SOX2.profile"
        return countouttaskname, profileouttaskname

    def get_keras_model(self):
        raise NotImplementedError()


class RcBPNetArch(AbstractProfileModel):
    def __init__(self, kmers=1, **kwargs):
        super().__init__(**kwargs)
        self.kmers = kmers

    def get_keras_model(self):
        from equinet import ToKmerLayer
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        countouttaskname, profileouttaskname = self.get_names()

        out_pred_len = self.get_output_profile_len()
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1)

        kmer_inp = ToKmerLayer(k=self.kmers)(inp)
        first_conv = RevCompConv1D(filters=self.filters,
                                   kernel_size=self.conv1_kernel_size - self.kmers + 1,
                                   kernel_initializer=self.kernel_initializer,
                                   padding='valid',
                                   activation='relu')(kmer_inp)

        prev_layers = first_conv
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2 ** i

            conv_output = RevCompConv1D(filters=self.filters,
                                        kernel_size=self.dil_kernel_size,
                                        kernel_initializer=self.kernel_initializer,
                                        padding='valid',
                                        activation='relu',
                                        dilation_rate=dilation_rate)(prev_layers)

            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)

            curr_layer_size = (curr_layer_size - width_to_trim)

            prev_layers = self.trim_flanks_of_conv_layer(
                conv_layer=prev_layers, output_len=curr_layer_size,
                width_to_trim=width_to_trim, filters=2 * self.filters)

            if (self.is_add):
                prev_layers = kl.add([prev_layers, conv_output])
            else:
                prev_layers = kl.average([prev_layers, conv_output])

        combined_conv = prev_layers

        # Counts prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Reshape((-1,), name=countouttaskname)(
            RevCompConv1D(filters=1, kernel_size=1, kernel_initializer=self.kernel_initializer)(
                kl.Reshape((1, -1))(kl.concatenate([
                    # concatenation of the bias layer both before and after
                    # is needed for rc symmetry
                    kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
                    gap_combined_conv,
                    bias_counts_input], axis=-1))))

        # Profile prediction
        profile_out_prebias = RevCompConv1D(
            filters=1, kernel_size=self.outconv_kernel_size,
            kernel_initializer=self.kernel_initializer, padding='valid')(combined_conv)

        profile_out = RevCompConv1D(
            filters=1, kernel_size=1, name=profileouttaskname, kernel_initializer=self.kernel_initializer)(
            kl.concatenate([
                # concatenation of the bias layer both before and after
                # is needed for rc symmetry
                kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
                profile_out_prebias,
                bias_profile_input], axis=-1))

        model = keras.models.Model(
            inputs=[inp, bias_counts_input, bias_profile_input],
            outputs=[count_out, profile_out])

        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight])

        return model


class SiameseBPNetArch(AbstractProfileModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def trim_flanks_of_conv_layer_revcomp(self, conv_layer, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            lambda x: x[:,
                      (width_to_trim - int(0.5 * (width_to_trim))):-int(0.5 * (width_to_trim))],
            output_shape=(output_len, filters))(conv_layer)
        return layer

    def get_keras_model(self):
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        rev_inp = kl.Lambda(lambda x: x[:, ::-1, ::-1])(inp)

        countouttaskname, profileouttaskname = self.get_names()

        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               kernel_initializer=self.kernel_initializer,
                               padding='valid',
                               activation='relu')
        first_conv_fwd = first_conv(inp)
        first_conv_rev = first_conv(rev_inp)

        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1)

        prev_layers_fwd = first_conv_fwd
        prev_layers_rev = first_conv_rev

        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2 ** i
            conv_output = kl.Conv1D(self.filters, kernel_size=self.dil_kernel_size,
                                    padding='valid',
                                    kernel_initializer=self.kernel_initializer,
                                    activation='relu',
                                    dilation_rate=dilation_rate)

            conv_output_fwd = conv_output(prev_layers_fwd)
            conv_output_rev = conv_output(prev_layers_rev)

            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)

            curr_layer_size = (curr_layer_size - width_to_trim)

            prev_layers_fwd = self.trim_flanks_of_conv_layer(
                conv_layer=prev_layers_fwd, output_len=curr_layer_size,
                width_to_trim=width_to_trim, filters=self.filters)

            prev_layers_rev = self.trim_flanks_of_conv_layer_revcomp(
                conv_layer=prev_layers_rev, output_len=curr_layer_size,
                width_to_trim=width_to_trim, filters=self.filters)

            if (self.is_add):
                prev_layers_fwd = kl.add([prev_layers_fwd, conv_output_fwd])
                prev_layers_rev = kl.add([prev_layers_rev, conv_output_rev])
            else:
                prev_layers_fwd = kl.average([prev_layers_fwd, conv_output_fwd])
                prev_layers_rev = kl.average([prev_layers_rev, conv_output_rev])

            combined_conv_fwd = prev_layers_fwd
            combined_conv_rev = prev_layers_rev

        # Counts Prediction
        counts_dense_layer = kl.Dense(2, kernel_initializer=self.kernel_initializer, )
        gap_combined_conv_fwd = kl.GlobalAvgPool1D()(combined_conv_fwd)
        gap_combined_conv_rev = kl.GlobalAvgPool1D()(combined_conv_rev)

        main_count_out_fwd = counts_dense_layer(
            kl.concatenate([gap_combined_conv_fwd, bias_counts_input], axis=-1))

        main_count_out_rev = counts_dense_layer(
            kl.concatenate([bias_counts_input, gap_combined_conv_rev], axis=-1))
        rc_rev_count_out = kl.Lambda(lambda x: x[:, ::-1])(main_count_out_rev)

        avg_count_out = kl.Average(name=countouttaskname)(
            [main_count_out_fwd, rc_rev_count_out])

        # Profile Prediction
        profile_penultimate_conv = kl.Conv1D(filters=2,
                                             kernel_size=self.outconv_kernel_size,
                                             kernel_initializer=self.kernel_initializer,
                                             padding='valid')
        profile_final_conv = kl.Conv1D(2, kernel_size=1, kernel_initializer=self.kernel_initializer, )

        profile_out_prebias_fwd = profile_penultimate_conv(combined_conv_fwd)
        main_profile_out_fwd = profile_final_conv(kl.concatenate(
            [profile_out_prebias_fwd, bias_profile_input], axis=-1))

        profile_out_prebias_rev = profile_penultimate_conv(combined_conv_rev)
        rev_bias_profile_input = kl.Lambda(lambda x: x[:, ::-1, :])(bias_profile_input)
        main_profile_out_rev = profile_final_conv(kl.concatenate(
            [profile_out_prebias_rev, rev_bias_profile_input], axis=-1))
        rc_rev_profile_out = kl.Lambda(lambda x: x[:, ::-1, ::-1])(main_profile_out_rev)

        avg_profile_out = kl.Average(name=profileouttaskname)(
            [main_profile_out_fwd, rc_rev_profile_out])

        model = keras.models.Model(
            inputs=[inp, bias_counts_input, bias_profile_input],
            outputs=[avg_count_out, avg_profile_out])

        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight])
        return model


class StandardBPNetArch(AbstractProfileModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_keras_model(self):

        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        countouttaskname, profileouttaskname = self.get_names()

        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               kernel_initializer=self.kernel_initializer,
                               padding='valid',
                               activation='relu')(inp)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1)

        prev_layers = first_conv
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2 ** i
            conv_output = kl.Conv1D(self.filters, kernel_size=self.dil_kernel_size,
                                    kernel_initializer=self.kernel_initializer,
                                    padding='valid',
                                    activation='relu',
                                    dilation_rate=dilation_rate)(prev_layers)

            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)

            curr_layer_size = (curr_layer_size - width_to_trim)
            prev_layers = self.trim_flanks_of_conv_layer(
                conv_layer=prev_layers, output_len=curr_layer_size,
                width_to_trim=width_to_trim, filters=self.filters)

            if self.is_add:
                prev_layers = kl.add([prev_layers, conv_output])
            else:
                prev_layers = kl.average([prev_layers, conv_output])

        combined_conv = prev_layers

        # Counts Prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Dense(2, kernel_initializer=self.kernel_initializer, name=countouttaskname)(
            kl.concatenate([gap_combined_conv, bias_counts_input], axis=-1))

        # Profile Prediction
        profile_out_prebias = kl.Conv1D(filters=2,
                                        kernel_size=self.outconv_kernel_size,
                                        kernel_initializer=self.kernel_initializer,
                                        padding='valid')(combined_conv)
        profile_out = kl.Conv1D(2, kernel_size=1, kernel_initializer=self.kernel_initializer, name=profileouttaskname)(
            kl.concatenate([profile_out_prebias, bias_profile_input], axis=-1))

        model = keras.models.Model(
            inputs=[inp, bias_counts_input, bias_profile_input],
            outputs=[count_out, profile_out])

        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight])
        return model


class EquiNetBP(kl.Layer):
    def __init__(self,
                 dataset,
                 input_seq_len=1346,
                 c_task_weight=0,
                 p_task_weight=1,
                 filters=((64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)),
                 kernel_sizes=(21, 3, 3, 3, 3, 3, 3, 75),
                 outconv_kernel_size=75,
                 weight_decay=0.01,
                 optimizer='Adam',
                 lr=0.001,
                 kernel_initializer="glorot_uniform",
                 seed=42,
                 is_add=True,
                 kmers=1,
                 **kwargs):
        super(EquiNetBP, self).__init__(**kwargs)

        self.dataset = dataset
        self.input_seq_len = input_seq_len
        self.c_task_weight = c_task_weight
        self.p_task_weight = p_task_weight
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.outconv_kernel_size = outconv_kernel_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr = lr
        self.learning_rate = lr
        self.kernel_initializer = kernel_initializer
        self.seed = seed
        self.is_add = is_add
        self.n_dil_layers = len(filters) - 1

        # Add k-mers, if k=1, it's just a placeholder
        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        self.conv1_kernel_size = kernel_sizes[0] - self.kmers + 1
        reg_in = self.to_kmer.features // 2
        first_a, first_b = filters[0]
        self.first_conv = RegToIrrepConv(reg_in=reg_in,
                                         a_out=first_a,
                                         b_out=first_b,
                                         kernel_size=self.conv1_kernel_size,
                                         kernel_initializer=self.kernel_initializer,
                                         padding='valid')
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        # Now add the intermediate layer : sequence of conv, activation
        self.irrep_layers = []
        self.activation_layers = []
        self.croppings = []
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            dilation_rate = 2 ** i
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a,
                b_in=prev_b,
                a_out=next_a,
                b_out=next_b,
                kernel_size=kernel_sizes[i],
                dilatation=dilation_rate
            ))
            self.croppings.append((kernel_sizes[i] - 1) * dilation_rate)
            # self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            self.activation_layers.append(IrrepActivationLayer(a=next_a, b=next_b))

        self.last_a, self.last_b = filters[-1]
        self.prebias = IrrepToRegConv(reg_out=1,
                                      a_in=self.last_a,
                                      b_in=self.last_b,
                                      kernel_size=self.outconv_kernel_size,
                                      kernel_initializer=self.kernel_initializer,
                                      padding='valid')
        self.last = RegToRegConv(reg_in=3,
                                 reg_out=1,
                                 kernel_size=1,
                                 kernel_initializer=self.kernel_initializer,
                                 padding='valid')

        self.last_count = IrrepToRegConv(a_in=self.last_a + 2,
                                         b_in=self.last_b,
                                         reg_out=1,
                                         kernel_size=1,
                                         kernel_initializer=self.kernel_initializer)

    def get_output_profile_len(self):
        embedding_len = self.input_seq_len
        embedding_len -= (self.conv1_kernel_size - 1)
        for cropping in self.croppings:
            embedding_len -= cropping
        out_profile_len = embedding_len - (self.outconv_kernel_size - 1)
        return out_profile_len

    def trim_flanks_of_inputs(self, inputs, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            function=lambda x: x[:, int(0.5 * (width_to_trim)):-(width_to_trim - int(0.5 * (width_to_trim)))],
            output_shape=(output_len, filters))(inputs)
        return layer

    def get_inputs(self):
        out_pred_len = self.get_output_profile_len()

        inp = kl.Input(shape=(self.input_seq_len, 4), name='sequence')
        if self.dataset == "SPI1":
            bias_counts_input = kl.Input(shape=(1,), name="control_logcount")
            bias_profile_input = kl.Input(shape=(out_pred_len, 2),
                                          name="control_profile")
        else:
            bias_counts_input = kl.Input(shape=(2,), name="patchcap.logcount")
            # if working with raw counts, go from logcount->count
            bias_profile_input = kl.Input(shape=(1000, 2),
                                          name="patchcap.profile")
        return inp, bias_counts_input, bias_profile_input

    def get_names(self):
        if self.dataset == "SPI1":
            countouttaskname = "task0_logcount"
            profileouttaskname = "task0_profile"
        elif self.dataset == 'NANOG':
            countouttaskname = "CHIPNexus.NANOG.logcount"
            profileouttaskname = "CHIPNexus.NANOG.profile"
        elif self.dataset == "OCT4":
            countouttaskname = "CHIPNexus.OCT4.logcount"
            profileouttaskname = "CHIPNexus.OCT4.profile"
        elif self.dataset == "KLF4":
            countouttaskname = "CHIPNexus.KLF4.logcount"
            profileouttaskname = "CHIPNexus.KLF4.profile"
        elif self.dataset == "SOX2":
            countouttaskname = "CHIPNexus.SOX2.logcount"
            profileouttaskname = "CHIPNexus.SOX2.profile"
        else:
            raise ValueError("The dataset asked does not exist")
        return countouttaskname, profileouttaskname

    def get_keras_model(self):
        """
        Make a first convolution, then use skip connections with dilatations (that shrink the input)
        to get 'combined_conv'

        Then create two heads :
         - one is used to predict counts (and has a weight of zero in the loss)
         - one is used to predict the profile
        """
        sequence_input, bias_counts_input, bias_profile_input = self.get_inputs()

        kmer_inputs = self.to_kmer(sequence_input)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1) - (self.kmers - 1)
        prev_layers = self.first_conv(kmer_inputs)

        for i, (conv_layer, activation_layer, cropping) in enumerate(zip(self.irrep_layers,
                                                                         self.activation_layers,
                                                                         self.croppings)):

            conv_output = conv_layer(prev_layers)
            conv_output = activation_layer(conv_output)
            curr_layer_size = curr_layer_size - cropping

            trimmed_prev_layers = self.trim_flanks_of_inputs(inputs=prev_layers,
                                                             output_len=curr_layer_size,
                                                             width_to_trim=cropping,
                                                             filters=self.filters[i][0] + self.filters[i][1])
            if self.is_add:
                prev_layers = kl.add([trimmed_prev_layers, conv_output])
            else:
                prev_layers = kl.average([trimmed_prev_layers, conv_output])

        combined_conv = prev_layers

        countouttaskname, profileouttaskname = self.get_names()

        # ============== Placeholder for counts =================
        count_out = kl.Lambda(lambda x: x, name=countouttaskname)(bias_counts_input)

        # gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        # stacked = kl.Reshape((1, -1))(kl.concatenate([
        #     # concatenation of the bias layer both before and after
        #     # is needed for rc symmetry
        #     kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
        #     gap_combined_conv,
        #     bias_counts_input], axis=-1))
        # convout = self.last_count(stacked)
        # count_out = kl.Reshape((-1,), name=countouttaskname)(convout)

        # ============== Profile prediction ======================
        profile_out_prebias = self.prebias(combined_conv)

        # # concatenation of the bias layer both before and after is needed for rc symmetry
        concatenated = kl.concatenate([kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
                                       profile_out_prebias,
                                       bias_profile_input], axis=-1)
        profile_out = self.last(concatenated)
        profile_out = kl.Lambda(lambda x: x, name=profileouttaskname)(profile_out)

        model = keras.models.Model(
            inputs=[sequence_input, bias_counts_input, bias_profile_input],
            outputs=[count_out, profile_out])
        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight])
        # print(model.summary())
        return model

    def eager_call(self, sequence_input, bias_counts_input, bias_profile_input):
        """
        Testing only
        """
        kmer_inputs = self.to_kmer(sequence_input)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1) - (self.kmers - 1)
        prev_layers = self.first_conv(kmer_inputs)

        for i, (conv_layer, activation_layer, cropping) in enumerate(zip(self.irrep_layers,
                                                                         self.activation_layers,
                                                                         self.croppings)):

            conv_output = conv_layer(prev_layers)
            conv_output = activation_layer(conv_output)
            curr_layer_size = curr_layer_size - cropping

            trimmed_prev_layers = self.trim_flanks_of_inputs(inputs=prev_layers,
                                                             output_len=curr_layer_size,
                                                             width_to_trim=cropping,
                                                             filters=self.filters[i][0] + self.filters[i][1])
            if self.is_add:
                prev_layers = kl.add([trimmed_prev_layers, conv_output])
            else:
                prev_layers = kl.average([trimmed_prev_layers, conv_output])

        combined_conv = prev_layers

        # Placeholder for counts
        count_out = bias_counts_input

        # Profile prediction
        profile_out_prebias = self.prebias(combined_conv)

        # concatenation of the bias layer both before and after is needed for rc symmetry
        rc_profile_input = bias_profile_input[:, :, ::-1]
        concatenated = K.concatenate([rc_profile_input,
                                      profile_out_prebias,
                                      bias_profile_input], axis=-1)

        profile_out = self.last(concatenated)

        return count_out, profile_out


if __name__ == '__main__':

    import tensorflow as tf
    from equinet import BPNGenerator

    eager = False

    if eager:
        tf.enable_eager_execution()

        rc_model = RcBPNetArch(dataset='SOX2').get_keras_model()
        print(rc_model.summary())
        rc_model = EquiNetBP(dataset='SOX2', kmers=4).get_keras_model()
        print(rc_model.summary())
        generator = BPNGenerator(inlen=1346, outfeat=2, outlen=1000, eager=eager, length=3)
        rc_model.fit_generator(generator)

    else:
        pass
        generator = BPNGenerator(inlen=1346, outfeat=2, outlen=1000, eager=eager, bs=2)
        inputs = next(iter(generator))
        a, b, c = inputs[0].values()
        rc_model = EquiNetBP(dataset='SOX2')
        rc_model.eager_call(a, b, c)
