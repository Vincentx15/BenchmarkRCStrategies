import collections
import os
import numpy as np
import gzip
from collections import namedtuple
from sklearn.metrics import roc_auc_score

import tensorflow as tf

# tf.enable_eager_execution()

import keras
from keras import backend as K
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.callbacks import History
from keras.engine import Layer
from keras.models import Sequential
import keras.layers as kl
from keras.engine.base_layer import InputSpec
from keras import initializers

from keras_genomics.layers import RevCompConv1D
import momma_dragonn
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
# from seqdataloader.batchproducers.coordbased.coordbatchproducers import DownsampleNegativesCoordsBatchProducer
# from seqdataloader.batchproducers.coordbased.coordbatchproducers import SimpleCoordsBatchProducer
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import AbstractCountAndProfileTransformer
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import LogCountsPlusOne
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import SmoothProfiles
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import BigWigReader
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import smooth_profiles
# from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import rolling_window
# from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
# from seqdataloader.batchproducers.coordbased.coordbatchtransformers import AbstractCoordBatchTransformer
# from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp

from BinaryArchs import get_rc_model, get_reg_model
import equinet


def apply_mask(tomask, mask):
    if isinstance(tomask, dict):
        return dict([(key, val[mask]) for key, val in tomask.items()])
    elif isinstance(tomask, list):
        return [x[mask] for x in tomask]
    else:
        return tomask[mask]


class KerasBatchGenerator(keras.utils.Sequence):
    """
    Args:
        coordsbatch_producer (KerasSequenceApiCoordsBatchProducer)
        inputs_coordstovals (CoordsToVals)
        targets_coordstovals (CoordsToVals)
        sampleweights_coordstovals (CoordsToVals)
        coordsbatch_transformer (AbstracCoordBatchTransformer)
        qc_func (callable): function that can be used to filter
            out poor-quality sequences.
        sampleweights_coordstoval: either this argument or
            sampleweights_from_inputstargets could be used to
            specify sample weights. sampleweights_coordstoval
            takes a batch of coords as inputs.
        sampleweights_from_inputstargets: either this argument or
            sampleweights_coordstoval could be used to
            specify sample weights. sampleweights_from_inputstargets
            takes the inputs and targets values to generate the weights.
    """

    def __init__(self, coordsbatch_producer,
                 inputs_coordstovals,
                 targets_coordstovals,
                 coordsbatch_transformer=None,
                 qc_func=None,
                 sampleweights_coordstovals=None,
                 sampleweights_from_inputstargets=None):
        self.coordsbatch_producer = coordsbatch_producer
        self.inputs_coordstovals = inputs_coordstovals
        self.targets_coordstovals = targets_coordstovals
        self.coordsbatch_transformer = coordsbatch_transformer
        self.sampleweights_coordstovals = sampleweights_coordstovals
        self.sampleweights_from_inputstargets = \
            sampleweights_from_inputstargets
        if sampleweights_coordstovals is not None:
            assert sampleweights_from_inputstargets is None
        if sampleweights_from_inputstargets is not None:
            assert sampleweights_coordstovals is None
        self.qc_func = qc_func

    def __getitem__(self, index):
        coords_batch = self.coordsbatch_producer[index]
        if (self.coordsbatch_transformer is not None):
            coords_batch = self.coordsbatch_transformer(coords_batch)
        inputs = self.inputs_coordstovals(coords_batch)
        if (self.targets_coordstovals is not None):
            targets = self.targets_coordstovals(coords_batch)
        else:
            targets = None
        if (self.qc_func is not None):
            qc_mask = self.qc_func(inputs=inputs, targets=targets)
            inputs = apply_mask(tomask=inputs, mask=qc_mask)
            if (targets is not None):
                targets = apply_mask(tomask=targets, mask=qc_mask)
        else:
            qc_mask = None
        if (self.sampleweights_coordstovals is not None):
            sample_weights = self.sampleweights_coordstovals(coords_batch)
            return (inputs, targets, sample_weights)
        elif (self.sampleweights_from_inputstargets is not None):
            sample_weights = self.sampleweights_from_inputstargets(
                inputs=inputs, targets=targets)
            return (inputs, targets, sample_weights)
        else:
            if (self.targets_coordstovals is not None):
                return (inputs, targets)
            else:
                return inputs

    def __len__(self):
        return len(self.coordsbatch_producer)

    def on_epoch_end(self):
        self.coordsbatch_producer.on_epoch_end()


def get_new_coors_around_center(coors, center_size_to_use):
    new_coors = []
    for coor in coors:
        coor_center = int(0.5 * (coor.start + coor.end))
        left_flank = int(0.5 * center_size_to_use)
        right_flank = center_size_to_use - left_flank
        new_start = coor_center - left_flank
        new_end = coor_center + right_flank
        new_coors.append(Coordinates(chrom=coor.chrom,
                                     start=new_start, end=new_end,
                                     isplusstrand=coor.isplusstrand))
    return new_coors


class CoordsToVals(object):

    def __call__(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates`):
        Returns:
            numpy ndarray OR list of ndarrays OR a dict of mode_name->ndarray.
              Returns a list of ndarrays if returning multiple modes.
              Alternatively, returns a dict where key is the mode name
              and the value is the ndarray for the mode.
        """
        raise NotImplementedError()


class CoordsToValsJoiner(CoordsToVals):

    def __init__(self, coordstovals_list):
        """
        Joins batches returned by other CoordsToVals objects
        Args:
            coorstovals_list (:obj:`list` of :obj:`CoordsToVals`): List of
                CoordsToVals whose values to combine
        """
        self.coordstovals_list = coordstovals_list

    def __call__(self, coors):
        batch_to_return = None
        for idx, coordstovals_obj in enumerate(self.coordstovals_list):
            the_batch = coordstovals_obj(coors=coors)
            assert the_batch is not None
            if isinstance(the_batch, dict):
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, dict))), (
                        "coordstovals object at idx" + str(idx)
                        + " returned a dict, but previous coordstovals"
                        + " objects had a return type incompatible with this")
                if (batch_to_return is None):
                    batch_to_return = {}
                for key in the_batch:
                    assert key not in batch_to_return, (
                            "coordstovals object at idx" + str(idx)
                            + " returned a dict with a key of " + key
                            + ", which collides with a pre-existing key returned by"
                            + " another coordstovals object")
                batch_to_return.update(the_batch)
            else:
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, list))), (
                        "coordstovals object at idx" + str(idx)
                        + " returned a type incompatible with dict, but previous"
                        + " coordstovals objects had a return type of dict")
                if (isinstance(the_batch, list) == False):
                    the_batch = [the_batch]
                if (batch_to_return is None):
                    batch_to_return = []
                batch_to_return.extend(the_batch)
        if (batch_to_return is None):
            batch_to_return = []
        return batch_to_return


class AbstractSingleNdarrayCoordsToVals(CoordsToVals):

    def __init__(self, mode_name=None):
        """
        Args:
            mode_name (:obj:`str`, optional): default None. If None, then
                the return of __call__ will be a numpy ndarray. Otherwise, it
                will be a dictionary with a key of mode_name and a value being
                the numpy ndarray.
        """
        self.mode_name = mode_name

    def _get_ndarray(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates):
            
        Returns:
            numpy ndarray
        """
        raise NotImplementedError()

    def __call__(self, coors):
        ndarray = self._get_ndarray(coors)
        if (self.mode_name is None):
            return ndarray
        else:
            return {self.mode_name: ndarray}


Coordinates = namedtuple("Coordinates",
                         ["chrom", "start", "end", "isplusstrand"])
Coordinates.__new__.__defaults__ = (True,)


class SimpleLookup(AbstractSingleNdarrayCoordsToVals):

    def __init__(self, lookup_file,
                 transformation=None,
                 default_returnval=0.0, **kwargs):
        super(SimpleLookup, self).__init__(**kwargs)
        self.lookup_file = lookup_file
        self.transformation = transformation
        self.default_returnval = default_returnval
        self.lookup = {}
        self.num_labels = None
        for line in (gzip.open(self.lookup_file) if ".gz"
                                                    in self.lookup_file else open(self.lookup_file)):
            (chrom, start_str, end_str, *labels) = \
                line.decode("utf-8").rstrip().split("\t")
            coord = Coordinates(chrom=chrom,
                                start=int(start_str),
                                end=int(end_str))
            labels = [(self.transformation(float(x))
                       if self.transformation is not None else float(x))
                      for x in labels]
            self.lookup[(coord.chrom, coord.start, coord.end)] = labels
            if (self.num_labels is None):
                self.num_labels = len(labels)
            else:
                assert len(labels) == self.num_labels, (
                        "Unequal label lengths; " + str(len(labels), self.num_labels))

    def _get_ndarray(self, coors):
        to_return = []
        for coor in coors:
            if (coor.chrom, coor.start, coor.end) not in self.lookup:
                to_return.append(np.ones(self.num_labels)
                                 * self.default_returnval)
            else:
                to_return.append(
                    self.lookup[(coor.chrom, coor.start, coor.end)])
        return np.array(to_return)


def get_generators(TF, seq_len, is_aug, seed):
    inputs_coordstovals = coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="data/hg19.genome.fa",
        center_size_to_use=seq_len)

    targets_coordstovals = SimpleLookup(
        lookup_file=f"data/{TF}/{TF}_lookup.bed.gz",
        transformation=None,
        default_returnval=0.0)

    target_proportion_positives = 1 / 5

    coords_batch_producer = coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
        pos_bed_file=f"data/{TF}/{TF}_foreground_train.bed.gz",
        neg_bed_file=f"data/{TF}/{TF}_background_train.bed.gz",
        target_proportion_positives=target_proportion_positives,
        batch_size=100,
        shuffle_before_epoch=True,
        seed=seed)
    coordsbatch_transformer = coordbatchtransformers.ReverseComplementAugmenter() if is_aug else None

    train_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coords_batch_producer,
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals,
        coordsbatch_transformer=coordsbatch_transformer)
    return train_batch_generator


class AuRocCallback(keras.callbacks.Callback):
    def __init__(self, model, valid_X, valid_Y):
        self.model = model
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.best_auroc_sofar = 0.0
        self.best_weights = None
        self.last_weights = None
        self.best_epoch_number = 0

    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(self.valid_X)
        auroc = roc_auc_score(y_true=self.valid_Y, y_score=preds)
        self.last_weights = self.model.get_weights()
        if (auroc > self.best_auroc_sofar):
            self.best_weights = self.model.get_weights()
            self.best_epoch_number = epoch
            self.best_auroc_sofar = auroc


def train_model(model, curr_seed, train_data_loader, batch_generator,
                valid_data, epochs_to_train_for, upsampling):
    np.random.seed(curr_seed)
    tf.set_random_seed(curr_seed)

    if not upsampling:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=epochs_to_train_for,
            restore_best_weights=True)

        auroc_callback = AuRocCallback(model=model,
                                       valid_X=valid_data.X,
                                       valid_Y=valid_data.Y)
        history = History()
        loss_history = model.fit_generator(train_data_loader.get_batch_generator(),
                                           validation_data=(valid_data.X, valid_data.Y),
                                           epochs=epochs_to_train_for,
                                           steps_per_epoch=50,
                                           class_weight={0: 1, 1: 4.75},
                                           callbacks=[auroc_callback, early_stopping_callback, history])
        return early_stopping_callback, auroc_callback, history, model
    else:
        auroc_callback = AuRocCallback(model=model,
                                       valid_X=valid_data.X,
                                       valid_Y=valid_data.Y)
        history = History()
        loss_history = model.fit_generator(batch_generator,
                                           validation_data=(valid_data.X, valid_data.Y),
                                           epochs=epochs_to_train_for,
                                           steps_per_epoch=50,
                                           callbacks=[auroc_callback, history],
                                           workers=os.cpu_count(),
                                           use_multiprocessing=True
                                           )
        return auroc_callback, history, model


class AuRocNoCallback():
    def __init__(self, model, valid_X, valid_Y):
        self.model = model
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.best_auroc_sofar = 0.0
        self.best_weights = None
        self.best_epoch_number = 0

    def on_epoch_end(self, epoch):
        preds = self.model.predict(self.valid_X)
        auroc = roc_auc_score(y_true=self.valid_Y, y_score=preds)
        if auroc > self.best_auroc_sofar:
            self.best_weights = self.model.get_weights()
            self.best_epoch_number = epoch
            self.best_auroc_sofar = auroc
        return auroc


def eager_train_step(model, inputs, target,
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     loss_fn=tf.keras.losses.binary_crossentropy,
                     metrics="accuracy"):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss = loss_fn(target, pred)
        grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def eager_train(model,
                train_dataset,
                validation_object,
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss_fn=tf.keras.losses.MeanSquaredError(),
                metrics="accuracy",
                epochs_to_train_for=10):
    """
    Model : a keras model to be run on eager
    train_dataset : a tf.data
    validation_object : an instance of AuRocNoCallback
    """
    # total_batch = len(train_dataset)
    total_batch = 10
    for epoch in range(epochs_to_train_for):
        for batch_idx, (batch_in, batch_out) in enumerate(train_dataset):
            loss = eager_train_step(model,
                                    inputs=batch_in,
                                    target=batch_out,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    metrics=metrics)
            print(loss.numpy().item(), batch_idx, total_batch)
            break
        # auroc = validation_object.on_epoch_end(epoch=epoch)
        # print(auroc)


if __name__ == '__main__':
    def post_hoc_from_model(trained_model, seq_len=1000):
        binary_model_getlogits = keras.models.Model(inputs=trained_model.inputs,
                                                    outputs=trained_model.layers[-2].output)

        fwd_sequence_input = keras.layers.Input(shape=(seq_len, 4))
        rev_sequence_input = keras.layers.Lambda(function=lambda x: x[:, ::-1, ::-1])(fwd_sequence_input)
        fwd_logit_output = binary_model_getlogits(fwd_sequence_input)
        rev_logit_output = binary_model_getlogits(rev_sequence_input)
        average_logits = keras.layers.Average()([fwd_logit_output, rev_logit_output])
        sigmoid_out = keras.layers.Activation("sigmoid")(average_logits)

        siamese_model = keras.models.Model(inputs=[fwd_sequence_input],
                                           outputs=[sigmoid_out])
        return siamese_model


    def train_test_model(model, epochs_to_train_for=160, TF='CTCF', seed=1234, one_return=True, is_aug=False,
                         post_hoc=False):
        valid_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
            path_to_hdf5=f"data/{TF}/valid_data.hdf5", strip_enclosing_dictionary=True)
        valid_data = valid_data_loader.get_data()

        test_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
            path_to_hdf5=f"data/{TF}/test_data.hdf5", strip_enclosing_dictionary=True)
        test_data = test_data_loader.get_data()

        standard_train_batch_generator = get_generators(TF=TF,
                                                        seq_len=1000,
                                                        seed=seed,
                                                        is_aug=is_aug)
        if one_return:
            auroc_callback, history, trained_model = train_model(model=model,
                                                                 curr_seed=seed,
                                                                 train_data_loader=None,
                                                                 batch_generator=standard_train_batch_generator,
                                                                 valid_data=valid_data,
                                                                 epochs_to_train_for=epochs_to_train_for,
                                                                 upsampling=True)
            trained_model.set_weights(auroc_callback.best_weights)
            if post_hoc:
                trained_model = post_hoc_from_model(trained_model)

            a = roc_auc_score(y_true=valid_data.Y, y_score=trained_model.predict(valid_data.X))
            b = roc_auc_score(y_true=test_data.Y, y_score=trained_model.predict(test_data.X))
            print(a)
            print(b)
            print("Validation AUROC at best-auroc early stopping:", a)
            print("Test set AUROC at best-auroc early stopping:", b)

            return a, b

        auroc_callback = AuRocCallback(model=model,
                                       valid_X=valid_data.X,
                                       valid_Y=valid_data.Y)
        # Each epoch is 50 batches of 100 sequences ie 5000 seq.
        # CTCF has 37000 foreground sequences and 200742 so a total of 240k sequences.
        # Looping over all is about 48 epochs.
        step_per_epoch = 50
        epochs_to_try = [5, 10, 20, 40, 80, 160]
        epochs_results = {}
        history = History()
        last_epoch = 0
        for epoch in epochs_to_try:
            epochs_to_train_for = epoch - last_epoch
            last_epoch = epoch
            model.fit_generator(standard_train_batch_generator,
                                validation_data=(valid_data.X, valid_data.Y),
                                epochs=epochs_to_train_for,
                                steps_per_epoch=step_per_epoch,
                                callbacks=[auroc_callback, history],
                                workers=os.cpu_count(),
                                use_multiprocessing=True
                                )
            model.set_weights(auroc_callback.best_weights)
            if post_hoc:
                inference_model = post_hoc_from_model(model)
                a = roc_auc_score(y_true=valid_data.Y, y_score=inference_model.predict(valid_data.X))
                b = roc_auc_score(y_true=test_data.Y, y_score=inference_model.predict(test_data.X))
            else:
                a = roc_auc_score(y_true=valid_data.Y, y_score=model.predict(valid_data.X))
                b = roc_auc_score(y_true=test_data.Y, y_score=model.predict(test_data.X))
            model.set_weights(auroc_callback.last_weights)
            epochs_results[epoch] = (a, b)
        return epochs_results


    def test_model(model, logname, aggregatedname, model_name, seed_max=10, rc_aug=False, post_hoc=False):
        aggregated = collections.defaultdict(list)
        for seed in range(seed_max):
            dict_res = train_test_model(model, one_return=False, seed=seed, is_aug=rc_aug, post_hoc=post_hoc)
            with open(logname, 'a') as f:
                f.write(f'{model_name} with seed={seed}\n')
                for epoch, values in dict_res.items():
                    f.write(f'{epoch} {values[0]} {values[1]}\n')
                f.write(f'\n')

            for epoch, values in dict_res.items():
                aggregated[epoch].append(values)
        # Now value is a list of tuples of results, one for each seed.
        # Let us aggregate it into a mean and std for each
        with open(aggregatedname, 'a') as f:
            f.write(f'{model_name}\n')
            for epoch, values in aggregated.items():
                np_values = np.array(values)
                mean_value = np.mean(np_values)
                std_value = np.std(np_values)
                f.write(f'{epoch} {mean_value} {std_value}\n')
            f.write(f'\n')


    parameters = {
        'filters': 16,
        'input_length': 1000,
        'pool_size': 40,
        'strides': 20
    }

    logname = 'logfile_reproduce_posthoc.txt'
    with open(logname, 'w') as f:
        f.write('Log of the experiments results for reproducibility and prior inclusion :\n')

    aggname = 'outfile_reproduce_posthoc.txt'
    with open(aggname, 'w') as f:
        f.write('Experiments results for reproducibility and prior inclusion :\n')

    for tf in ['MAX', 'SPI1', 'CTCF']:
        # Make the classical models
        model = get_reg_model(parameters)
        model_name = f'rc_post_hoc with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name, rc_aug=True, post_hoc=True)

    """
    logname = 'logfile_reproduce_all.txt'
    with open(logname, 'w') as f:
        f.write('Log of the experiments results for reproducibility and prior inclusion :\n')

    aggname = 'outfile_reproduce_all.txt'
    with open(aggname, 'w') as f:
        f.write('Experiments results for reproducibility and prior inclusion :\n')

    for tf in ['MAX', 'SPI1', 'CTCF']:
        
        # Make the classical models
        model = get_reg_model(parameters)
        model_name = f'non equivariant with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)
        
        # Make the classical models with post_hoc
        model = get_reg_model(parameters)
        model_name = f'rc_post_hoc with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name, rc_aug=True, post_hoc=True)
        
        # Get the RCPS
        for k in range(1, 5):
            model = equinet.CustomRCPS(kmers=k)
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'RCPS with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            # Get the Equinet with different blends of a_n, b_n
            model = equinet.EquiNetBinary(kmers=k, filters=((16, 16), (16, 16), (16, 16)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 50a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            # Get the Equinet with different blends of a_n, b_n
            model = equinet.EquiNetBinary(kmers=k, filters=((32, 0), (32, 0), (32, 0)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 100a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = equinet.EquiNetBinary(kmers=k, filters=((24, 8), (24, 8), (24, 8)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 75a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = equinet.EquiNetBinary(kmers=k, filters=((8, 24), (8, 24), (8, 24)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 25a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = equinet.EquiNetBinary(kmers=k, filters=((0, 32), (0, 32), (0, 32)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 0a_n k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

    """

    """
    epochs_to_train_for = 160
    outname = 'outfile.txt'
    with open(outname, 'a') as f:
        f.write('Experiments results :\n')

    for k in range(1, 4):
        print(f'RCPS trained with K={k}')
        model = equinet.CustomRCPS(kmers=k)
        model = model.func_api_model()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        a, b, c, d = plot_values(model, epochs_to_train_for=epochs_to_train_for)
        with open(outname, 'a') as f:
            f.write(f'RCPS_{k} {a} {b} {c} {d}\n')

    for k in range(1, 4):
        print(f'Equinet trained with K={k}')
        model = equinet.EquiNetBinary(filters=[(16, 16), (16, 16), (16, 16)], kernel_sizes=[15, 14, 14], kmers=k)
        model = model.func_api_model()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        a, b, c, d = plot_values(model, epochs_to_train_for=epochs_to_train_for)
        with open(outname, 'a') as f:
            f.write(f'Equinet_{k} {a} {b} {c} {d}\n')
    """
