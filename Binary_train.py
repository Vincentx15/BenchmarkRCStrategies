import os
import numpy as np
import gzip
from collections import namedtuple
from sklearn.metrics import roc_auc_score

import tensorflow as tf

import keras
from keras import backend as K
from keras.layers.core import Dropout
from keras.layers.core import Flatten
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

import equinet

TF = 'CTCF'

valid_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
    path_to_hdf5=f"data/{TF}/valid_data.hdf5", strip_enclosing_dictionary=True)
valid_data = valid_data_loader.get_data()

test_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
    path_to_hdf5=f"data/{TF}/test_data.hdf5", strip_enclosing_dictionary=True)
test_data = test_data_loader.get_data()

Coordinates = namedtuple("Coordinates",
                         ["chrom", "start", "end", "isplusstrand"])
Coordinates.__new__.__defaults__ = (True,)


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


def get_generators(TF, seq_len, is_aug, curr_seed):
    inputs_coordstovals = coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="data/hg19.genome.fa",
        center_size_to_use=seq_len)

    targets_coordstovals = SimpleLookup(
        lookup_file=f"data/{TF}/{TF}_lookup.bed.gz",
        transformation=None,
        default_returnval=0.0)

    target_proportion_positives = 1 / 5

    if not is_aug:
        standard_train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
                pos_bed_file=f"data/{TF}/{TF}_foreground_train.bed.gz",
                neg_bed_file=f"data/{TF}/{TF}_background_train.bed.gz",
                target_proportion_positives=target_proportion_positives,
                batch_size=100,
                shuffle_before_epoch=True,
                seed=curr_seed),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals)
        return standard_train_batch_generator
    else:
        aug_train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
                pos_bed_file=f"{TF}_foreground_train.bed.gz",
                neg_bed_file=f"{TF}_background_train.bed.gz",
                target_proportion_positives=target_proportion_positives,
                batch_size=100,
                shuffle_before_epoch=True,
                seed=curr_seed),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.ReverseComplementAugmenter())
        return aug_train_batch_generator


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


def first_equi(parameters):
    model = keras.models.Sequential()
    model.add(equinet.RegToIrrepConv(input_shape=(1000, 4), reg_in=2, a_out=3, b_out=2, filter_length=15, use_bias=False))
    model.add(equinet.IrrepToIrrepConv(a_in=3, b_in=2, a_out=3, b_out=2, filter_length=15, use_bias=False))
    model.add(equinet.IrrepToIrrepConv(a_in=3, b_in=2, a_out=3, b_out=0, filter_length=15, use_bias=False))
    model.add(keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                      strides=parameters['strides']))
    model.add(Flatten())
    model.add(keras.layers.core.Dense(output_dim=1, trainable=True,
                                      init="glorot_uniform"))
    model.add(keras.layers.core.Activation("sigmoid"))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    return model


parameters = {
    'filters': 16,
    'input_length': 1000,
    'pool_size': 40,
    'strides': 20
}

# model = get_reg_model(parameters)
model = first_equi(parameters)

epochs_to_train_for = 160
standard_train_batch_generator = get_generators(TF=TF,
                                                seq_len=1000,
                                                curr_seed=1234,
                                                is_aug=False)

from keras.callbacks import History


class AuRocCallback(keras.callbacks.Callback):
    def __init__(self, model, valid_X, valid_Y):
        self.model = model
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.best_auroc_sofar = 0.0
        self.best_weights = None
        self.best_epoch_number = 0

    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(self.valid_X)
        auroc = roc_auc_score(y_true=self.valid_Y, y_score=preds)
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


auroc_callback, history, model = train_model(model=model,
                                             curr_seed=1234,
                                             train_data_loader=None,
                                             batch_generator=standard_train_batch_generator,
                                             valid_data=valid_data,
                                             epochs_to_train_for=epochs_to_train_for,
                                             upsampling=True)

model.set_weights(auroc_callback.best_weights)
print("Validation set AUROC with best-loss early stopping:",
      roc_auc_score(y_true=valid_data.Y, y_score=model.predict(valid_data.X)))
print("Test set AUROC with best-loss early stopping:",
      roc_auc_score(y_true=test_data.Y, y_score=model.predict(test_data.X)))
model.set_weights(auroc_callback.best_weights)
print("Validation AUROC at best-auroc early stopping:",
      roc_auc_score(y_true=valid_data.Y, y_score=model.predict(valid_data.X)))
print("Test set AUROC at best-auroc early stopping:",
      roc_auc_score(y_true=test_data.Y, y_score=model.predict(test_data.X)))