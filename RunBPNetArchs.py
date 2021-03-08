from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased.coordbatchproducers import SimpleCoordsBatchProducer
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import AbstractCountAndProfileTransformer
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import LogCountsPlusOne
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import SmoothProfiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import BigWigReader
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import smooth_profiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import rolling_window
from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import AbstractCoordBatchTransformer
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
import keras.layers as kl
from keras.engine import Layer
from keras.engine.base_layer import InputSpec
from keras.callbacks import History

import os


def get_inputs_and_targets(dataset, seq_len, out_pred_len):
    inputs_coordstovals = coordstovals.core.CoordsToValsJoiner(
        coordstovals_list=[
            coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
                genome_fasta_path="data/mm10_no_alt_analysis_set_ENCODE.fasta",
                mode_name="sequence",
                center_size_to_use=seq_len),
            coordstovals.bigwig.PosAndNegSeparateLogCounts(
                counts_mode_name="patchcap.logcount",
                profile_mode_name="patchcap.profile",
                pos_strand_bigwig_path="data/patchcap/counts.pos.bw",
                neg_strand_bigwig_path="data/patchcap/counts.neg.bw",
                center_size_to_use=out_pred_len),
        ]
    )

    targets_coordstovals = coordstovals.core.CoordsToValsJoiner(
        coordstovals_list=[
            coordstovals.bigwig.PosAndNegSeparateLogCounts(
                counts_mode_name="CHIPNexus.%s.logcount" % dataset,
                profile_mode_name="CHIPNexus.%s.profile" % dataset,
                pos_strand_bigwig_path="data/%s/counts.pos.bw" % dataset,
                neg_strand_bigwig_path="data/%s/counts.neg.bw" % dataset,
                center_size_to_use=out_pred_len)
        ]
    )
    return inputs_coordstovals, targets_coordstovals


class GeneralReverseComplement(AbstractCoordBatchTransformer):
    def __call__(self, coords):
        return [get_revcomp(x) for x in coords]


class RevcompTackedOnSimpleCoordsBatchProducer(SimpleCoordsBatchProducer):
    def _get_coordslist(self):
        return [x for x in self.bed_file.coords_list] + [get_revcomp(x) for x in self.bed_file.coords_list]


def get_train_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals, model_arch):
    dataset = PARAMETERS['dataset']
    train_file = os.path.join('data', dataset, f"bpnet_{dataset}_train_1k_around_summits.bed.gz")
    chromsizes_file = os.path.join('data', "mm10.chrom.sizes")

    if model_arch == "Standard-RCAug":
        train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file=train_file,
                batch_size=64,
                shuffle_before_epoch=True,
                seed=PARAMETERS['seed']),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.ReverseComplementAugmenter().chain(
                coordbatchtransformers.UniformJitter(
                    maxshift=200, chromsizes_file=chromsizes_file)))
    else:
        train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file=train_file,
                batch_size=64,
                shuffle_before_epoch=True,
                seed=PARAMETERS['seed']),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                maxshift=200, chromsizes_file=chromsizes_file))

    return train_batch_generator


def get_val_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals):
    dataset = PARAMETERS['dataset']
    valid_file = os.path.join('data', dataset, f"bpnet_{dataset}_valid_1k_around_summits.bed.gz")
    chromsizes_file = os.path.join('data', "mm10.chrom.sizes")
    val_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=valid_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=PARAMETERS['seed']),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals)

    return val_batch_generator


def get_test_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals):
    dataset=PARAMETERS['dataset']
    train_file = os.path.join('data',dataset, f"bpnet_{dataset}_test_1k_around_summits.bed.gz")
    chromsizes_file = os.path.join('data',"mm10.chrom.sizes")

    keras_test_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=test_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=PARAMETERS['seed']),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals)

    keras_rc_test_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=test_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=PARAMETERS['seed']),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals,
        coordsbatch_transformer=GeneralReverseComplement())

    return keras_test_batch_generator, keras_rc_test_batch_generator


def save_results(PARAMETERS, model, model_arch, model_history):
    txt_file_name = ("%s.txt" % (model_arch))
    loss_file = open(txt_file_name, "w")
    loss_file.write("model parameters" + "\n")
    for x in PARAMETERS:
        loss_file.write(str(x) + ": " + str(PARAMETERS[x]) + "\n")

    loss_file.write("val_loss\n")
    for row in model_history.history["val_loss"]:
        loss_file.write(str(row) + "\n")
    loss_file.write("min val loss: " + str(np.min(model_history.history["val_loss"])))

    loss_file.close()
    if PARAMETERS['filters'] == 32:
        model_save_name = ("%s-half.h5" % (model_arch))
    else:
        model_save_name = ("%s.h5" % (model_arch))

    model.save(model_save_name)


def train_model(PARAMETERS, inputs_coordstovals, targets_coordstovals, epochs_to_train_for, model, model_arch):
    train_batch_generator = get_train_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals, model_arch)
    val_batch_generator = get_val_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)

    model_history = History()
    model.fit_generator(train_batch_generator,
                        epochs=epochs_to_train_for,
                        validation_data=val_batch_generator,
                        callbacks=[early_stopping_callback, model_history])
    model.set_weights(early_stopping_callback.best_weights)
    save_results(PARAMETERS, model, model_arch, model_history)


PARAMETERS = {
    'dataset': 'SOX2',
    'input_seq_len': 1346,
    'c_task_weight': 0,
    'p_task_weight': 1,
    'filters': 64,
    'n_dil_layers': 6,
    'conv1_kernel_size': 21,
    'dil_kernel_size': 3,
    'outconv_kernel_size': 75,
    'optimizer': 'Adam',
    'weight_decay': 0.01,
    'lr': 0.001,
    'kernel_initializer': "glorot_uniform",
    'seed': 42
}
out_pred_len = 1000
epochs_to_train_for = 200

seed = 42
np.random.seed(seed)

tf.set_random_seed(seed)

inputs_coordstovals, targets_coordstovals = get_inputs_and_targets(dataset=PARAMETERS['dataset'],
                                                                   seq_len=PARAMETERS['input_seq_len'],
                                                                   out_pred_len=out_pred_len)

batch_generator, keras_rc_test_batch_generator = get_test_generator(PARAMETERS=PARAMETERS,
                                                                    inputs_coordstovals=inputs_coordstovals,
                                                                    targets_coordstovals=targets_coordstovals)

from BPNetArchs import RcBPNetArch
from equinet import EquiNetBP

rcps_model = RcBPNetArch(is_add=True, **PARAMETERS).get_keras_model()
equinet_model = EquiNetBP(dataset=PARAMETERS['dataset'])
