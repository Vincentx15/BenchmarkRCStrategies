#!/usr/bin/env python
# coding: utf-8
import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine.base_layer import InputSpec
from keras.callbacks import History
import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import tensorflow as tf
import numpy as np
import seqdataloader
import keras.layers as kl
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from BPNetArchs import RcBPNetArch
from BPNetArchs import SiameseBPNetArch
from BPNetArchs import StandardBPNetArch

import RunBPNetArchs
from RunBPNetArchs import get_inputs_and_targets
from RunBPNetArchs import get_specific_generator
from RunBPNetArchs import save_all

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
    'size': 100,
    'kernel_initializer': "glorot_uniform",
    'seed': 1234
}

seq_len = 1346
out_pred_len = 1000
curr_seed = PARAMETERS['seed']

inputs_coordstovals, targets_coordstovals = get_inputs_and_targets(dataset=PARAMETERS['dataset'],
                                                                   seq_len=seq_len,
                                                                   out_pred_len=out_pred_len)
early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)


def run_model(model, model_arch, PARAMETERS):
    train_batch_generator, val_batch_generator = get_specific_generator(PARAMETERS,
                                                                        inputs_coordstovals,
                                                                        targets_coordstovals,
                                                                        model_arch,
                                                                        curr_seed)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    model_history = History()
    model.fit_generator(train_batch_generator,
                        epochs=200,
                        validation_data=val_batch_generator,
                        callbacks=[early_stopping_callback, model_history])
    model.set_weights(early_stopping_callback.best_weights)
    save_all(PARAMETERS, model, model_arch, model_history)


np.random.seed(curr_seed)
tf.set_random_seed(curr_seed)

rc_model = RcBPNetArch(is_add=True, **PARAMETERS).get_keras_model()
run_model(model=rc_model, model_arch="RevComp_half", PARAMETERS=PARAMETERS)
