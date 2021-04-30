#!/usr/bin/env python
# coding: utf-8
import keras
import numpy as np
from keras.callbacks import History
from keras.models import load_model

from keras_genomics.layers.convolutional import RevCompConv1D

from BPNetArchs import RcBPNetArch
from RunBPNetArchs import get_generators, get_test_values

import collections
from equinet import *
from equinet import EquiNetBP

from keras.utils import CustomObjectScope
from keras.models import load_model
import keras.losses

# Global variables
keras.losses.MultichannelMultinomialNLL = MultichannelMultinomialNLL
equilayers = {'RegToRegConv': RegToRegConv,
              'RegToIrrepConv': RegToIrrepConv,
              'IrrepToRegConv': IrrepToRegConv,
              'IrrepToIrrepConv': IrrepToIrrepConv,
              'IrrepActivationLayer': IrrepActivationLayer,
              'RegBatchNorm': RegBatchNorm,
              'IrrepBatchNorm': IrrepBatchNorm,
              'IrrepConcatLayer': IrrepConcatLayer,
              'RegConcatLayer': RegConcatLayer,
              'loss': MultichannelMultinomialNLL,
              'MultichannelMultinomialNLL': MultichannelMultinomialNLL,
              'RevCompConv1D': RevCompConv1D,
              'ToKmerLayer': ToKmerLayer}


def train_model(train_generator, val_generator, epochs_to_train_for, model, save_name=None):
    early_stopping_callback = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)

    model_history = History()
    model.fit_generator(train_generator,
                        epochs=epochs_to_train_for,
                        validation_data=val_generator,
                        callbacks=[early_stopping_callback, model_history],
                        )
    model.set_weights(early_stopping_callback.best_weights)
    if save_name is not None:
        model.save(save_name)
    return model


def test_saved_model(model_name, test_generator, dataset, custom_objects=equilayers):
    model = load_model(model_name, custom_objects=custom_objects)

    jsd, pears, spear, mse = get_test_values(model, test_generator, dataset=dataset)
    print(f'{model_name} performance : ', jsd, pears, spear, mse)
    return jsd, pears, spear, mse


def train_test_model(model, dataset, seed, epochs=80, seq_len=1346, out_pred_len=1000, is_aug=False, model_name=None):
    train_generator, val_generator, test_generator, _ = get_generators(dataset=dataset,
                                                                       seed=seed,
                                                                       seq_len=seq_len,
                                                                       out_pred_len=out_pred_len,
                                                                       is_aug=is_aug)
    model = train_model(train_generator, val_generator, epochs, model, save_name=model_name)
    jsd, pears, spear, mse = get_test_values(model, test_generator, dataset=dataset)
    print(f'{model_name} performance : ', jsd, pears, spear, mse)
    return jsd, pears, spear, mse


def test_BPN_model(model, logname, aggregatedname, dataset, model_name=None, seed_max=10, is_aug=False):
    aggregated = list()
    for seed in range(seed_max):
        jsd, pears, spear, mse = train_test_model(model=model,
                                                  model_name=model_name,
                                                  dataset=dataset,
                                                  seed=seed,
                                                  is_aug=is_aug)
        aggregated.append((jsd, pears, spear, mse))
        with open(logname, 'a') as f:
            f.write(f'{model_name} with seed={seed}\n')
            f.write(f'{jsd} {pears} {spear} {mse}\n')
            f.write(f'\n')

    # Now value is a list of tuples of results, one for each seed.
    # Let us aggregate it into a mean and std for each
    with open(aggregatedname, 'a') as f:
        f.write(f'{model_name}\n')
        aggregated = np.array(aggregated)
        jsd, pears, spear, mse = np.mean(aggregated, axis=0)
        f.write(f'{jsd} {pears} {spear} {mse}\n')
        f.write(f'\n')


if __name__ == '__main__':
    pass

    first_seed = 0
    np.random.seed(first_seed)
    tf.set_random_seed(first_seed)

    logname = 'logfile_bpn_posthoc.txt'
    with open(logname, 'w') as f:
        f.write('Log of the experiments on BPN :\n')

    aggname = 'outfile_bpn_posthoc.txt'
    with open(aggname, 'w') as f:
        f.write('Experiments results for BPN :\n')

    for dataset in ['KLF4', 'NANOG', 'SOX2', 'OCT4']:
        model_name = f'rc_post_hoc with dataset={dataset}'
        equinet_model = EquiNetBP(dataset=dataset).get_keras_model()
        # jsd, pears, spear, mse = train_test_model(equinet_model, model_name='equinet_oct4', dataset=dataset,
        # seed=seed, is_aug=False)
        test_BPN_model(model=equinet_model, model_name=model_name, dataset=dataset,
                       logname=logname, aggregatedname=aggname)
