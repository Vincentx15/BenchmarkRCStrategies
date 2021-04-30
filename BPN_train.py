#!/usr/bin/env python
# coding: utf-8
import keras
from keras.callbacks import History
from keras.models import load_model

from keras_genomics.layers.convolutional import RevCompConv1D

from BPNetArchs import RcBPNetArch
from RunBPNetArchs import get_generators, get_test_values

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


if __name__ == '__main__':
    dataset = 'OCT4'
    MODEL_NAME = 'equinet_oct4'

    seq_len = 1346
    out_pred_len = 1000
    epochs_to_train_for = 80
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    rcps_model = RcBPNetArch(dataset=dataset).get_keras_model()
    equinet_model = EquiNetBP(dataset=dataset).get_keras_model()
    jsd, pears, spear, mse = train_test_model(equinet_model, model_name='equinet_oct4', dataset=dataset, seed=seed,
                                              is_aug=False)
