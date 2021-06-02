#!/usr/bin/env python
# coding: utf-8
import keras
import numpy as np
from keras.callbacks import History

from keras_genomics.layers.convolutional import RevCompConv1D

from BPNetArchs import RcBPNetArch, EquiNetBP, StandardBPNetArch, MultichannelMultinomialNLL, RegularBPN
from RunBPNetArchs import get_generators, get_test_values

from equinet import *
import collections

from keras.models import load_model
import keras.losses
import gzip
import random

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


def train_model(model, train_generator, val_generator, epochs, save_name=None):
    early_stopping_callback = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)
    model_history = History()
    model.fit_generator(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=[early_stopping_callback, model_history])
    model.set_weights(early_stopping_callback.best_weights)
    if save_name is not None:
        model.save(save_name)
    return model


def test_saved_model(model_name, test_generator, dataset, custom_objects=equilayers, post_hoc=False):
    model = load_model(model_name, custom_objects=custom_objects)

    jsd, pears, spear, mse = get_test_values(model, test_generator, dataset=dataset, post_hoc=post_hoc)
    print(f'{model_name} performance : ', jsd, pears, spear, mse)
    return jsd, pears, spear, mse


def train_test_model(model, dataset, seed, epochs=80, seq_len=1346, out_pred_len=1000, is_aug=False, model_name=None,
                     post_hoc=False, one_return=True, reduced=False):
    train_generator, val_generator, test_generator, _ = get_generators(dataset=dataset,
                                                                       seed=seed,
                                                                       seq_len=seq_len,
                                                                       out_pred_len=out_pred_len,
                                                                       is_aug=is_aug,
                                                                       reduced=reduced)
    if one_return:
        model = train_model(model, train_generator, val_generator, epochs, save_name=model_name)
        jsd, pears, spear, mse = get_test_values(model, test_generator, dataset=dataset, post_hoc=post_hoc)
        print(f'Performance : ', jsd, pears, spear, mse)
        return jsd, pears, spear, mse

    # step_per_epoch = 50
    # epochs_to_try = [5, 10, 20, 40, 80, 160]
    # epochs_results = {}
    # history = History()
    # last_epoch = 0
    # for epoch in epochs_to_try:
    #
    #     model.fit_generator(train_generator,
    #                         epochs=epochs,
    #                         validation_data=val_generator,
    #                         callbacks=[early_stopping_callback, history],
    #                         )
    #
    #     input()
    #     print(model.history)
    #     len(model.history['loss'])
    #     input()
    #     self.best_weights = self.model.get_weights()
    #
    #
    #     epochs_to_train_for = epoch - last_epoch
    #     last_epoch = epoch
    #     model.fit_generator(standard_train_batch_generator,
    #                         validation_data=(valid_data.X, valid_data.Y),
    #                         epochs=epochs_to_train_for,
    #                         steps_per_epoch=step_per_epoch,
    #                         callbacks=[auroc_callback, history],
    #                         workers=os.cpu_count(),
    #                         use_multiprocessing=True
    #                         )
    #     model.set_weights(auroc_callback.best_weights)
    #     if post_hoc:
    #         inference_model = post_hoc_from_model(model)
    #         a = roc_auc_score(y_true=valid_data.Y, y_score=inference_model.predict(valid_data.X))
    #         b = roc_auc_score(y_true=test_data.Y, y_score=inference_model.predict(test_data.X))
    #     else:
    #         a = roc_auc_score(y_true=valid_data.Y, y_score=model.predict(valid_data.X))
    #         b = roc_auc_score(y_true=test_data.Y, y_score=model.predict(test_data.X))
    #     model.set_weights(auroc_callback.last_weights)
    #     epochs_results[epoch] = (a, b)
    #
    # model.set_weights(early_stopping_callback.best_weights)
    # if model_name is not None:
    #     model.save(model_name)
    # return model
    # return epochs_results


def test_BPN_model(model, logname, dataset, model_name=None, epochs=80, seed_max=6, is_aug=False,
                   post_hoc=False, reduced=False):
    aggregated = list()
    for seed in range(seed_max):
        jsd, pears, spear, mse = train_test_model(model=model,
                                                  model_name=model_name,
                                                  dataset=dataset,
                                                  epochs=epochs,
                                                  seed=seed,
                                                  is_aug=is_aug,
                                                  post_hoc=post_hoc,
                                                  reduced=reduced)
        aggregated.append((jsd, pears, spear, mse))
        with open(logname, 'a') as f:
            f.write(f'{model_name} with seed={seed}\n')
            f.write(f'{jsd} {pears} {spear} {mse}\n')
            f.write(f'\n')


def test_all_models(logname):
    with open(logname, 'w') as f:
        f.write('Log of the experiments on BPN :\n')

    for dataset in ['KLF4', 'NANOG', 'SOX2', 'OCT4']:
        pass
        # Get the non equivariant model
        model_name = f'non equivariant with dataset={dataset}'
        standard_model = StandardBPNetArch(dataset=dataset).get_keras_model()
        test_BPN_model(model=standard_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        # Compare to the RCPS, original and custom
        model_name = f'RCPS with dataset={dataset}'
        equinet_model = RcBPNetArch(dataset=dataset).get_keras_model()
        test_BPN_model(model=equinet_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        model_name = f'RCPS_custom with dataset={dataset}'
        equinet_model = RegularBPN(dataset=dataset).get_keras_model()
        test_BPN_model(model=equinet_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        # Compare to equinets 75 with k=1 and k=2
        model_name = f'equi_75_k1 with dataset={dataset}'
        equi_model = EquiNetBP(dataset=dataset, kmers=1, filters=(
            (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32))).get_keras_model()
        test_BPN_model(model=equi_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        model_name = f'best_equi with dataset={dataset}'
        equi_model = EquiNetBP(dataset=dataset, kmers=2, filters=(
            (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32))).get_keras_model()
        test_BPN_model(model=equi_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        # Compare to regular 2
        model_name = f'RCPS_2 with dataset={dataset}'
        rcps2_model = RcBPNetArch(dataset=dataset, kmers=2).get_keras_model()
        test_BPN_model(model=rcps2_model, model_name=model_name, dataset=dataset,
                       logname=logname)

        # Try assessing the impact of data augmentation on equivariant models
        model_name = f'best_equi_aug with dataset={dataset}'
        equi_model = EquiNetBP(dataset=dataset, kmers=2, filters=(
            (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32))).get_keras_model()
        test_BPN_model(model=equi_model, model_name=model_name, dataset=dataset,
                       logname=logname, is_aug=True)

        # Look at the post-hoc performance
        model_name = f'rc_post_hoc with dataset={dataset}'
        rc_model = StandardBPNetArch(dataset=dataset).get_keras_model()
        test_BPN_model(model=rc_model, model_name=model_name, dataset=dataset,
                       logname=logname, is_aug=True, post_hoc=True)


def test_all_models_reduced(logname):
    with open(logname, 'w') as f:
        f.write('Log of the experiments on BPN :\n')

    for dataset in ['KLF4', 'NANOG', 'SOX2', 'OCT4']:
        pass
        # Get the non equivariant model
        model_name = f'reduced_non_equivariant with dataset={dataset}'
        standard_model = StandardBPNetArch(dataset=dataset).get_keras_model()
        test_BPN_model(model=standard_model, model_name=model_name, dataset=dataset,
                       logname=logname, reduced=True)

        # Compare to the custom RCPS
        model_name = f'reduced_RCPS_custom with dataset={dataset}'
        equinet_model = RegularBPN(dataset=dataset).get_keras_model()
        test_BPN_model(model=equinet_model, model_name=model_name, dataset=dataset,
                       logname=logname, reduced=True)

        # Compare to equinets 75 with k=2
        model_name = f'reduced_best_equi with dataset={dataset}'
        equi_model = EquiNetBP(dataset=dataset, kmers=2, filters=(
            (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32), (96, 32))).get_keras_model()
        test_BPN_model(model=equi_model, model_name=model_name, dataset=dataset,
                       logname=logname, reduced=True)

        # Compare to regular 2
        model_name = f'reduced_RCPS_2 with dataset={dataset}'
        rcps2_model = RcBPNetArch(dataset=dataset, kmers=2).get_keras_model()
        test_BPN_model(model=rcps2_model, model_name=model_name, dataset=dataset,
                       logname=logname, reduced=True)


if __name__ == '__main__':
    pass

    first_seed = 0
    np.random.seed(first_seed)
    tf.set_random_seed(first_seed)

    test_all_models(logname='logfile_bpn.txt')
    test_all_models_reduced(logname='logfile_bpn_reduced.txt')
