import collections
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import tensorflow as tf

import keras
from keras.callbacks import History

import momma_dragonn
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers

from BinaryArchs import get_reg_model, EquiNetBinary, CustomRCPS
from RunBinaryArchs import SimpleLookup, KerasBatchGenerator, AuRocCallback
import gzip
import random


def get_reduced_bed(infile, outfile, size=1000):
    with gzip.open(infile) as f:
        lines = f.readlines()
    selected_indices = sorted(random.sample(list(range(len(lines))), size))
    selected_lines = [lines[selected] for selected in selected_indices]

    with gzip.open(outfile, 'wb') as f:
        for line in selected_lines:
            f.write(line)


def get_generator(TF, seq_len, is_aug, seed, reduced=True):
    inputs_coordstovals = coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="data/hg19.genome.fa",
        center_size_to_use=seq_len)

    targets_coordstovals = SimpleLookup(
        lookup_file=f"data/{TF}/{TF}_lookup.bed.gz",
        transformation=None,
        default_returnval=0.0)

    target_proportion_positives = 1 / 5

    pos_bed = f"data/{TF}/{TF}_foreground_train.bed.gz"
    neg_bed = f"data/{TF}/{TF}_background_train.bed.gz"

    if reduced:
        pos_bed_reduced = f"data/{TF}/{TF}_reduced_foreground_train.bed.gz"
        if not os.path.exists(pos_bed_reduced) or True:
            get_reduced_bed(infile=pos_bed, outfile=pos_bed_reduced)
        pos_bed = pos_bed_reduced

        # neg_bed_reduced = f"data/{TF}/{TF}_reduced_background_train.bed.gz"
        # get_reduced_bed(infile=neg_bed, outfile=neg_bed_reduced)
        # neg_bed = neg_bed_reduced


    coords_batch_producer = coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
        pos_bed_file=pos_bed,
        neg_bed_file=neg_bed,
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


def train_model(model, train_generator,
                valid_data, epochs_to_train_for, upsampling, steps_per_epoch=50):
    auroc_callback = AuRocCallback(model=model,
                                   valid_X=valid_data.X,
                                   valid_Y=valid_data.Y)
    history = History()

    if not upsampling:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=epochs_to_train_for,
            restore_best_weights=True)

        loss_history = model.fit_generator(train_generator,
                                           validation_data=(valid_data.X, valid_data.Y),
                                           epochs=epochs_to_train_for,
                                           steps_per_epoch=50,
                                           class_weight={0: 1, 1: 4.75},
                                           callbacks=[auroc_callback, early_stopping_callback, history])
        return early_stopping_callback, auroc_callback, history, model
    else:
        loss_history = model.fit_generator(train_generator,
                                           validation_data=(valid_data.X, valid_data.Y),
                                           epochs=epochs_to_train_for,
                                           steps_per_epoch=steps_per_epoch,
                                           callbacks=[auroc_callback, history],
                                           workers=os.cpu_count(),
                                           use_multiprocessing=True
                                           )
        return auroc_callback, history, model


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
                     post_hoc=False, reduced=False):
    valid_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
        path_to_hdf5=f"data/{TF}/valid_data.hdf5", strip_enclosing_dictionary=True)
    valid_data = valid_data_loader.get_data()

    if reduced:
        random_indices = np.random.choice(len(valid_data.X), size=2000, replace=False)
        extracted_X = valid_data.X[random_indices]
        extracted_Y = valid_data.Y[random_indices]
        valid_data.X = extracted_X
        valid_data.Y = extracted_Y

    test_data_loader = momma_dragonn.data_loaders.hdf5_data_loader.MultimodalAtOnceDataLoader(
        path_to_hdf5=f"data/{TF}/test_data.hdf5", strip_enclosing_dictionary=True)
    test_data = test_data_loader.get_data()

    standard_train_batch_generator = get_generator(TF=TF,
                                                   seq_len=1000,
                                                   seed=seed,
                                                   is_aug=is_aug,
                                                   reduced=reduced)
    steps_per_epoch = 50 if not reduced else None

    auroc_callback = AuRocCallback(model=model,
                                   valid_X=valid_data.X,
                                   valid_Y=valid_data.Y)
    history = History()
    if one_return:
        model.fit_generator(standard_train_batch_generator,
                            validation_data=(valid_data.X, valid_data.Y),
                            epochs=epochs_to_train_for,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[auroc_callback, history],
                            workers=os.cpu_count(),
                            use_multiprocessing=True
                            )
        model.set_weights(auroc_callback.best_weights)
        if post_hoc:
            trained_model = post_hoc_from_model(model)

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
                            steps_per_epoch=steps_per_epoch,
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


def test_model(model, logname, aggregatedname, model_name, seed_max=10, rc_aug=False, post_hoc=False, reduced=False):
    aggregated = collections.defaultdict(list)
    for seed in range(seed_max):
        dict_res = train_test_model(model, one_return=False, seed=seed, is_aug=rc_aug, post_hoc=post_hoc,
                                    reduced=reduced)
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


def test_all(logname='logfile_reproduce_all.txt', aggname='outfile_reproduce_all.txt'):
    with open(logname, 'a') as f:
        f.write('Log of the experiments results for reproducibility and prior inclusion :\n')
    with open(aggname, 'a') as f:
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

        for k in range(1, 5):
            # Get the RCPS
            model = CustomRCPS(kmers=k)
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'RCPS with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            # Get the Equinet with different blends of a_n, b_n
            model = EquiNetBinary(kmers=k, filters=((16, 16), (16, 16), (16, 16)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 50a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            # Get the Equinet with different blends of a_n, b_n
            model = EquiNetBinary(kmers=k, filters=((32, 0), (32, 0), (32, 0)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 100a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = EquiNetBinary(kmers=k, filters=((24, 8), (24, 8), (24, 8)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 75a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = EquiNetBinary(kmers=k, filters=((8, 24), (8, 24), (8, 24)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 25a_n with k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)

            model = EquiNetBinary(kmers=k, filters=((0, 32), (0, 32), (0, 32)))
            model = model.func_api_model()
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
            model_name = f'Equinet 0a_n k={k} with tf={tf}'
            test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name)


def test_reduced(logname='logfile_reproduce_reduced.txt', aggname='outfile_reproduce_reduced.txt'):
    with open(logname, 'a') as f:
        f.write('Log of the experiments results for reproducibility and prior inclusion :\n')
    with open(aggname, 'a') as f:
        f.write('Experiments results for reproducibility and prior inclusion :\n')

    for tf in ['MAX', 'SPI1', 'CTCF']:
        # Make the classical models
        model = get_reg_model(parameters)
        model_name = f'reduced non equivariant with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name, reduced=True)

        # Make the classical models with post_hoc
        model = get_reg_model(parameters)
        model_name = f'reduced_post_hoc with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name, rc_aug=True, post_hoc=True,
                   reduced=True)

        model = EquiNetBinary(kmers=2, filters=((24, 8), (24, 8), (24, 8)))
        model = model.func_api_model()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        model_name = f'reduced_equinet_2_75 with tf={tf}'
        test_model(model, logname=logname, aggregatedname=aggname, model_name=model_name, reduced=True)


if __name__ == '__main__':
    parameters = {
        'filters': 16,
        'input_length': 1000,
        'pool_size': 40,
        'strides': 20
    }

    # test_all()
    test_reduced()
