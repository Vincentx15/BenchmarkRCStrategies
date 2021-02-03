#!/usr/bin/env python
# coding: utf-8
import os

from SimArchs import RCArch
from SimArchs import RegArch
from SimArchs import SiameseArch
from RunSimArchs import train_model
from RunSimArchs import prepare_sequences
from RunSimArchs import save_all

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PARAMETERS = {
    'filters': 20,
    'kernel_size': 21,
    'input_length': 1000,
    'pool_size': 20,
    'strides': 20
}

RC_WRAPPER = RCArch(**PARAMETERS)
REG_WRAPPER = RegArch(**PARAMETERS)
SIAMESE_WRAPPER = SiameseArch(**PARAMETERS)

# get_ipython().system('densityMotifSimulation.py --seed 1234 --motifNames ELF1_known2 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200  --rc-prob 0.5 --numSeqs 10000')
# get_ipython().system('densityMotifSimulation.py --seed 1234 --motifNames GATA_known6 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --rc-prob 0.5 --numSeqs 10000')
# get_ipython().system('densityMotifSimulation.py --seed 1234 --motifNames RXRA_known1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --rc-prob 0.5 --numSeqs 10000')

seq_len = 1000
curr_seed = 1234
x_train, x_val, x_test, y_train, y_train_mutate, y_val, y_test = prepare_sequences(seq_len=seq_len)
model = RC_WRAPPER.get_model()

early_stopping_callback, auroc_callback = train_model(model=model,
                                                      aug=None,
                                                      curr_seed=curr_seed,
                                                      batch_size=500,
                                                      x=x_train,
                                                      y=y_train,
                                                      val_data=(x_val, y_val))
save_all(filepath="path/to/file",
         model_arch="model_name",
         curr_seed=curr_seed,
         callback=auroc_callback,
         model=model,
         val_data=(x_val, y_val))
