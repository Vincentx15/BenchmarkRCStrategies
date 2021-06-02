# Reverse-Complement Equivariant Networks for DNA Sequences

## General structure

This code contains the steps necessary to reproduce the results of the submission.
It is structured as follows :
* equinet.py defines the new equivariant layers 
* Binary(resp BPN)Archs defines the networks architectures used in the Binary (resp Profile) task
* RunBinary(resp BPN) defines utils necessary to run the Binary (resp Profile) task
* Binary(resp BPN)_train contains the actual scripts that were launched with the different hyperparameters for the 
  Binary (resp Profile) task as well as the seeding mechnanism
* plot.py contains the plotting scripts. 

## Important Notes 
* Keras 2.2.4 was used to train all models. Keras 2.3 has a bug where the validation set loss is not computed correctly. More information here: https://github.com/keras-team/keras/issues/13389
* Typical BPNet architectures have both a profile prediction head and a total-counts prediction head, but here we benchmarked on only the profile prediction head (equivalent to setting the weight for the total-counts prediction head to zero). 


## Reproducing the results
This project has a few dependencies, but mostly relies on Keras 2.2.4 with tf 1.14 backend.
A .yml file is included for the cpu setting. Other packages include standard genomic ones
as well as some developed by Anshul Kundaje's lab.
To use the .yml, just run :
```
conda create -f rcps_cpu.yml
```

Once the environment was setup.
To produce the raw data, one just need to run :
```
python Binary_train.py
python BPN_train.py
```

This will produce four logfiles : one for each task and in each of the reduced/full_data setting.

Then these files can be used for plotting. 
They can be concatenated by hand in one file for easier plotting.
We chose to concatenate ours to get just one for the binary and one for the profile.
We named them *'archives_results/logfile_binary_all.txt'* and *'archives_results/logfile_bpn_all.txt'*
Then the user can simply run : 
```
python plot.py
```

This will produce all figures from the paper.

## Notes on the data Format and loading

### Binary task
####  Binary task data 
The inputs are sequences and their annotations. 
These sequences are stored as a fasta file of the whole genome (hg19.genome.fa)
and bedfiles to extract the relevant parts.
A bedfile looks like a file whose lines follow this scheme :


chromosome [tab] start [tab] end [tab] info


A first step consists in downloading these bed files and getting 
{TF}_{background/foreground}.bed.gz files. Then these files can be split into
training and testing splits based on their chromosomes, using bash scripting.
Then another 'lookup' bedfile can be created, from the positive/negative splits
that replace the 'info' with a 1 for positive, and a zero for negative.
The negative one is not needed as we can rely on negative as a default
value.
However, these files are also directly available for downloading.

Actually, these files are only used for the train split, as for the other splits,
the data fits in memory, and the dataset is
therefore expected to be provided in the form of an .hdf5 file, also
available for downloading.

####  Binary task data processing

These files are then blended together by the get_generator function.
It leverages two utilities : PyfaidxCoordsToVals and SimpleLookup respectively
for the input and output.
These utilities take coordinates as inputs and 
return the corresponding sequence (resp. annotation).

It is then fed to a DownsampleNegativesCoordsBatchProducer. 
This class iterates over positive and negative bed files to produce
coordinates and then feed it to the aforementioned utilities to yield the
successive sequences and annotations.
There are two ways to balance the datasets. In the upsampling scheme, 
we fix the proportion of positive labels that we want. 
In the upweighting scheme, we need to additionally provide the class weights.

###  BPNet task data processing

The data and data processing here is of the same kind as in the previous section.
However, there is no hdf5 files in the production of the validation or test splits,
all splits are produced similarly. 
Instead of using a SimpleLookup object for the target, the format used is the bigwig format.

The rest of the processing is the same.


