# Benchmarking Reverse-Complement Strategies for Deep Learning Models in Genomics 

Link to BiorXiv preprint: https://www.biorxiv.org/content/10.1101/2020.11.04.368803v1

## Important Notes 
* Keras 2.2.4 was used to train all models. Keras 2.3 has a bug where the validation set loss is not computed correctly. More information here: https://github.com/keras-team/keras/issues/13389
* Typical BPNet architectures have both a profile prediction head and a total-counts prediction head, but here we benchmarked on only the profile prediction head (equivalent to setting the weight for the total-counts prediction head to zero). 

## Data Format and loading
### Simulated task

The data is produced using the simdata utility which produces simulated input sequences
as well as embedding files that denote the presence of known motifs.
The sequence files are one-hot encoded.
The embedding files are thus screened for the simulated motifs and one-hot encoded.
These files are then split in train/validation/test splits.

The loading is then quite easy as it amounts to loading numpy tables.

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
the data fits in memory (I guess that is the reason), and the dataset is
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

## Code snippets for converting trained models to post-training conjoined models
### Simulated and Binary Prediction Models 

```python
model = load_model('model_name')
binary_model_getlogits = keras.models.Model(inputs=model.inputs,
                                            outputs=model.layers[-2].output)

fwd_sequence_input = keras.layers.Input(shape=('seq_len', 4))
rev_sequence_input = keras.layers.Lambda(function=lambda x: x[:,::-1,::-1])(fwd_sequence_input)
fwd_logit_output = binary_model_getlogits(fwd_sequence_input)
rev_logit_output = binary_model_getlogits(rev_sequence_input)
average_logits = keras.layers.Average()([fwd_logit_output, rev_logit_output])
sigmoid_out = keras.layers.Activation("sigmoid")(average_logits)

siamese_model = keras.models.Model(inputs=[fwd_sequence_input],
                                           outputs=[sigmoid_out])
```

### Profile Prediction Models

```python
loaded_model = load_model('model_name)

#Let's create the model
#Define the inputs
fwd_sequence_input = keras.models.Input(shape=(1346,4))
fwd_patchcap_logcount = keras.models.Input(shape=(2,))
fwd_patchcap_profile = keras.models.Input(shape=(1000,2))

#RevComp input
rev_sequence_input = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(fwd_sequence_input)
rev_patchcap_logcount = keras.layers.Lambda(lambda x: x[:,::-1])(fwd_patchcap_logcount)
#note that last axis is NOT fwd vs reverse strand, but different smoothing levels
#that's why we only flip the middle axis
rev_patchcap_profile = keras.layers.Lambda(lambda x: x[:,::-1])(fwd_patchcap_profile)

#Run the model on the original fwd inputs
fwd_logcount, fwd_profile = loaded_model(
    [fwd_sequence_input, fwd_patchcap_logcount, fwd_patchcap_profile])

#Run the original model on the reverse inputs
rev_logcount, rev_profile = loaded_model(
    [rev_sequence_input, rev_patchcap_logcount, rev_patchcap_profile])

#Reverse complement rev_logcount and rev_profile to be compatible with fwd
revcompd_rev_logcount = keras.layers.Lambda(lambda x: x[:,::-1])(rev_logcount)
revcompd_rev_profile = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(rev_profile)

#Average the two
avg_logcount = keras.layers.Average()([fwd_logcount, revcompd_rev_logcount])
avg_profile = keras.layers.Average()([fwd_profile, revcompd_rev_profile])

#Create a model that goes from the inputs to the averaged output
siamese_model = keras.models.Model(inputs=[fwd_sequence_input,
                                           fwd_patchcap_logcount,
                                           fwd_patchcap_profile],
                                   outputs=[avg_logcount, avg_profile])
```
