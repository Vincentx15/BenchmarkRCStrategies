import os
import gzip
import random
import numpy as np
import scipy
from scipy.special import softmax

from seqdataloader.batchproducers.coordbased.coordbatchproducers import SimpleCoordsBatchProducer
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import AbstractCoordBatchTransformer
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp

import keras


class GeneralReverseComplement(AbstractCoordBatchTransformer):
    def __call__(self, coords):
        return [get_revcomp(x) for x in coords]


class RevcompTackedOnSimpleCoordsBatchProducer(SimpleCoordsBatchProducer):
    def _get_coordslist(self):
        return [x for x in self.bed_file.coords_list] + [get_revcomp(x) for x in self.bed_file.coords_list]


def get_inputs_and_targets_coordstoval(dataset, seq_len, out_pred_len):
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


def get_reduced_bed(infile, outfile, size=1000):
    with gzip.open(infile) as f:
        lines = f.readlines()
    selected_indices = sorted(random.sample(list(range(len(lines))), size))
    selected_lines = [lines[selected] for selected in selected_indices]

    with gzip.open(outfile, 'wb') as f:
        for line in selected_lines:
            f.write(line)


def get_train_generator(dataset, inputs_coordstovals, targets_coordstovals, seed, is_aug=False, reduced=False):
    chromsizes_file = os.path.join('data', "mm10.chrom.sizes")

    pos_bed = os.path.join('data', dataset, f"bpnet_{dataset}_train_1k_around_summits.bed.gz")
    if reduced:
        pos_bed_reduced = os.path.join('data', dataset, f"bpnet_{dataset}_reduced_train_1k_around_summits.bed.gz")
        if not os.path.exists(pos_bed_reduced) or True:
            get_reduced_bed(infile=pos_bed, outfile=pos_bed_reduced)
        pos_bed = pos_bed_reduced

    if is_aug:
        coords_transformer = coordbatchtransformers.ReverseComplementAugmenter().chain(
            coordbatchtransformers.UniformJitter(maxshift=200, chromsizes_file=chromsizes_file))
    else:
        coords_transformer = coordbatchtransformers.UniformJitter(maxshift=200, chromsizes_file=chromsizes_file)
    train_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=pos_bed,
            batch_size=64,
            shuffle_before_epoch=True,
            seed=seed),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals,
        coordsbatch_transformer=coords_transformer)

    return train_batch_generator


def get_val_generator(dataset, inputs_coordstovals, targets_coordstovals, seed):
    valid_file = os.path.join('data', dataset, f"bpnet_{dataset}_valid_1k_around_summits.bed.gz")

    val_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=valid_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=seed),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals)

    return val_batch_generator


def get_test_generator(dataset, inputs_coordstovals, targets_coordstovals, seed):
    test_file = os.path.join('data', dataset, f"bpnet_{dataset}_test_1k_around_summits.bed.gz")

    keras_test_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=test_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=seed),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals)

    keras_rc_test_batch_generator = KerasBatchGenerator(
        coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=test_file,
            batch_size=64,
            shuffle_before_epoch=False,
            seed=seed),
        inputs_coordstovals=inputs_coordstovals,
        targets_coordstovals=targets_coordstovals,
        coordsbatch_transformer=GeneralReverseComplement())

    return keras_test_batch_generator, keras_rc_test_batch_generator


def get_generators(dataset, seed, seq_len=1346, out_pred_len=1000, is_aug=False, reduced=False):
    inputs_coordstovals, targets_coordstovals = get_inputs_and_targets_coordstoval(dataset=dataset,
                                                                                   seq_len=seq_len,
                                                                                   out_pred_len=out_pred_len)
    train_generator = get_train_generator(dataset=dataset,
                                          seed=seed,
                                          inputs_coordstovals=inputs_coordstovals,
                                          targets_coordstovals=targets_coordstovals,
                                          is_aug=is_aug,
                                          reduced=reduced)
    val_generator = get_val_generator(dataset=dataset,
                                      seed=seed,
                                      inputs_coordstovals=inputs_coordstovals,
                                      targets_coordstovals=targets_coordstovals)
    test_generator, rc_test_generator = get_test_generator(dataset=dataset,
                                                           seed=seed,
                                                           inputs_coordstovals=inputs_coordstovals,
                                                           targets_coordstovals=targets_coordstovals)
    return train_generator, val_generator, test_generator, rc_test_generator


def bin_array_max(arr, bin_size, pad_value=0):
    """
    Given a NumPy array, returns a binned version of the array along the last
    dimension, where each bin contains the maximum value of its constituent
    elements. If the array is not a length that is a multiple of the bin size,
    then the given pad will be used at the end.
    """
    num_bins = int(np.ceil(arr.shape[-1] / bin_size))
    pad_amount = (num_bins * bin_size) - arr.shape[-1]
    if pad_amount:
        arr = np.pad(
            arr, ([(0, 0)] * (arr.ndim - 1)) + [(0, pad_amount)],
            constant_values=pad_value
        )
    new_shape = arr.shape[:-1] + (num_bins, bin_size)
    return np.max(np.reshape(arr, new_shape), axis=-1)


def binned_profile_corr_mse(true_prof_probs,
                            pred_prof_probs,
                            prof_count_corr_bin_sizes,
                            batch_size=50000):
    """
    Returns the correlations of the true and predicted PROFILE counts (i.e.
    per base or per bin).
    Arguments:
        `true_prof_probs`: a N x T x O x 2 array, containing the true profile
            RAW PROBABILITIES for each task and strand
        `pred_prof_probs`: a N x T x O x 2 array, containing the true profile
            RAW PROBABILITIES for each task and strand
        `batch_size`: performs computation in a batch size of this many samples
    Returns 3 N x T x Z arrays, containing the Pearson correlation, Spearman
    correlation, and mean squared error of the profile predictions (as log
    counts). Correlations/MSE are computed for each sample/task, for each bin
    size in `prof_count_corr_bin_sizes` (strands are pooled together).
    """

    def spearman_corr(arr1, arr2):

        def average_ranks(arr):
            """
            Computes the ranks of the elemtns of the given array along the last
            dimension. For ties, the ranks are _averaged_.
            Returns an array of the same dimension of `arr`.
            """
            # 1) Generate the ranks for each subarray, with ties broken arbitrarily
            sorted_inds = np.argsort(arr, axis=-1)  # Sorted indices
            ranks, ranges = np.empty_like(arr), np.empty_like(arr)
            ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))
            # Put ranks by sorted indices; this creates an array containing the ranks of
            # the elements in each subarray of `arr`
            np.put_along_axis(ranks, sorted_inds, ranges, -1)
            ranks = ranks.astype(int)

            # 2) Create an array where each entry maps a UNIQUE element in `arr` to a
            # unique index for that subarray
            sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
            diffs = np.diff(sorted_arr, axis=-1)
            del sorted_arr  # Garbage collect
            # Pad with an extra zero at the beginning of every subarray
            pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
            del diffs  # Garbage collect
            # Wherever the diff is not 0, assign a value of 1; this gives a set of
            # small indices for each set of unique values in the sorted array after
            # taking a cumulative sum
            pad_diffs[pad_diffs != 0] = 1
            unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)
            del pad_diffs  # Garbage collect

            # 3) Average the ranks wherever the entries of the `arr` were identical
            # `unique_inds` contains elements that are indices to an array that stores
            # the average of the ranks of each unique element in the original array
            unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
            # Each subarray will contain unused entries if there are no repeats in that
            # subarray; this is a sacrifice made for vectorization; c'est la vie
            # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which
            # result in putting the maximum rank in each unique location
            np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
            # We can compute the average rank for each bucket (from the maximum rank for
            # each bucket) using some algebraic manipulation
            diff = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
            unique_avgs = unique_maxes - ((diff - 1) / 2)
            del unique_maxes, diff  # Garbage collect

            # 4) Using the averaged ranks in `unique_avgs`, fill them into where they
            # belong
            avg_ranks = np.take_along_axis(
                unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1
            )

            return avg_ranks

        """
        Computes the Spearman correlation in the last dimension of `arr1` and
        `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are
        both A x B x L arrays, then the correlation of corresponding L-arrays will
        be computed and returned in an A x B array.
        """
        ranks1, ranks2 = average_ranks(arr1), average_ranks(arr2)
        return pearson_corr(ranks1, ranks2)

    def pearson_corr(arr1, arr2):
        """
        Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
        `arr1` and `arr2` must be the same shape. For example, if they are both
        A x B x L arrays, then the correlation of corresponding L-arrays will be
        computed and returned in an A x B array.
        """
        mean1 = np.mean(arr1, axis=-1, keepdims=True)
        mean2 = np.mean(arr2, axis=-1, keepdims=True)
        dev1, dev2 = arr1 - mean1, arr2 - mean2
        sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
        numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
        var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
        denom = np.sqrt(var1 * var2)

        # Divide numerator by denominator, but use NaN where the denominator is 0
        return np.divide(
            numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
        )

    def mean_squared_error(arr1, arr2):
        """
        Computes the mean squared error in the last dimension of `arr1` and `arr2`.
        `arr1` and `arr2` must be the same shape. For example, if they are both
        A x B x L arrays, then the MSE of corresponding L-arrays will be computed
        and returned in an A x B array.
        """
        return np.mean(np.square(arr1 - arr2), axis=-1)

    num_samples, num_tasks = true_prof_probs.shape[:2]
    num_bin_sizes = len(prof_count_corr_bin_sizes)
    pears = np.zeros((num_samples, num_tasks, num_bin_sizes))
    spear = np.zeros((num_samples, num_tasks, num_bin_sizes))
    mse = np.zeros((num_samples, num_tasks, num_bin_sizes))

    # Combine the profile length and strand dimensions (i.e. pool strands)
    new_shape = (num_samples, num_tasks, -1)
    true_prof_probs_flat = np.reshape(true_prof_probs, new_shape)
    pred_prof_probs_flat = np.reshape(pred_prof_probs, new_shape)

    for i, bin_size in enumerate(prof_count_corr_bin_sizes):
        true_prob_binned = bin_array_max(true_prof_probs_flat, bin_size)
        pred_prob_binned = bin_array_max(pred_prof_probs_flat, bin_size)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            true_batch = true_prob_binned[start:end, :, :]
            pred_batch = pred_prob_binned[start:end, :, :]

            pears[start:end, :, i] = pearson_corr(true_batch, pred_batch)
            x = spearman_corr(true_batch, pred_batch)
            spear[start:end, :, i] = spearman_corr(true_batch, pred_batch)
            mse[start:end, :, i] = mean_squared_error(true_batch, pred_batch)

    return pears, spear, mse


def profile_jsd(true_prof_probs, pred_prof_probs, jsd_smooth_kernel_sigma):
    """
    Computes the Jensen-Shannon divergence of the true and predicted profiles
    given their raw probabilities or counts. The inputs will be renormalized
    prior to JSD computation, so providing either raw probabilities or counts
    is sufficient.
    Arguments:
        `true_prof_probs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, O is the output profile length;
            contains the true profiles for each task and strand, as RAW
            PROBABILITIES or RAW COUNTS
        `pred_prof_probs`: N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES or RAW
            COUNTS
    Returns an N x T array, where the JSD is computed across the profiles and
    averaged between the strands, for each sample/task.
    """

    def _kl_divergence(probs1, probs2):
        """
        Computes the KL divergence in the last dimension of `probs1` and `probs2`
        as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
        if they are both A x B x L arrays, then the KL divergence of corresponding
        L-arrays will be computed and returned in an A x B array. Does not
        renormalize the arrays. If probs2[i] is 0, that value contributes 0.
        """
        quot = np.divide(
            probs1, probs2, out=np.ones_like(probs1),
            where=((probs1 != 0) & (probs2 != 0))
            # No contribution if P1 = 0 or P2 = 0
        )
        return np.sum(probs1 * np.log(quot), axis=-1)

    def jensen_shannon_distance(probs1, probs2):
        """
        Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and
        `probs2`. `probs1` and `probs2` must be the same shape. For example, if they
        are both A x B x L arrays, then the KL divergence of corresponding L-arrays
        will be computed and returned in an A x B array. This will renormalize the
        arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
        the resulting JSD will be NaN.
        """
        # Renormalize both distributions, and if the sum is NaN, put NaNs all around
        probs1_sum = np.sum(probs1, axis=-1, keepdims=True)
        probs1 = np.divide(
            probs1, probs1_sum, out=np.full_like(probs1, np.nan),
            where=(probs1_sum != 0)
        )
        probs2_sum = np.sum(probs2, axis=-1, keepdims=True)
        probs2 = np.divide(
            probs2, probs2_sum, out=np.full_like(probs2, np.nan),
            where=(probs2_sum != 0)
        )

        mid = 0.5 * (probs1 + probs2)
        return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))

    # Transpose to N x T x 2 x O, as JSD is computed along last dimension
    true_prof_swap = np.swapaxes(true_prof_probs, 2, 3)
    pred_prof_swap = np.swapaxes(pred_prof_probs, 2, 3)

    # Smooth the profiles
    if jsd_smooth_kernel_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = jsd_smooth_kernel_sigma, 1
    true_prof_smooth = scipy.ndimage.gaussian_filter1d(
        true_prof_swap, sigma, axis=-1, truncate=truncate
    )
    pred_prof_smooth = scipy.ndimage.gaussian_filter1d(
        pred_prof_swap, sigma, axis=-1, truncate=truncate
    )

    jsd = jensen_shannon_distance(true_prof_smooth, pred_prof_smooth)
    return np.mean(jsd, axis=-1)  # Average over strands


def post_hoc_from_model_BPN(model, dataset):
    # Let's create the model
    # Define the inputs
    # fwd_sequence_input = keras.models.Input(shape=(1346, 4))
    # fwd_patchcap_logcount = keras.models.Input(shape=(2,))
    # fwd_patchcap_profile = keras.models.Input(shape=(1000, 2))

    def get_inputs(dataset, out_pred_len=1000, input_seq_len=1346):

        inp = keras.layers.Input(shape=(input_seq_len, 4), name='sequence')
        if dataset == "SPI1":
            bias_counts_input = keras.layers.Input(shape=(1,), name="control_logcount")
            bias_profile_input = keras.layers.Input(shape=(out_pred_len, 2),
                                                    name="control_profile")
        else:
            bias_counts_input = keras.layers.Input(shape=(2,), name="patchcap.logcount")
            # if working with raw counts, go from logcount->count
            bias_profile_input = keras.layers.Input(shape=(out_pred_len, 2),
                                                    name="patchcap.profile")
        return inp, bias_counts_input, bias_profile_input

    fwd_sequence_input, fwd_patchcap_logcount, fwd_patchcap_profile = get_inputs(dataset=dataset)

    # RevComp input
    rev_sequence_input = keras.layers.Lambda(lambda x: x[:, ::-1, ::-1])(fwd_sequence_input)
    rev_patchcap_logcount = keras.layers.Lambda(lambda x: x[:, ::-1])(fwd_patchcap_logcount)
    # note that last axis is NOT fwd vs reverse strand, but different smoothing levels
    # that's why we only flip the middle axis
    rev_patchcap_profile = keras.layers.Lambda(lambda x: x[:, ::-1])(fwd_patchcap_profile)

    # Run the model on the original fwd inputs
    fwd_logcount, fwd_profile = model(
        [fwd_sequence_input, fwd_patchcap_logcount, fwd_patchcap_profile])

    # Run the original model on the reverse inputs
    rev_logcount, rev_profile = model(
        [rev_sequence_input, rev_patchcap_logcount, rev_patchcap_profile])

    # Reverse complement rev_logcount and rev_profile to be compatible with fwd
    revcompd_rev_logcount = keras.layers.Lambda(lambda x: x[:, ::-1])(rev_logcount)
    revcompd_rev_profile = keras.layers.Lambda(lambda x: x[:, ::-1, ::-1])(rev_profile)

    # Average the two
    avg_logcount = keras.layers.Average()([fwd_logcount, revcompd_rev_logcount])
    avg_profile = keras.layers.Average()([fwd_profile, revcompd_rev_profile])

    # Create a model that goes from the inputs to the averaged output
    post_hoc_model = keras.models.Model(inputs=[fwd_sequence_input,
                                                fwd_patchcap_logcount,
                                                fwd_patchcap_profile],
                                        outputs=[avg_logcount, avg_profile])
    return post_hoc_model


def get_test_values(model, test_generator, dataset, post_hoc=False):
    preds_profile = []
    labels_profile = []

    trained_model = post_hoc_from_model_BPN(model, dataset) if post_hoc else model

    for batch_idx in range(len(test_generator)):
        batch_inputs, batch_labels = test_generator[batch_idx]
        test_preds = trained_model.predict(batch_inputs)
        preds_profile.extend(softmax(test_preds[1], axis=1))
        labels_profile.extend(batch_labels['CHIPNexus.%s.profile' % dataset])
    preds_profile = np.array(preds_profile)[:, None, :, :]
    labels_profile = np.array(labels_profile)[:, None, :, :]

    jsd = profile_jsd(true_prof_probs=labels_profile,
                      pred_prof_probs=preds_profile,
                      jsd_smooth_kernel_sigma=3)
    pears, spear, mse = binned_profile_corr_mse(true_prof_probs=labels_profile,
                                                pred_prof_probs=preds_profile,
                                                prof_count_corr_bin_sizes=[1, 5, 10],
                                                batch_size=50000)

    jsd = np.average(jsd),
    pears = np.average(pears),
    spear = np.average(spear),
    mse = np.average(mse),
    return jsd, pears, spear, mse


if __name__ == '__main__':
    pass
