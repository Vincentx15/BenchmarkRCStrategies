import collections

import matplotlib.pyplot as plt
import numpy as np


# Get the data
def group_aggregated(aggregated_file='outfile_reproduce.txt'):
    with open(aggregated_file, 'r') as f:
        res_dict = {}
        new_exp = True
        key = None
        means = list()
        stds = list()
        epochs = list()
        for line in f.readlines():
            line = line.strip()
            if line.startswith('Experiments results'):
                continue
            if new_exp:
                key = line
                means = list()
                stds = list()
                epochs = list()
                new_exp = False
                # continue
            elif line == '':
                new_exp = True
                means, stds = np.array(means), np.array(stds)
                res_dict[key] = (means, stds)
            else:
                new_exp = False
                # print(line)
                epoch, mean, std = line.split()
                epoch, mean, std = float(epoch), float(mean), float(std)
                means.append(mean)
                stds.append(std)
                epochs.append(epoch)
    # print(res_dict)
    return res_dict, epochs


def group_logfile(logfile='logfile_reproduce.txt'):
    """
    directly parse the logfile
    """
    with open(logfile, 'r') as f:
        res_dict = {}
        new_exp = True
        for line in f.readlines():
            line = line.strip()
            if line.startswith('Log of the experiments'):
                continue
            # This code is read when we start a block
            if new_exp:
                key = line
                key_means = list()
                # val_means = list()
                epochs = list()
                new_exp = False
            # This is useful to get 'new_exp' blocks
            elif line == '':
                new_exp = True
                res_dict[key] = key_means
                # res_dict[key] = key_means, val_means
            else:
                new_exp = False
                epoch, val_mean, test_mean = line.split()
                epoch, val_mean, test_mean = float(epoch), float(val_mean), float(test_mean)
                key_means.append(test_mean)
                # val_means.append(val_mean)
                epochs.append(epoch)

    # We need to aggregate per seed :
    aggregated = collections.defaultdict(list)
    for key, value in res_dict.items():
        new_key = key.split(' with seed')[0]
        aggregated[new_key].append(value)

    reference = aggregated['Equinet with k=3 with tf=SPI1']
    np_reference = np.array((reference))[:, -1]
    for key, value in aggregated.items():
        np_values = np.array(value)[:, -1]
        from scipy.stats import wilcoxon
        from scipy import stats
        if not np.allclose(np_reference, np_values):
            w, p = wilcoxon(np_reference, np_values)
            t, p2 = stats.ttest_ind(np_reference, np_values)
            print(key, p, p2)

    transformed = {}
    for key, value in aggregated.items():
        if 'MAX' in key:
            continue
        value = np.array(value)
        # mean, std = np.mean(value, axis=(0,1)), np.std(value, axis=(0,1))/np.sqrt(len(value))
        # transformed[key] = mean, std
        # print(mean,std)
        mean = np.mean(value, axis=0)
        std = np.std(value, axis=0)
        n_samples = np.sqrt(len(value))
        transformed[key] = mean, std / n_samples
        # print(mean, std / n_samples)
    return transformed, epochs


def plot_res_dict(res_dict, epochs):
    fig, ax = plt.subplots(1)
    for key, value in res_dict.items():
        means, stds = value
        stds = stds
        ax.plot(epochs, means, lw=2, label=key)
        # min_bound = np.max((means - stds, 0.965 * np.ones_like(means)), axis=0)
        # max_bound = np.min((means + stds, 0.995 * np.ones_like(means)), axis=0)
        # ax.fill_between(epochs, min_bound, max_bound, alpha=0.5)
    ax.set_title('Auroc with different models')
    ax.legend(loc='lower right')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Auroc')
    ax.grid()
    plt.show()


# res_dict, epochs = group_aggregated()
# plot_res_dict(res_dict, epochs)

# res_dict, epochs = group_logfile(logfile='logfile_reproduce_tf.txt')
# res_dict, epochs = group_logfile(logfile='results_archives/temp_logfile_all.txt')
# plot_res_dict(res_dict, epochs)
# print(res_dict)

def fix_logfile_ctcf(logfile='logfile_reproduce.txt', logfile_fixed='logfile_reproduce_fixed.txt'):
    """
    Just add the with tf=CTCF keyword to the original logfile
    """
    with open(logfile, 'r') as f:
        lines = f.readlines()
    with open(logfile_fixed, 'w') as f:
        for line in lines:
            splits = line.split('with seed=')
            if len(splits) == 1:
                f.write(line)
            else:
                newline = ''.join([splits[0], 'with tf=CTCF with seed=', splits[1]])
                f.write(newline)


fix_logfile_ctcf()
