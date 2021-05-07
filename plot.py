import collections

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
    # aggregated is a list of arrays of shape (n_seed, epochs)
    aggregated = collections.defaultdict(list)
    for key, value in res_dict.items():
        new_key = key.split(' with seed')[0]
        aggregated[new_key].append(value)

    # GET BARPLOT
    do_barplot = False
    if do_barplot:

        rename_methods = {'non equivariant': 'Standard', 'best_equi': 'Equinet', 'RCPS': 'Rcps'}
        pandas_dict = collections.defaultdict(list)
        # pandas_series = {"method": method, "dataset": dataset, "value": value}
        for key, value in res_dict.items():
            # if 'rc_post_hoc' in key:
            #     continue
            new_key = key.split(' with seed')[0]

            method, dataset = new_key.split(' with tf=')
            # method = rename_methods[method]
            pandas_dict['Method'].append(method)
            pandas_dict['Dataset'].append(dataset)
            last_value = value[-1]
            pandas_dict['AuROC'].append(last_value)
        pandas_results = pd.DataFrame(pandas_dict)

        # MODEL_TYPES = ['Standard', 'Equinet', 'Rcps']
        # ax = sns.barplot(x="Dataset", y="Spearman Correlation", hue="Method", data=pandas_results, hue_order=MODEL_TYPES)
        ax = sns.barplot(x="Dataset", y="AuROC", hue="Method", data=pandas_results)
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=2)
        plt.ylim(0.96, 1)

        # sns.catplot(hue="Method", y="AuROC",
        #             col="Dataset",
        #             data=pandas_results, kind="bar")
        # plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig('figs/binary_barplot.pdf', bbox_inches="tight")
        # plt.savefig('figs/binary_barplot.pdf')
        # plt.show()
        return

        # === TO PRINT P VALUES COMPARED TO SMTHING ===
    # reference = aggregated['Equinet with k=3 with tf=SPI1']
    # np_reference = np.array((reference))[:, -1]
    # for key, value in aggregated.items():
    #     np_values = np.array(value)[:, -1]
    #     from scipy.stats import wilcoxon
    #     from scipy import stats
    #     if not np.allclose(np_reference, np_values):
    #         w, p = wilcoxon(np_reference, np_values)
    #         t, p2 = stats.ttest_ind(np_reference, np_values)
    #         print(key, p, p2)

    # TO PRINT ALL VALUES, get only the final means
    printer = {}
    for key, value in aggregated.items():
        mean = np.mean(value, axis=0)[-1]
        printer[key] = mean
    sorted_printer = {k: v for k, v in sorted(printer.items(), key=lambda item: item[1])}
    for key, value in sorted_printer.items():
        # if not 'SPI1' in key:
        #     continue
        print(key, value)

    # === GET AGGREGATED VALUES ALONG A CERTAIN SET OF KEYS ===
    # mean dict takes the mean of the final performance aggregated along key lines
    # mean_keys = ['Equinet 100a_n', 'Equinet 75a_n', 'Equinet 25a_n', 'Equinet 0a_n', 'Equinet w']
    # mean_keys = ['k=1', 'k=4', 'k=2', 'k=3']
    mean_keys = set([k.split('with tf')[0] for k in sorted_printer.keys()])
    mean_dict = collections.defaultdict(list)
    for key, value in sorted_printer.items():
        # if 'Equinet' in key:
        #     continue
        print(key, value)
        for query_key in mean_keys:
            if query_key in key:
                mean_dict[query_key].append(value)

    print()
    print('mean dict values')
    mean_dict = {k: np.mean(np.array(v)) for k, v in mean_dict.items()}
    sorted_mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
    for key, value in sorted_mean_dict.items():
        print(key, value)

    # print()
    # print('mean_dict p-values')
    # reference = mean_dict['k=2']
    # np_reference = np.array(reference)
    # for key, value in sorted_mean_dict.items():
    #     np_values = np.array(value)
    #     from scipy.stats import wilcoxon
    #     from scipy import stats
    #     if not np.allclose(np_reference, np_values):
    #         w, p = wilcoxon(np_reference, np_values)
    #         t, p2 = stats.ttest_ind(np_reference, np_values)
    #         print(key, p, p2)

    # === GET THE PLOTTING FORM for each key ===
    transformed = {}
    for key, value in aggregated.items():
        if not ('post_hoc' in key or 'RCPS with k=1' in key):
            if not ('k=2' in key or 'k=3' in key):
                continue
            if '25a_n' in key or '0a_n' in key:
                continue
        if 'SPI1' not in key:
            continue
        # if key not in ["RCPS with k=3 with tf=MAX", "RCPS with k=2 with tf=MAX", "rc_post_hoc with tf=MAX"]:
        #     continue
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
        color = None
        # if key.startswith('Equinet 100a_n '):
        #     color = 'b'
        # elif key.startswith('Equinet 75a_n '):
        #     color = 'g'
        # elif key.startswith('Equinet 25a_n '):
        #     color = 'r'
        # elif key.startswith('Equinet 0a_n '):
        #     color = 'c'
        # elif key.startswith('Equinet w'):
        #     color = 'm'
        # else:
        #     continue
        means, stds = value
        stds = stds
        ax.plot(epochs, means, lw=2, label=key, color=color)
        min_bound = means - stds
        max_bound = means + stds
        # min_bound = np.max((means - stds, 0.965 * np.ones_like(means)), axis=0)
        # max_bound = np.min((means + stds, 0.995 * np.ones_like(means)), axis=0)
        ax.fill_between(epochs, min_bound, max_bound, alpha=0.1)
        # ax.fill_between(epochs, min_bound, max_bound, alpha=0.5, color=color)
    ax.set_title('Auroc with different models')
    ax.legend(loc='lower right')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Auroc')
    plt.ylim(0.98, 0.9925)
    ax.grid()
    plt.savefig('figs/binary_max.pdf')
    plt.show()


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


def group_logfile_bpn(logfile='archives_results/logfile_bpn_all.txt'):
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
                new_exp = False
            # This is useful to get 'new_exp' blocks
            elif line == '':
                new_exp = True
                res_dict[key] = key_means
            else:
                jsd, pears, spear, mse = line.split()
                jsd, pears, spear, mse = float(jsd), float(pears), float(spear), float(mse)
                key_means = spear

    # We need to aggregate per seed :
    # aggregated is a list of arrays of shape (n_seed, epochs)
    aggregated = collections.defaultdict(list)
    for key, value in res_dict.items():
        new_key = key.split(' with seed')[0]
        aggregated[new_key].append(value)

    rename_methods = {'non equivariant': 'Standard', 'best_equi': 'Equinet', 'RCPS': 'Rcps'}
    pandas_dict = collections.defaultdict(list)
    # pandas_series = {"method": method, "dataset": dataset, "value": value}
    for key, value in res_dict.items():
        if 'rc_post_hoc' in key:
            continue
        new_key = key.split(' with seed')[0]
        method, dataset = new_key.split(' with dataset=')
        method = rename_methods[method]
        pandas_dict['Method'].append(method)
        pandas_dict['Dataset'].append(dataset)
        pandas_dict['Spearman Correlation'].append(value)
    pandas_results = pd.DataFrame(pandas_dict)

    # === TO PRINT P VALUES COMPARED TO SMTHING ===
    # reference = aggregated['Equinet with k=3 with tf=SPI1']
    # np_reference = np.array((reference))[:, -1]
    # for key, value in aggregated.items():
    #     np_values = np.array(value)[:, -1]
    #     from scipy.stats import wilcoxon
    #     from scipy import stats
    #     if not np.allclose(np_reference, np_values):
    #         w, p = wilcoxon(np_reference, np_values)
    #         t, p2 = stats.ttest_ind(np_reference, np_values)
    #         print(key, p, p2)

    # TO PRINT ALL VALUES, get only the final means
    printer = {}
    for key, value in aggregated.items():
        mean = np.mean(value)
        printer[key] = mean
    sorted_printer = {k: v for k, v in sorted(printer.items(), key=lambda item: item[1])}
    for key, value in sorted_printer.items():
        pass
        if not 'RCPS' in key:
            continue
        print(key, value)

    # === GET AGGREGATED VALUES ALONG A CERTAIN SET OF KEYS ===
    # mean dict takes the mean of the final performance aggregated along key lines
    # mean_keys = ['Equinet 100a_n', 'Equinet 75a_n', 'Equinet 25a_n', 'Equinet 0a_n', 'Equinet w']
    # mean_keys = ['k=1', 'k=4', 'k=2', 'k=3']
    mean_keys = set([k.split('with dataset')[0] for k in sorted_printer.keys()])
    mean_dict = collections.defaultdict(list)
    for key, value in sorted_printer.items():
        # if 'Equinet' in key:
        #     continue
        for query_key in mean_keys:
            if query_key in key:
                mean_dict[query_key].append(value)

    print()
    print('mean dict values')
    mean_dict = {k: np.mean(np.array(v)) for k, v in mean_dict.items()}
    sorted_mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
    for key, value in sorted_mean_dict.items():
        print(key, value)

    # print()
    # print('mean_dict p-values')
    # reference = mean_dict['k=2']
    # np_reference = np.array(reference)
    # for key, value in sorted_mean_dict.items():
    #     np_values = np.array(value)
    #     from scipy.stats import wilcoxon
    #     from scipy import stats
    #     if not np.allclose(np_reference, np_values):
    #         w, p = wilcoxon(np_reference, np_values)
    #         t, p2 = stats.ttest_ind(np_reference, np_values)
    #         print(key, p, p2)

    # === GET THE PLOTTING FORM for each key ===
    # transformed = {}
    # for key, value in aggregated.items():
    # if not ('post_hoc' in key or 'RCPS with k=1' in key):
    #     if not ('k=2' in key or 'k=3' in key):
    #         continue
    #     if '25a_n' in key or '0a_n' in key:
    #         continue
    # if 'MAX' not in key:
    #     continue
    # if key not in ["RCPS with k=3 with tf=MAX", "RCPS with k=2 with tf=MAX", "rc_post_hoc with tf=MAX"]:
    #     continue
    # value = np.array(value)
    # mean = np.mean(value)
    # std = np.std(value)
    # n_samples = np.sqrt(len(value))
    # transformed[key] = mean, std / n_samples
    MODEL_TYPES = ['Standard', 'Equinet', 'Rcps']
    ax = sns.barplot(x="Dataset", y="Spearman Correlation", hue="Method", data=pandas_results, hue_order=MODEL_TYPES)
    plt.ylim(0.25, 0.45)
    plt.savefig('figs/bpn.pdf')
    plt.show()
    # sns.catplot(x="method", y="value",
    #                 hue="dataset", col="time",
    #                 data=pandas_results, kind="bar",
    #                 height=4, aspect=.7);


if __name__ == '__main__':
    pass
    res_dict, epochs = group_logfile(logfile='archives_results/logfile_all.txt')
    plot_res_dict(res_dict, epochs)

    # fix_logfile_ctcf()

    # res_dict = group_logfile_bpn()
