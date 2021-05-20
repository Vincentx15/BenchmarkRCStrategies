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
    return res_dict, epochs


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
    return res_dict


def use_res_dict(res_dict, which_plot_idx=None):
    # We need to aggregate per seed :
    # aggregated is a list of arrays of shape (n_seed, epochs)
    aggregated = collections.defaultdict(list)
    for key, value in res_dict.items():
        new_key = key.split(' with seed')[0]
        aggregated[new_key].append(value)

    # GET BARPLOT
    do_barplot = True
    # When running with profile data, avoids bugs
    which_plot_idx = which_plot_idx if which_plot_idx is not None else 3
    which_plot = ['a_n', 'ks', 'binary', 'reduced', 'bpn']
    which_plot = which_plot[which_plot_idx]
    if do_barplot:
        if which_plot == 'a_n':
            rename_methods = {'Equinet 0a_n': '0',
                              'Equinet 25a_n': '25',
                              'Equinet 50a_n': '50',
                              'Equinet 75a_n': '75',
                              'Equinet 100a_n': '100'}
        if which_plot == 'ks':
            rename_methods = {'k=1': '1',
                              'k=2': '2',
                              'k=3': '3',
                              'k=4': '4'}

        # For the best ones plotting
        if which_plot == 'binary':
            rename_methods = {'non equivariant': 'Standard',
                              'RCPS with k=1': 'RCPS',
                              'RCPS with k=2': 'Regular',
                              'Equinet 75a_n with k=2': 'Irrep'}
        if which_plot == 'reduced':
            # For the reduction
            rename_methods = {'reduced_non_equivariant': 'Standard reduced',
                              'reduced_equinet_2_75': 'Equinet reduced',
                              # 'reduced_post_hoc': 'Post-hoc reduced',
                              'non equivariant': 'Standard',
                              'Equinet 75a_n with k=2': 'Equinet'
                              }
        if which_plot == 'bpn':
            rename_methods = {'non equivariant': 'Standard',
                              'RCPS_2': 'Regular_2',
                              'best_equi ': 'Irrep_2',
                              # 'best_equi_aug': 'Irrep_2_aug',
                              'RCPS ': 'RCPS',
                              'equi_75_k1': 'Irrep_1'}
        query_keys = list(rename_methods.keys())

        do_pvalues = True
        if do_pvalues:
            grouped_dict = collections.defaultdict(list)
            for key, value in res_dict.items():
                new_key = key.split(' with seed')[0]
                if which_plot == 'bpn':
                    _, dataset = new_key.split(' with dataset=')
                    last_value = value
                else:
                    _, dataset = new_key.split(' with tf=')
                    last_value = value[-1]
                # if not 'MAX' in key:
                #     continue
                for query_key in query_keys:
                    if query_key in key:
                        # if 'reduced_equinet_2_75' in key:
                        #     print(key)
                        grouped_dict[query_key].append(last_value)
            # Just reorder the values
            # grouped_dict = {k: v for k, v in sorted(grouped_dict.items(), key=lambda item: item[0])}
            ordered_grouped_dict = {k: grouped_dict[k] for k in query_keys}
            grouped_dict = ordered_grouped_dict

            print()
            print('mean dict values')
            mean_dict = {k: np.mean(np.array(v)) for k, v in grouped_dict.items()}
            sorted_mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
            for key, value in sorted_mean_dict.items():
                print(key, value)

            if which_plot == 'a_n':
                key_reference = 'Equinet 75a_n'
            if which_plot == 'ks':
                key_reference = 'k=2'
            if which_plot == 'binary':
                key_reference = 'Equinet 75a_n with k=2'
            if which_plot == 'reduced':
                key_reference = 'reduced_equinet_2_75'
            if which_plot == 'bpn':
                key_reference = 'RCPS '

            print()
            print('mean_dict p-values')
            reference = grouped_dict[key_reference]
            np_reference = np.array(reference)
            grouped_pvalues = dict()
            # print(np_reference.shape)
            grouped_pvalues[key_reference] = 0
            for key, value in grouped_dict.items():
                np_values = np.array(value)
                print(key, np_values.shape)
                from scipy.stats import wilcoxon
                from scipy import stats
                if np_reference.shape != np_values.shape or not np.allclose(np_reference, np_values):
                    w, p = wilcoxon(np_reference, np_values)
                    t, p2 = stats.ttest_ind(np_reference, np_values)
                    print(key, p, p2)
                    grouped_pvalues[key] = p2

        do_plot = True
        if do_plot:
            pandas_dict = collections.defaultdict(list)
            # pandas_series = {"method": method, "dataset": dataset, "value": value}
            for key, value in res_dict.items():
                keep = False
                for filter in query_keys:
                    if filter in key:
                        keep = True
                        break
                if not keep:
                    continue
                # if 'rc_post_hoc' in key:
                #     continue
                new_key = key.split(' with seed')[0]
                if which_plot == 'bpn':
                    _, dataset = new_key.split(' with dataset=')
                    last_value = value
                else:
                    _, dataset = new_key.split(' with tf=')
                    last_value = value[-1]
                # print(filter)
                method = rename_methods[filter]
                pandas_dict['Method'].append(method)
                pandas_dict['Dataset'].append(dataset)
                pandas_dict['AuROC'].append(last_value)
                if which_plot in ('binary', 'reduced', 'bpn'):
                    pandas_dict['Method'].append(method)
                    pandas_dict['Dataset'].append('Overall')
                    pandas_dict['AuROC'].append(last_value)

            pandas_results = pd.DataFrame(pandas_dict)
            MODEL_TYPES = list(rename_methods.values())
            if which_plot == 'a_n':
                new_name = "Rate of \'+1\' representation"
                pandas_results = pandas_results.rename(columns={"Method": new_name})
                ax = sns.barplot(x=new_name, y="AuROC", data=pandas_results, order=MODEL_TYPES)
                # ax.legend(loc="upper left", ncol=1)
                plt.ylim(0.97, 0.99)
                plt.savefig('figs/a_n.pdf', bbox_inches="tight")
                plt.show()
            if which_plot == 'ks':
                new_name = "Length of k-mer used"
                pandas_results = pandas_results.rename(columns={"Method": new_name})
                ax = sns.barplot(x=new_name, y="AuROC", data=pandas_results, order=MODEL_TYPES)
                # ax.legend(loc="upper left", ncol=1)
                plt.ylim(0.982, 0.989)
                plt.savefig('figs/ks.pdf', bbox_inches="tight")
                plt.show()
            if which_plot == 'binary':
                ax = sns.barplot(x="Dataset", y="AuROC", hue="Method", data=pandas_results,
                                 order=['CTCF', 'MAX', 'SPI1', 'Overall'], hue_order=MODEL_TYPES)
                # ax.legend(loc="upper left", ncol=1)
                ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=1)
                plt.ylim(0.975, 0.995)
                plt.savefig('figs/binary_barplot.pdf', bbox_inches="tight")
                plt.show()
            if which_plot == 'reduced':
                ax = sns.barplot(x="Dataset", y="AuROC", hue="Method", data=pandas_results,
                                 order=['CTCF', 'MAX', 'SPI1', 'Overall'], hue_order=MODEL_TYPES)
                ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=1)
                ax.legend(loc="upper center", ncol=2)
                plt.ylim(0.93, 1.005)
                plt.savefig('figs/reduced_barplot.pdf', bbox_inches="tight")
                plt.show()
            if which_plot == 'bpn':
                new_name_value = "Spearman Correlation"
                pandas_results = pandas_results.rename(columns={"AuROC": new_name_value})
                ax = sns.barplot(x="Dataset", y=new_name_value, hue="Method", data=pandas_results,
                                 order=['SOX2', 'OCT4', 'KLF4', 'NANOG', 'Overall'], hue_order=MODEL_TYPES)
                plt.ylim(0.25, 0.45)
                # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=1)
                # plt.savefig('figs/binary_barplot.pdf', bbox_inches="tight")
                ax.legend(loc="upper left", ncol=1)
                plt.savefig('figs/bpn.pdf')
                plt.show()

    # TO PRINT ALL VALUES, get only the final means
    print_all_and_means = False
    if print_all_and_means:
        printer = {}
        for key, value in aggregated.items():
            mean = np.mean(value, axis=0)[-1]
            printer[key] = mean
        sorted_printer = {k: v for k, v in sorted(printer.items(), key=lambda item: item[1])}
        for key, value in sorted_printer.items():
            # if not 'MAX' in key:
            #     continue
            print(key, value)

        # === GET AGGREGATED VALUES ALONG A CERTAIN SET OF KEYS ===
        # mean dict takes the mean of the final performance aggregated along key lines
        # mean_keys = ['Equinet 0a_n', 'Equinet 25a_n', 'Equinet 50a_n', 'Equinet 75a_n', 'Equinet 100a_n']
        # mean_keys = ['k=1', 'k=4', 'k=2', 'k=3']
        mean_keys = ['Equinet 75a_n with k=2', 'RCPS with k=1', 'RCPS with k=2', 'non equivariant']
        # mean_keys = set([k.split('with tf')[0] for k in sorted_printer.keys()])
        grouped_dict = collections.defaultdict(list)
        for key, value in sorted_printer.items():
            # if 'Equinet' in key:
            #     continue
            if not 'MAX' in key:
                continue
            print(key, value)
            for query_key in mean_keys:
                if query_key in key:
                    grouped_dict[query_key].append(value)
        # Just reorder the values
        # grouped_dict = {k: v for k, v in sorted(grouped_dict.items(), key=lambda item: item[0])}
        ordered_grouped_dict = {k: grouped_dict[k] for k in mean_keys}
        grouped_dict = ordered_grouped_dict

        print()
        print('mean dict values')
        mean_dict = {k: np.mean(np.array(v)) for k, v in grouped_dict.items()}
        sorted_mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
        for key, value in sorted_mean_dict.items():
            print(key, value)

        print()
        print('mean_dict p-values')
        key_reference = 'Equinet 75a_n with k=2'
        # key_reference = 'Equinet 75a_n'
        reference = grouped_dict[key_reference]
        np_reference = np.array(reference)
        print(np_reference.shape)
        grouped_pvalues = dict()
        grouped_pvalues[key_reference] = 0
        for key, value in grouped_dict.items():
            np_values = np.array(value)
            from scipy.stats import wilcoxon
            from scipy import stats
            if not np.allclose(np_reference, np_values):
                w, p = wilcoxon(np_reference, np_values)
                t, p2 = stats.ttest_ind(np_reference, np_values)
                print(key, p, p2)
                grouped_pvalues[key] = p2

    do_line_plot = False
    if do_line_plot:

        # mean_keys = ['Equinet 0a_n', 'Equinet 25a_n', 'Equinet 50a_n', 'Equinet 75a_n', 'Equinet 100a_n']
        mean_keys = ['k=1', 'k=2', 'k=3', 'k=4']
        # mean_keys = set([k.split('with tf')[0] for k in sorted_printer.keys()])
        grouped_dict = collections.defaultdict(list)
        for key, value in aggregated.items():
            # keys are just stripped of seeds, and values are of shape (n_seed, epochs): shape =(10,6)
            for query_key in mean_keys:
                # if query_key in key and 'MAX' in key:
                if query_key in key:
                    final_values = [val[-1] for val in value]
                    grouped_dict[query_key].extend(final_values)
        # Just reorder the values
        # grouped_dict = {k: v for k, v in sorted(grouped_dict.items(), key=lambda item: item[0])}
        ordered_grouped_dict = {k: grouped_dict[k] for k in mean_keys}
        grouped_dict = ordered_grouped_dict

        # Make it into a pandas DF
        pandas_dict = collections.defaultdict(list)
        for method, values in grouped_dict.items():
            for value in values:
                pandas_dict['Method'].append(method)
                pandas_dict['AuROC'].append(value)

        pandas_results = pd.DataFrame(pandas_dict)

        # For all models
        MODEL_TYPES = ['Standard', 'Equinet', 'RCPS', 'RCPS_2']
        # For reduced models
        MODEL_TYPES = ['Standard', 'Equinet', 'Post_hoc']
        ax = sns.barplot(x="Dataset", y="AuROC", hue="Method", data=pandas_results, hue_order=MODEL_TYPES)
        ax.legend(loc="upper left", ncol=1)
        # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=1)
        plt.ylim(0.92, 1)

        # print()
        # print('mean_dict p-values')
        # key_reference = 'k=2'
        # reference = grouped_dict[key_reference]
        # np_reference = np.array(reference)
        # grouped_pvalues = dict()
        # grouped_pvalues[key_reference] = 0
        # for key, value in grouped_dict.items():
        #     np_values = np.array(value)
        #     from scipy.stats import wilcoxon
        #     from scipy import stats
        #     if not np.allclose(np_reference, np_values):
        #         w, p = wilcoxon(np_reference, np_values)
        #         t, p2 = stats.ttest_ind(np_reference, np_values)
        #         print(key, p, p2)
        #         grouped_pvalues[key] = p2
        #
        # names = list(grouped_dict.keys())
        # values = list([np.mean(val) for val in grouped_dict.values()])
        # yerr = list([np.std(val) / np.sqrt(len(val)) for val in grouped_dict.values()])
        # # print(values, yerr)
        #
        # fig, ax = plt.subplots()
        # # ax.bar(names, values)
        # ax.errorbar(names, values, yerr=yerr, label='both limits (default)')
        # fig.suptitle('Impact of the use of different lengths k-mers')
        # plt.ylim(0.9825, 0.988)
        # plt.show()
        # return

    # === GET THE PLOTTING FORM for each key ===
    get_new_plot_epoch = False
    if get_new_plot_epoch:
        rename_methods = {'Equinet 75a_n with k=2': 'Irrep',
                          'RCPS with k=2': 'Regular',
                          'RCPS with k=1': 'RCPS',
                          'non equivariant': 'Standard'}
        rename_methods = {'reduced_non_equivariant': 'Standard reduced',
                          'reduced_equinet_2_75': 'Equinet reduced',
                          # 'reduced_post_hoc': 'Post-hoc reduced',
                          # 'non equivariant': 'Standard',
                          # 'Equinet 75a_n with k=2': 'Equinet'
                          }
        query_keys = list(rename_methods.keys())

        grouped_dict = collections.defaultdict(list)
        for key, value in res_dict.items():
            new_key = key.split(' with seed')[0]
            _, dataset = new_key.split(' with tf=')
            for query_key in query_keys:
                if query_key in key:
                    # if 'reduced_equinet_2_75' in key:
                    #     print(key)
                    grouped_dict[query_key].append(value)
        transformed = {}
        for key, value in grouped_dict.items():
            mean = np.mean(value, axis=0)
            std = np.std(value, axis=0)
            n_samples = np.sqrt(len(value))
            transformed[key] = mean, std / n_samples
        fig, ax = plt.subplots(1)
        # Iterating over queries ensures the order
        for old_name, new_name in rename_methods.items():
            means, errors = transformed[old_name]
            ax.plot(epochs, means, lw=2, label=new_name)
            min_bound = means - errors
            max_bound = means + errors
            # min_bound = np.max((means - stds, 0.965 * np.ones_like(means)), axis=0)
            # max_bound = np.min((means + stds, 0.995 * np.ones_like(means)), axis=0)
            ax.fill_between(epochs, min_bound, max_bound, alpha=0.1)
            # ax.fill_between(epochs, min_bound, max_bound, alpha=0.5, color=color)
        # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=2)
        # ax.legend(loc="lower right", ncol=2)
        ax.legend(loc="lower right", ncol=1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Auroc')
        plt.ylim(0.9, 0.97)
        ax.grid()
        # plt.savefig('figs/binary_max.pdf')
        plt.savefig('figs/reduced_epochs.pdf', bbox_inches="tight")
        plt.show()

    get_plot_epoch = False
    if get_plot_epoch:
        transformed = {}
        to_plot = ['reduced_equinet_2_75', 'reduced_post_hoc', 'reduced_non_equivariant']
        for key, value in aggregated.items():
            keep = False
            for filter in to_plot:
                if filter in key:
                    keep = True
                    break
            if not keep:
                continue
            if 'SPI1' not in key:
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
        return transformed


def plot_epochs(res_dict, epochs):
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
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncol=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Auroc')
    plt.ylim(0.94, 0.9925)
    ax.grid()
    # plt.savefig('figs/binary_max.pdf')
    plt.savefig('figs/binary_max.pdf', bbox_inches="tight")
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


def deprecated_tools(res_dict):
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
    res_dict_binary, epochs = group_logfile(logfile='archives_results/logfile_all.txt')
    res_dict_bpn = group_logfile_bpn()

    use_res_dict(res_dict_bpn, which_plot_idx=4)
    # use_res_dict(res_dict_binary)

    # res_dict, epochs = group_logfile(logfile='archives_results/logfile_all.txt')
    # plot_res_dict(res_dict, epochs)

    # fix_logfile_ctcf()
