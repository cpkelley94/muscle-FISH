from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os
import pandas as pd
import seaborn as sns

def calculate_ci(real, bootstraps, ci=0.95):
    real_sorted = sorted(real)
    bs_sorted = [sorted(b) for b in bootstraps]
    bs_y = [[1.-(float(j)/len(b)) for j in range(len(b))] for b in bs_sorted]
    y_mins = []
    y_maxs = []

    for x in real_sorted:
        ys = [np.interp(x, b, bs_y[i]) for i, b in enumerate(bs_sorted)]
        y_min = np.percentile(ys, 100*(1.-ci)/2.)
        y_max = np.percentile(ys, 100*(1.+ci)/2.)
        y_mins.append(y_min)
        y_maxs.append(y_max)
    
    return y_mins, y_maxs


# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('folders', type=str, nargs='*', help='Paths to folder containing merged spot intensity data.')
parser.add_argument('-c', '--conditions', nargs='*', help='Names of conditions.', default=None)
parser.add_argument('-o', '--outfolder', help='Path to output directory.', default=os.getcwd())

# parse arguments
args = vars(parser.parse_args())
in_dirs = args['folders']
conditions = args['conditions']
out_dir = args['outfolder']

# check user input
if not conditions:
    conditions = [str(i+1) for i in range(len(in_dirs))]
elif len(conditions) != len(in_dirs):
    conditions = [str(i+1) for i in range(len(in_dirs))]

# iterate through conditions and get all gene names
spots_dict = {}
large_intens = {}
genes = set()
for c, in_dir in zip(conditions, in_dirs):
    for f in os.listdir(in_dir):
        if f.startswith('intens_merged') and f.endswith('.csv'):
            gene = f.split('^')[1].split('.')[0]
            genes.add(gene)

# create empty entries
for gene in sorted(list(genes)):
    spots_dict[gene] = {c:{'cyt':[], 'peri':[]} for c in conditions}
    large_intens[gene] = {c:{'cyt':[], 'peri':[]} for c in conditions}

# iterate over file data
for c, in_dir in zip(conditions, in_dirs):
    files = [f for f in os.listdir(in_dir) if f.startswith('intens_merged') and f.endswith('.csv')]

    for f in files:
        print(f)
        gene = f.split('^')[1].split('.')[0]
        if '_cyt^' in f:
            category = 'cyt'
        elif '_peri^' in f:
            category = 'peri'

        # open spots file
        with open(os.path.join(in_dir, f), 'r') as infile:
            reader = csv.reader(infile)
            lines = [row for row in reader]

        intensities = [float(val) for val in lines[0]]
        spots_dict[gene][c][category] = deepcopy(intensities)
        large_intens[gene][c][category] = np.percentile(intensities, 95)

# make CDF plots
print('\nDrawing CDF plots...')
for g in sorted(list(genes)):
    fig, ax = plt.subplots()
    x_C = sorted(spots_dict[g]['C']['peri'])
    y_C = [float(i)/len(x_C) for i in range(len(x_C))]
    x_N = sorted(spots_dict[g]['N']['peri'])
    y_N = [float(i)/len(x_N) for i in range(len(x_N))]

    if x_C == [] or x_N == []:
        continue

    ax.step(x_C, y_C, 'k')
    ax.step(x_N, y_N, 'r')
    ax.set_xscale('log')

    # ax.hist(spots_dict[g]['C']['peri'], normed=True, cumulative=True, label='control', histtype='step', color='#aaaaaa')
    # ax.hist(spots_dict[g]['N']['peri'], normed=True, cumulative=True, label='control', histtype='step', color='r')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Spot intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cdf_intens^' + g + '.pdf'), dpi=300)
    plt.close()

# make swarm plots
print('Drawing strip plots...')
for g in sorted(list(genes)):
    # print(g)
    data_list = []
    for cat in ['cyt', 'peri']:
        for c in conditions:
            for intens in spots_dict[g][c][cat]:
                data_list.append([c+'_'+cat, intens])
    
    df = pd.DataFrame(data_list, columns=['compartment', 'intensity'])

    # make strip plots
    ax = sns.stripplot(x='compartment', y='intensity', data=df, size=2, jitter=0.4, color='#aaaaaa')
    ax.set_ylabel('Spot intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_strip_intens^' + g + '.pdf'), dpi=300)
    plt.close()

    fig, log_ax = plt.subplots()
    log_ax.set_yscale('log')  # log first
    log_ax.set_ylabel('Spot intensity')
    sns.boxplot(x='compartment', y='intensity', data=df, ax=log_ax, color='#bbbbbb', whis=np.inf, notch=True)
    sns.stripplot(x='compartment', y='intensity', data=df, ax=log_ax, size=2, jitter=0.3, color='#5a5a5a')
    
    x = 0
    for i, cat in enumerate(['cyt', 'peri']):
        for j, c in enumerate(conditions):
            y = large_intens[g][c][cat]
            if y:
                log_ax.plot([x-0.4, x+0.4], [y, y], 'r-')
                x += 1

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_strip_log_intens^' + g + '.pdf'), dpi=300)
    plt.close()

# make CCDF plots
print('Drawing tail distribution plots (with bootstrapping)...')
for g in sorted(list(genes)):
    if g in ['ACTB', 'ACTN2', 'ATP2A1', 'CHRNE']:
        continue
    
    print('\n'+g)
    fig, ax = plt.subplots()
    x_C = sorted(spots_dict[g]['C']['peri'])
    y_C = [1.-(float(i)/len(x_C)) for i in range(len(x_C))]
    x_N = sorted(spots_dict[g]['N']['peri'])
    y_N = [1.-(float(i)/len(x_N)) for i in range(len(x_N))]

    if x_C == [] or x_N == []:
        continue

    # bootstrap
    bs_x_C = []
    bs_y_C = []
    bs_x_N = []
    bs_y_N = []

    n_samples = 100
    
    print('Bootstrapping...')
    for n in range(n_samples):
        sample_C = sorted(np.random.choice(spots_dict[g]['C']['peri'], size=len(spots_dict[g]['C']['peri'])))
        sample_N = sorted(np.random.choice(spots_dict[g]['N']['peri'], size=len(spots_dict[g]['N']['peri'])))
        bs_x_C.append(sample_C)
        bs_y_C.append([1.-(float(i)/len(sample_C)) for i in range(len(sample_C))])
        bs_x_N.append(sample_N)
        bs_y_N.append([1.-(float(i)/len(sample_N)) for i in range(len(sample_N))])
    
    print('Calculating CIs...')
    y_mins_C, y_maxs_C = calculate_ci(x_C, bs_x_C, ci=0.95)
    y_mins_N, y_maxs_N = calculate_ci(x_N, bs_x_N, ci=0.95)

    # plot
    print('Drawing plot...')
    ax.step(x_C, y_C, 'k')
    ax.step(x_N, y_N, 'r')
    ax.fill_between(x_C, y_mins_C, y_maxs_C, step='pre', color='k', alpha=0.3, linewidth=0)
    ax.fill_between(x_N, y_mins_N, y_maxs_N, step='pre', color='r', alpha=0.3, linewidth=0)


    # for n in range(n_samples):
    #     ax.step(bs_x_C[n], bs_y_C[n], 'k', alpha=0.2)
    #     ax.step(bs_x_N[n], bs_y_N[n], 'r', alpha=0.2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # ax.hist(spots_dict[g]['C']['peri'], normed=True, cumulative=True, label='control', histtype='step', color='#aaaaaa')
    # ax.hist(spots_dict[g]['N']['peri'], normed=True, cumulative=True, label='control', histtype='step', color='r')
    ax.set_ylabel('Tail distribution')
    ax.set_xlabel('Spot intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'taildist_intens^' + g + '.pdf'), dpi=300)
    plt.close()