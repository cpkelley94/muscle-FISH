from collections import defaultdict
from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os
import pandas as pd
import seaborn as sns

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('infolder', type=str, nargs=1, help='Path to folder containing merged 3D Gaussian curve fit data.')
parser.add_argument('outfolder', type=str, nargs=1, help='Path to output directory.')

# parse arguments
args = vars(parser.parse_args())
in_dir = args['infolder'][0]
out_dir = args['outfolder'][0]

files = [f for f in os.listdir(in_dir) if f.startswith('intens_merged') and f.endswith('.csv')]
spots_dict = defaultdict(dict)
genes = set()

for f in files:
    print(f)
    gene = f.split('^')[1].split('.')[0]
    if '_cyt^' in f:
        outname = 'hist_intens_cyt^' + gene + '.pdf'
        category = 'cyt'
    elif '_peri^' in f:
        outname = 'hist_intens_peri^' + gene + '.pdf'
        category = 'peri'

    # open spots file
    with open(os.path.join(in_dir, f), 'r') as infile:
        reader = csv.reader(infile)
        lines = [row for row in reader]

    intensities = [float(val) for val in lines[0]]
    spots_dict[gene][category] = intensities
    genes.add(gene)

    # plot spot intensity histograms
    fig, ax = plt.subplots()
    ax.hist(intensities, bins=200, range=(0, 200))
    ax.set_xlim([0, 200])
    ax.set_xlabel('Spot intensity')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, outname), dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(intensities, bins=np.logspace(-0.5,4,100))
    ax.set_xscale('log')
    ax.set_xlabel('Spot intensity')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'log_' + outname), dpi=300)
    plt.close()

# make swarm plots
for g in sorted(list(genes)):
    print(g)
    data_list = []
    for cat in ['cyt', 'peri']:
        for intens in spots_dict[g][cat]:
            data_list.append([cat, intens])
    
    df = pd.DataFrame(data_list, columns=['compartment', 'intensity'])

    # make strip plots
    ax = sns.stripplot(x='compartment', y='intensity', data=df, size=2)
    ax.set_ylabel('Spot intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'strip_intens^' + g + '.pdf'), dpi=300)
    plt.close()

    fig, log_ax = plt.subplots()
    log_ax.set_yscale('log')  # log first
    log_ax.set_ylabel('Spot intensity')
    sns.stripplot(x='compartment', y='intensity', data=df, ax=log_ax, size=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'strip_log_intens^' + g + '.pdf'), dpi=300)
    plt.close()