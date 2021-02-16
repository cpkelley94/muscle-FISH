# python 2.7.14

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('gene', type=str, nargs=1, help='Name of the gene to analyze.')

# parse arguments
args = vars(parser.parse_args())
gene_name = args['gene'][0]

# open all the counts data for this gene
data = []
paths = [f for f in os.listdir(os.getcwd()) if f.startswith(gene_name) and f.endswith('counts.csv')]
for p in paths:
    with open(p, 'r') as infile:
        reader = csv.reader(infile)
        data.append({row[0]:np.array(row[1:]).astype(float) for row in list(reader)[1:]})

# organize data into new dictionaries
regions = ['nuc', 'peri', 'cyt']
densities = {key:[] for key in regions}
total_count = []
total_vol = []

for counts_data in data:
    for reg in regions:
        densities[reg].append(counts_data[reg][2])
    total_count.append(np.sum([counts_data[reg][1] for reg in regions]))
    total_vol.append(np.sum([counts_data[reg][0] for reg in regions]))

# calculate enrichment of mRNA density in each compartment over average
total_dens = np.divide(total_count, total_vol)

enrichment = {}
for reg in regions:
    enrichment.update({reg:np.divide(densities[reg], total_dens)})

enrich_mean = {reg:np.mean(enrichment[reg]) for reg in regions}
enrich_std = {reg:np.std(enrichment[reg]) for reg in regions}

# plot
means = [enrich_mean[reg] for reg in regions]
errs = [enrich_std[reg] for reg in regions]
colors = ['#4d4dff', '#4dff4d', '#ff4d4d']
indices = [0, 1, 2]
fig, ax = plt.subplots()
fig.set_size_inches(2, 4)
ax.bar(indices, [enrich_mean[reg] for reg in regions], color=colors, edgecolor='k', yerr=errs, capsize=3, linewidth=0.8, error_kw={'elinewidth':0.8})
ax.hlines(1., -1, 4, linestyle='dashed', lw=0.8, color='k')
ax.set_title(gene_name)
ax.set_xlim([-0.69, 2.69])
ax.set_ylim([1./20., 20.])
ax.set_xticks(indices)
ax.set_xticklabels(['N', 'P', 'C'])
ax.set_ylabel('Enrichment of transcript localization')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(gene_name + '_enrichment.png', dpi=300)
# plt.show()
