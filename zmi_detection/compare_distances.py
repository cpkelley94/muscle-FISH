"""Compare distance-to-ZMI distributions for all genes.
"""
import matplotlib
matplotlib.use('Agg')  # for plotting on cluster

# figure params
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['font.sans-serif'] = ['Arial']

from collections import defaultdict
from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os
import scipy.stats as ss

def fdr(p_vals):
    """Correct p-values using Benjamini-Hochberg false discovery rate (FDR).
    """

    ranked_p_values = ss.rankdata(p_vals)
    q = p_vals * float(len(p_vals)) / ranked_p_values
    q[q > 1] = 1

    return q

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, nargs=1, help='Directory containing pooled distances by gene.')
parser.add_argument('genes', type=str, nargs='*', help='Names of genes to compare.')

# parse arguments
args = vars(parser.parse_args())
dists_dir = args['dir'][0]
genes = args['genes']

# get data
expt_dists = {}
rand_dists = {}
for gene in genes:
    expt_path = os.path.join(dists_dir, 'dists_zmi_' + gene + '_experiment.txt')
    with open(expt_path, 'r') as expt_file:
        reader = csv.reader(expt_file, delimiter='\t')
        expt_dists[gene] = [float(row[0]) for row in reader]

    rand_path = os.path.join(dists_dir, 'dists_zmi_' + gene + '_randomized.txt')
    with open(rand_path, 'r') as rand_file:
        reader = csv.reader(rand_file, delimiter='\t')
        rand_dists[gene] = [float(row[0]) for row in reader]

# make histograms for each gene
header = ['gene', 'num_spots', 'log10FC', 'U', 'p', 'FDRq']
stats_table = []
p_values = []
for gene in genes:
    max_dist = max(max(expt_dists[gene]), max(rand_dists[gene]))

    fig, ax = plt.subplots()
    ax.hist(expt_dists[gene], bins=1000, range=(0,max_dist), density=True, histtype='step', cumulative=True, color='r')
    ax.hist(rand_dists[gene], bins=1000, range=(0,max_dist), density=True, histtype='step', cumulative=True, color='k')
    ax.set_xlabel('Distance to ZMI (um)')
    ax.set_ylabel('Cumulative fraction')
    plt.tight_layout()
    plt.savefig('dist_hist_cdf_pooled_replicates^' + gene + '.png', dpi=300)
    plt.savefig('dist_hist_cdf_pooled_replicates^' + gene + '.pdf', dpi=300)
    plt.close()

    U, p = ss.mannwhitneyu(expt_dists[gene], rand_dists[gene], alternative='less')
    logfc = np.log10(np.median(expt_dists[gene])/np.median(rand_dists[gene]))
    stats_table.append([gene, len(expt_dists[gene]), logfc, U, p])
    p_values.append(p)

# FDR correction
p_np = np.array(p_values)
q_vals = fdr(p_np)

# output statistics for each gene
for i, row in enumerate(stats_table):
    row.append(q_vals[i])

with open('zmi_dist_statistics_by_gene.txt', 'w') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows([header] + stats_table)

# make volcano plot including all genes
log_fcs = [r[2] for r in stats_table]
nlq = -1.*np.log10(q_vals)
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(log_fcs, nlq, 'k.')
for i, g in enumerate(genes):
	ax.annotate(g, (log_fcs[i], nlq[i]))
ax.plot([0,0], [-1, 11], '--', c='#bbbbbb', lw=0.8)
ax.plot([-0.2,0.2], [2, 2], '--', c='#bbbbbb', lw=0.8)
ax.set_xlabel('log10(fold change)')
ax.set_ylabel('-log10(q)')
ax.set_xlim([-0.15, 0.15])
ax.set_ylim([0, 10])
plt.tight_layout()
plt.savefig('zmi_dist_volcano.png', dpi=300)
plt.savefig('zmi_dist_volcano.pdf', dpi=300)
plt.close()
