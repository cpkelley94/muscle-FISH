# python 2.7 (!)
"""
Estimate the degradation rate of mRNAs from each gene from time-course of FISH 
spot detection data. Use non-linear least squares regression to fit an 
exponential decay curve to spot density measurements.
"""

# plot formatting
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 6
rcParams['font.sans-serif'] = ['Arial']

from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, linregress
import argparse
import csv
import numpy as np

def expo(x, A, b):
    return A*np.exp(-1.*b*x)  # fit function for scipy.optimize.curve_fit()

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('tables', type=str, nargs='*',
    help='Paths to the cytoplasmic RNA density tables.')
parser.add_argument('-t', '--times', type=int, nargs='*',
    help='Time points for each density table.', default=None)
parser.add_argument('-n', '--noco', help='Path to nocodazole depletion .CSV file.', default=None)
parser.add_argument('-d', '--densities', help='Path to cytoplasmic densities from nocodazole experiment.', default=None)

# parse arguments
args = vars(parser.parse_args())
p_tables = args['tables']
times = args['times']
p_noco = args['noco']
p_dens = args['densities']

if times is None:
    raise ValueError('No time points provided. Use \'t\' to provide a list of time points.')

gene_names = ['Hnrnpa2b1', 'Atp2a1', 'Polr2a', 'Myom1',
    'Gapdh', 'Hist1h1c', 'Vcl', 'Chrne1',
    'Ttn', 'Dmd', 'Actn2']

# open the tables
data_by_time = []
for j, p in enumerate(p_tables):
    with open(p, 'r') as table_file:
        reader = csv.reader(table_file)
        data = {gene_names[int(row[0])-1]:np.array(row[1:]).astype(float) for i, row in enumerate(reader) if i}
        data_by_time.append(data)

# reorganize
mean_by_gene = defaultdict(list)
err_by_gene = defaultdict(list)
genes = set()
for j in range(len(times)):
    for g in data_by_time[j].keys():
        genes.add(g)
        mean_by_gene[g].append(np.mean(data_by_time[j][g]))
        err_by_gene[g].append(np.std(data_by_time[j][g]))

gene_list = sorted(list(genes))

# curve_fit and make plots per gene
genes_to_draw = ['Polr2a', 'Vcl', 'Dmd', 'Hist1h1c', 
    'Hnrnpa2b1', 'Myom1', 'Ttn', 'Gapdh']
curves = []
curve_dict = {}
fig, ax = plt.subplots(2, 4)
fig.set_size_inches(8, 4.5)
for i, g in enumerate(genes_to_draw):
    exp_params, exp_cov = curve_fit(expo, times, mean_by_gene[g], p0=[mean_by_gene[g][0], -np.log(2)/6.])
    curves.append([g, genes_to_draw[i]] + list(exp_params) + list(np.sqrt(np.diag(exp_cov))) + [np.log(2)/exp_params[1]])
    curve_dict.update({genes_to_draw[i]:exp_params})
    x_fit = np.linspace(min(times)-3, max(times)+3, num=250)
    y_fit = expo(x_fit, *exp_params)

    ij = (int(i/4), i%4)

    ax[ij].errorbar(times, mean_by_gene[g], fmt='ko', yerr=err_by_gene[g], elinewidth=0.8, capsize=5)
    ax[ij].plot(x_fit, y_fit, 'k-', lw=0.8)
    ax[ij].set_xlim([min(times)-2, max(times)+2])
    ax[ij].set_ylim(bottom=0)
    ax[ij].set_title(g)
    ax[ij].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # hide the right and top spines
    ax[ij].spines['right'].set_visible(False)
    ax[ij].spines['top'].set_visible(False)

    # only show ticks on the left and bottom spines
    ax[ij].yaxis.set_ticks_position('left')
    ax[ij].xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.savefig('rna_decay_by_gene.pdf', dpi=300)
plt.close()

# output curves
with open('exponential_decay_fit.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows([['id', 'gene', 'A (molec um^-3)', 'k (hr^-1)', 'dA (molec um^-3)', 'dk (hr^-1)', 'half-life (hr)']] + curves)

# compare to nocodazole depletion data
if p_noco is not None:
    with open(p_noco, 'r') as noco_file:
        reader = csv.reader(noco_file)
        noco_data = {row[0].capitalize():float(row[1]) for row in reader}
    
    half_lives = []
    fracs_theo = []
    fracs_expt = []
    genes_in_both = []

    for name in gene_names:
        if name in noco_data and name != 'Actn2':
            frac_theo = expo(18., 1., curve_dict[name][1])
            half_lives.append(np.log(2)/curve_dict[name][1])
            fracs_theo.append(frac_theo)
            fracs_expt.append(noco_data[name])
            genes_in_both.append(name)
    
    # plot depletion vs. half-life
    fig, ax = plt.subplots()
    fig.set_size_inches(4,4)
    ax.plot(half_lives, fracs_expt, 'k.')
    ax.set_ylim([0,1])
    ax.set_xlabel('Half-life (hr)')
    ax.set_ylabel('Fraction of mRNAs 5+ um away from nucleus\nremaining after nocodazole treatment')
    plt.tight_layout()
    plt.savefig('depletion_vs_half_life.png', dpi=300)
    plt.savefig('depletion_vs_half_life.pdf', dpi=300)
    plt.close()

    # plot experimental depletion vs. predicted
    slope, intercept, r_val, p_val, sem = linregress(fracs_theo, fracs_expt)
    x_lin = np.linspace(0, 1, num=2, endpoint=True)
    y_lin = slope*x_lin + intercept

    print slope, intercept, r_val, p_val, sem

    fig, ax = plt.subplots()
    fig.set_size_inches(2,2)
    ax.plot(fracs_theo, fracs_expt, 'k.')
    for i, g in enumerate(genes_in_both):
        ax.annotate(g, (fracs_theo[i], fracs_expt[i]))
    ax.plot([0,1], [0,1], '--', c='#bbbbbb', lw=0.8, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('Predicted fraction remaining')
    ax.set_ylabel('Observed fraction remaining')
    plt.tight_layout()
    plt.savefig('depletion_vs_predicted.png', dpi=300)
    plt.savefig('depletion_vs_predicted.pdf', dpi=300)
    plt.close()

    # statistics
    corr, p_val = pearsonr(fracs_theo, fracs_expt)
    print corr

    out_arr = np.column_stack((genes_in_both, half_lives, fracs_theo, fracs_expt))
    out_arr = np.insert(out_arr, 0, ['gene', 'half_life (hr)', 'predicted rem.', 'experiment rem.'], axis=0)

    with open('depletion.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(out_arr)

if p_dens is not None:
    with open(p_dens, 'r') as dens_file:
        reader = csv.reader(dens_file)
        dens_data = {row[0].capitalize():[float(j) for j in row[1:]] for row in list(reader)[1:]}

    print dens_data

    dens_theo = []
    dens_expt = []
    genes_in_both = []

    for name in gene_names:
        if name in noco_data and name != 'Actn2':
            d_theo = expo(18., 1., curve_dict[name][1])
            dens_theo.append(d_theo)
            dens_expt.append(dens_data[name][2]/dens_data[name][0])
            genes_in_both.append(name)

    # plot experimental depletion vs. predicted
    slope, intercept, r_val, p_val, sem = linregress(dens_theo, dens_expt)
    x_lin = np.linspace(0, 1, num=2, endpoint=True)
    y_lin = slope*x_lin + intercept

    print slope, intercept, r_val, p_val, sem

    fig, ax = plt.subplots()
    fig.set_size_inches(4,4)
    ax.plot(dens_theo, dens_expt, 'k.')
    for i, g in enumerate(genes_in_both):
        ax.annotate(g, (dens_theo[i], dens_expt[i]))
    ax.plot([0,1], [0,1], '--', c='#bbbbbb', lw=0.8, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('Predicted cytoplasmic fraction remaining')
    ax.set_ylabel('Observed cytoplasmic fraction remaining')
    plt.tight_layout()
    plt.savefig('cyto_frac_vs_predicted.png', dpi=300)
    plt.savefig('cyto_frac_vs_predicted.pdf', dpi=300)
    plt.close()

    # statistics
    corr, p_val = pearsonr(dens_theo, dens_expt)
    print corr