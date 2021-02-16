from collections import defaultdict
import argparse
import csv
import os
import scipy.stats as ss

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, nargs=1, help='Path to directory containing spot distance files.')

# parse arguments
args = vars(parser.parse_args())
dist_dir = args['dir'][0]

# open each file and organize by gene
true_dists = defaultdict(list)
rand_dists = defaultdict(list)

files = [f for f in os.listdir(dist_dir) if f.endswith('.txt')]

genes = set()
for f in files:
    print(f)

    # get gene
    prefix = os.path.splitext(f)[0]
    gene = prefix.split('^')[-1]

    # open data and add to correct dictionary
    with open(os.path.join(dist_dir, f), 'r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        if '_expt_' in f:
            true_dists[gene].extend([float(row[0]) for row in reader])
        elif '_rand_' in f:
            rand_dists[gene].extend([float(row[0]) for row in reader])
        else:
            print(f + ' not recognized.')
            continue
    
    genes.add(gene)

# for each gene, output combined distances for experiment and randomized
for g in list(genes):
    with open('dists_tjunc_' + g + '_experiment.txt', 'w') as outfile:
        outfile.writelines([str(d)+'\n' for d in true_dists[g]])
    with open('dists_tjunc_' + g + '_randomized.txt', 'w') as outfile:
        outfile.writelines([str(d)+'\n' for d in rand_dists[g]])

