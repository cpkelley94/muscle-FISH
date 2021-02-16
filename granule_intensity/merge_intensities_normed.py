from collections import defaultdict
import argparse
import csv
import numpy as np
import os

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('infolder', type=str, nargs=1, help='Path to folder containing 3D Gaussian curve fit data.')
parser.add_argument('outfolder', type=str, nargs=1, help='Path to output directory.')

# parse arguments
args = vars(parser.parse_args())
intens_dir = args['infolder'][0]
out_dir = args['outfolder'][0]

cyt_spots_by_gene = defaultdict(list)
peri_spots_by_gene = defaultdict(list)

genes = set()
for p in os.listdir(intens_dir):
    if '_intens_cyt' in p and p.endswith('.csv'):
        gene = p.split('^')[1].split('.')[0]
        print(gene)
        genes.add(gene)

        # get cytoplasmic intensities
        with open(os.path.join(intens_dir, p), 'r') as infile:
            reader = csv.reader(infile)
            intensities_cyt = [float(x) for x in [row for row in reader][0]]
            norm_factor = np.median(intensities_cyt)
            intens_cyt_normed = [x/norm_factor for x in intensities_cyt]

            cyt_spots_by_gene[gene].extend(intens_cyt_normed)

        # get perinuclear intensities
        p_peri = p.replace('_cyt^', '_peri^')
        with open(os.path.join(intens_dir, p_peri), 'r') as infile:
            reader = csv.reader(infile)
            intensities_peri = [float(x) for x in [row for row in reader][0]]
            intens_peri_normed = [x/norm_factor for x in intensities_peri]

            peri_spots_by_gene[gene].extend(intens_peri_normed)


for g in list(genes):
    with open(os.path.join(out_dir, 'intens_merged_cyt^' + g + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([cyt_spots_by_gene[g]])
    
    with open(os.path.join(out_dir, 'intens_merged_peri^' + g + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([peri_spots_by_gene[g]])


