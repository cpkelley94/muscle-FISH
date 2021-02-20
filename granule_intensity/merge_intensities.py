# python 3.6.5
"""Combine spot intensity data across all images from the same gene.
"""
from collections import defaultdict
import argparse
import csv
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
    if '_intens' in p and p.endswith('.csv'):
        gene = p.split('^')[1].split('.')[0]
        print(gene)
        genes.add(gene)

        with open(os.path.join(intens_dir, p), 'r') as infile:
            reader = csv.reader(infile)
            data_list = [row for row in reader]
            if p.split('^')[0].endswith('cyt'):
                cyt_spots_by_gene[gene].extend(data_list[0])
            elif p.split('^')[0].endswith('peri'):
                peri_spots_by_gene[gene].extend(data_list[0])
            else:
                print('warning: compartment not recognized for file ' + p)

for g in list(genes):
    with open(os.path.join(out_dir, 'intens_merged_cyt^' + g + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([cyt_spots_by_gene[g]])
    
    with open(os.path.join(out_dir, 'intens_merged_peri^' + g + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([peri_spots_by_gene[g]])


