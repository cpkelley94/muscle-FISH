import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology
from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os
import pandas as pd
import random
import tifffile

# custom libraries
import scope_utils3 as su


# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_name', type=str, nargs=1, help='Name of image.')
parser.add_argument('indir', type=str, nargs=1, help='Directory of input files for simulation.')
parser.add_argument('gene', type=str, nargs=1, help='Gene name.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory of output files.')
parser.add_argument('-t', '--time', type=float, help='Time at which to draw the frame (hr).', default=0)
parser.add_argument('--color', action='store_true', help='Color spots by state.')

# parse arguments
args = vars(parser.parse_args())
img_name = args['image_name'][0]
indir = args['indir'][0]
gene = args['gene'][0]
outdir = args['outdir'][0]
time_to_draw = float(args['time'])*3600. - 1.
color_spots = args['color']

p_fiber = os.path.join(indir, img_name + '_fiber.npy')
p_nuc = os.path.join(indir, img_name + '_nuclei.npy')
p_dims = os.path.join(indir, img_name + '_dims.csv')

p_x = os.path.join(outdir, img_name+'_paths_x^'+gene+'.csv')
p_y = os.path.join(outdir, img_name+'_paths_y^'+gene+'.csv')
p_z = os.path.join(outdir, img_name+'_paths_z^'+gene+'.csv')
p_state = os.path.join(outdir, img_name+'_paths_state^'+gene+'.csv')

# open fiber and nuclei masks
mask_fiber = np.load(p_fiber)
mask_nuc = np.load(p_nuc)
tifffile.imwrite(os.path.join(outdir, img_name+'_fiber.tiff'), data=mask_fiber.transpose(2,0,1).astype(np.uint8)*255, compress=6, photometric='minisblack')
tifffile.imwrite(os.path.join(outdir, img_name+'_nuc.tiff'), data=mask_nuc.transpose(2,0,1).astype(np.uint8)*255, compress=6, photometric='minisblack')

# get image dimensions
with open(p_dims, 'r') as infile:
    reader = csv.reader(infile)
    dims = {row[0]:float(row[1]) for row in reader}
print(dims)
dims_xyz = np.array([dims['x'], dims['y'], dims['z']])

# get last line of one of the files
# to determine number of simulated RNAs and final time
rna_dict = {}
files = [p_x, p_y, p_z, p_state]

# make 2d histogram of average RNA position in fiber
fiber_2d = np.amax(mask_fiber.astype(int), axis=2)
nuc_2d = np.amax(mask_nuc.astype(int), axis=2)
x_max, y_max = fiber_2d.shape

# load data at timepoints
print('Loading frame data...')
with open(p_x, 'r') as xf, open(p_y, 'r') as yf, open(p_z, 'r') as zf, open(p_state, 'r') as sf:
    for i, lines in enumerate(zip(xf, yf, zf, sf)):
        xl, yl, zl, sl = lines

        t = float(xl[:xl.find(',')])
        if time_to_draw < t:
            print('capturing frame at ' + str(t) + ' s')
            rowx = xl[:-1].split(',')
            rowy = yl[:-1].split(',')
            rowz = zl[:-1].split(',')
            rows = sl[:-1].split(',')

            x = np.array([float(r) for r in rowx[1:]])
            y = np.array([float(r) for r in rowy[1:]])
            z = np.array([float(r) for r in rowz[1:]])
            s = np.array([round(float(r)) for r in rows[1:]])

            x[s == -1] = np.nan
            y[s == -1] = np.nan
            z[s == -1] = np.nan

            break

# draw final frame
print('Drawing frame...')
if color_spots:
    spot_cmap = {-2:'k', -1:'#333333', 0:'#008ffd', 1:'#75e900', 2:'#eede0a', 3:'#f075f5'}
else:
    spot_cmap = {i:'#ff3333' for i in range(-2, 4)}
fig, ax = plt.subplots()
ax.imshow(fiber_2d, vmax=10, cmap='binary_r')
ax.imshow(nuc_2d, cmap=su.cmap_NtoW)
scatter = ax.scatter(y, x, c=[spot_cmap[si] for si in s], s=3)
ax.set_xlim([0, fiber_2d.shape[1]])
ax.set_ylim([0, fiber_2d.shape[0]])
plt.savefig(os.path.join(outdir, img_name + '_frame' + str(time_to_draw) + '^' + gene + '.pdf'), dpi=300)
plt.close()