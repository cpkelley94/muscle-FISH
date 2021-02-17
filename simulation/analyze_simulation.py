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
parser.add_argument('-b', '--burnin', type=float, help='Time before which data should be discarded (hr).', default=0)

# parse arguments
args = vars(parser.parse_args())
img_name = args['image_name'][0]
indir = args['indir'][0]
gene = args['gene'][0]
outdir = args['outdir'][0]
burnin = args['burnin']

FRAME_TO_DRAW = 300000

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

# generate mask for all allowed positions
mask_nuc_eroded = morphology.binary_erosion(mask_nuc.astype(bool))
mask_allowed = np.logical_and(mask_fiber.astype(bool), np.logical_not(mask_nuc_eroded).astype(bool))
labeled_mask_allowed, n_regions = morphology.label(mask_allowed, return_num=True)
voxel_counts = []
for l in range(1, n_regions+1):
    voxel_counts.append(np.count_nonzero(labeled_mask_allowed == l))
mask_allowed = (labeled_mask_allowed == np.argmax(voxel_counts) + 1)

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

print('Retrieving information about simulation...')

# with open(p_state, 'r') as infile:
#     for line in infile:
#         pass
#     last_line = line

with open(p_state, 'rb') as f:
    f.seek(-2, os.SEEK_END)
    while f.read(1) != b'\n':
        f.seek(-2, os.SEEK_CUR)
    last_line = f.readline().decode()

n_rnas = last_line.count(',')
t_final = float(last_line[:last_line.find(',')])  # s

# get list of timepoints that fall approx. every hour after burnin
first_hr = int(burnin + 1)
last_hr = int(t_final/3600.)
timepoints = list(3600*np.array(range(first_hr, last_hr+1)))  # must be sorted in ascending order
print(str(len(timepoints)) + ' hourly timepoints to analyze.')

# make 2d histogram of average RNA position in fiber
fiber_2d = np.amax(mask_fiber.astype(int), axis=2)
nuc_2d = np.amax(mask_nuc.astype(int), axis=2)
x_max, y_max = fiber_2d.shape
grassfire = distance_transform_edt(np.logical_not(mask_nuc), sampling=dims_xyz)

# load data at timepoints
print('Loading simulation position and state data...')

hists = np.zeros((100, 100))
dist_hists = np.zeros(200)
n_t = 0
line_num = 0
with open(p_x, 'r') as xf, open(p_y, 'r') as yf, open(p_z, 'r') as zf, open(p_state, 'r') as sf:
    for i, lines in enumerate(zip(xf, yf, zf, sf)):
        xl, yl, zl, sl = lines

        if len(timepoints) == 0:
            break

        t = float(xl[:xl.find(',')])
        if timepoints[0] < t:
            rowx = xl[:-1].split(',')
            rowy = yl[:-1].split(',')
            rowz = zl[:-1].split(',')
            rows = sl[:-1].split(',')

            num_to_extend = n_rnas + 1 - len(rows)
            x = np.array([float(r) for r in rowx[1:]] + ([np.nan]*num_to_extend))
            y = np.array([float(r) for r in rowy[1:]] + ([np.nan]*num_to_extend))
            z = np.array([float(r) for r in rowz[1:]] + ([np.nan]*num_to_extend))
            s = np.array([round(float(r)) for r in rows[1:]] + ([-2]*num_to_extend))

            x[s == -1] = np.nan
            y[s == -1] = np.nan
            z[s == -1] = np.nan

            grid_x = list(range(grassfire.shape[0]))
            grid_y = list(range(grassfire.shape[1]))
            grid_z = list(range(grassfire.shape[2]))
            dists = interpn((grid_x, grid_y, grid_z), grassfire, np.column_stack((x, y, z)), bounds_error=False, fill_value=np.nan)

            hist = np.histogram2d(x, y, bins=100, range=[[0, x_max], [0, y_max]], density=True)[0]
            hists += hist

            dist_hist, edges = np.histogram(dists, bins=200, range=(0, 25), density=True)
            dist_hists += dist_hist

            n_t += 1
            timepoints.pop(0)
            # print(t)

        # # store ultimate and penultimate lines
        # if i == FRAME_TO_DRAW - 1:
        #     xl_pen = xl
        #     yl_pen = yl
        #     sl_pen = zl
        # elif i == FRAME_TO_DRAW:
        #     xl_ult = xl
        #     yl_ult = yl
        #     sl_ult = sl

mean_hist = hists/float(n_t)
mean_dist_hist = dist_hists/float(n_t)
print(np.amax(mean_hist))

# randomize RNA positions to get null distribution
print('Randomizing RNA positions for null distribution...')
allowed_positions = list(np.column_stack(np.nonzero(mask_allowed)))
random_spots = np.row_stack(random.choices(allowed_positions, k=188*n_t))  # with replacement
rand_dists = interpn((grid_x, grid_y, grid_z), grassfire, random_spots, bounds_error=False, fill_value=np.nan)
rand_dist_hist, rand_edges = np.histogram(rand_dists, bins=200, range=(0, 25), density=True)
assert np.array_equal(edges, rand_edges)

with open(os.path.join(outdir, img_name + '_disthist_data^' + gene + '.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(zip([(edges[i] + edges[i+1])/2. for i in range(len(edges)-1)], mean_dist_hist, rand_dist_hist))

print('Drawing plots...')
fig, ax = plt.subplots()
ax.imshow(mean_hist, cmap='binary', interpolation='bicubic', vmax=1.4607992354521603e-05)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
# ax.set_xlim([0, fiber_2d.shape[1]])
# ax.set_ylim([0, fiber_2d.shape[0]])
plt.savefig(os.path.join(outdir, img_name + '_hist2d^' + gene + '.pdf'), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot([(edges[i] + edges[i+1])/2. for i in range(len(edges)-1)], mean_dist_hist, 'k-')
ax.plot([(edges[i] + edges[i+1])/2. for i in range(len(edges)-1)], rand_dist_hist, 'g-')
ax.set_xlim([0, 25])
ax.set_ylim([0, 0.55])
ax.set_xlabel('Distance to nearest nucleus (um)')
ax.set_ylabel('Frequency')
plt.savefig(os.path.join(outdir, img_name + '_disthist^' + gene + '.pdf'), dpi=300)
plt.close()

# # draw final frame
# rowx_pen = xl_pen[:-1].split(',')
# rowy_pen = yl_pen[:-1].split(',')
# rows_pen = sl_pen[:-1].split(',')
# rowx_ult = xl_ult[:-1].split(',')
# rowy_ult = yl_ult[:-1].split(',')
# rows_ult = sl_ult[:-1].split(',')
# x_pen = np.array([float(r) for r in rowx_pen[1:]])
# y_pen = np.array([float(r) for r in rowy_pen[1:]])
# s_pen = np.array([round(float(r)) for r in rows_pen[1:]])
# x_pen[s_pen == -1] = np.nan
# y_pen[s_pen == -1] = np.nan
# x_ult = np.array([float(r) for r in rowx_ult[1:]])
# y_ult = np.array([float(r) for r in rowy_ult[1:]])
# s_ult = np.array([round(float(r)) for r in rows_ult[1:]])
# x_ult[s_ult == -1] = np.nan
# y_ult[s_ult == -1] = np.nan

# spot_cmap = {-2:'k', -1:'#333333', 0:'#008ffd', 1:'#75e900', 2:'#eede0a', 3:'#f075f5'}
# fig, ax = plt.subplots()
# ax.imshow(fiber_2d, vmax=10, cmap='binary_r')
# ax.imshow(nuc_2d, cmap=su.cmap_NtoW)
# scatter = ax.scatter(y_ult, x_ult, c=[spot_cmap[si] for si in s_ult], s=3)
# transport_events = []
# for i, state in enumerate(s_ult):
#     if state == 2 or state == 3:
#         # draw directed transport as line segment
#         x1 = x_pen[i]
#         x2 = x_ult[i]
#         y1 = y_pen[i]
#         y2 = y_ult[i]
#         event = ax.plot([y1, y2], [x1, x2], ls='-', lw=3, c=spot_cmap[state], solid_capstyle='round', zorder=1000)
#         transport_events.append(event)

# ax.set_xlim([0, fiber_2d.shape[1]])
# ax.set_ylim([0, fiber_2d.shape[0]])
# plt.savefig(os.path.join(outdir, img_name + '_frame^' + gene + '.pdf'), dpi=300)
# plt.close()