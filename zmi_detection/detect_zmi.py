# python 3.6.5, HiPerGator

import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import correlate2d
from skimage import morphology, feature
import argparse
import math
import numpy as np
import os
import random
import scipy.stats as ss
import tifffile

def find_subarrays_crosscorr(arr, subarrs):
    """Search a n-dimensional numpy array for subarrays.
    """

    if any(s.ndim != arr.ndim for s in subarrs):
        raise TypeError('one or more subarrays does not match dimensions of array')
    
    # iterate over all array dimensions simuiltaneously
    positions = set()
    for i, s in enumerate(subarrs):
        print('\rSearch progress: ' + str(i+1) + '/' + str(len(subarrs)) + '...', end='')
        # max_corr = np.amax(correlate2d(s,s))
        # corr_mat = correlate2d(arr, s)
        # print(feature.match_template(s,s))
        max_corr = np.amax(feature.match_template(s,s))
        corr_mat = feature.match_template(arr, s)
        # print(corr_mat)
        pos = np.argwhere(np.isclose(corr_mat, max_corr))

        if len(pos):
            pos_offset = np.add(pos, [(np.array(s.shape)-1)/2]*len(pos))
            for p in pos_offset:
                positions.add(tuple(p))
    
    print('Done.')

    # collapse calls that are within 2 pixels of each other in all dimensions
    pos_list = list(positions)
    for i, pt1 in enumerate(pos_list):
        for pt2 in pos_list[i+1:]:
            if all([abs(pt1[d]-pt2[d]) <= 2 for d in range(len(pt1))]):
                positions.discard(pt2)
    
    return sorted(list(positions))

def plotpath(name, prefix, gene):
    return os.path.join(p_out, 'plots', prefix + '_' + name + '^' + gene + '.png')

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('zlines', type=str, nargs=1, help='Path to 3D .TIF image file containing the z-line mask.')
parser.add_argument('tubules', type=str, nargs=1, help='Path to 3D .TIF image file containing the microtubule mask.')
parser.add_argument('nuclei', type=str, nargs=1, help='Path to 3D .TIF image file containing the nuclei mask.')
parser.add_argument('spots', type=str, nargs=1, help='Path to the TXT file containing spot information.')
parser.add_argument('gene', type=str, nargs=1, help='Name of the gene being analyzed.')
parser.add_argument('-o', '--outdir', help='Path to the output directory.', default=os.getcwd())

# parse arguments
args = vars(parser.parse_args())
p_zlines = args['zlines'][0]
p_tubules = args['tubules'][0]
p_nuc = args['nuclei'][0]
p_spots = args['spots'][0]
gene = args['gene'][0]
p_out = args['outdir']

prefix = os.path.basename(p_zlines).split('_')[0]

print('Analyzing image ' + prefix + '...')

# open the files
img_zlines = tifffile.imread(p_zlines)
img_tubules = tifffile.imread(p_tubules)
img_nuc = tifffile.imread(p_nuc)
spots_2d = np.loadtxt(p_spots)[:,1:3]

# max project and skeletonize
maxp_zlines = np.where(np.max(img_zlines, axis=0) > 0, 1, 0)
maxp_zlines = morphology.binary_closing(maxp_zlines).astype(int)
maxp_tubules = np.where(np.max(img_tubules, axis=0) > 0, 1, 0)
maxp_nuc = np.where(np.max(img_nuc, axis=0) > 0, 1, 0)
skel_zlines = morphology.skeletonize(maxp_zlines)
skel_tubules = morphology.skeletonize(maxp_tubules)
merged_mask = skel_zlines + 2*skel_tubules
merged_mask[maxp_nuc > 0] = 0  # remove the nuclei from the mask

# distance transform on the zline and tubule skeletons
dim_xy = 5.7657348678044378e-2  # um
dt_zlines = distance_transform_edt(np.logical_not(skel_zlines), sampling=(dim_xy, dim_xy))
dt_tubules = distance_transform_edt(np.logical_not(skel_tubules), sampling=(dim_xy, dim_xy))
dt_skel = np.minimum(dt_zlines, dt_tubules)

# remove nuclear spots
spots_cyt_list = []
for spot in spots_2d:
    if not maxp_nuc[tuple(spot.astype(int))]:
        spots_cyt_list.append(spot)
spots_cyt = np.row_stack(spots_cyt_list)

dists_skel = []
for spot in spots_cyt:
    dists_skel.append(dt_skel[tuple(spot.astype(int))])

# histogram of distances to cytoskeletal elements
fig, ax = plt.subplots()
ax.hist(dists_skel, bins=100, linewidth=0.8, edgecolor='k')
ax.set_xlabel('Distance to cytoskeleton (um)')
ax.set_ylabel('Counts')
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_spot_histogram^' + gene + '.png'), dpi=300)
plt.close()

# get all spots close to cytoskeletal elements
DIST_THRESHOLD = 0.25
spots_skel = []
spots_nonskel = []
for i, spot in enumerate(spots_cyt):
    if dists_skel[i] <= DIST_THRESHOLD:
        spots_skel.append(spot)
    else:
        spots_nonskel.append(spot)

spots_skel = np.row_stack(spots_skel)
spots_nonskel = np.row_stack(spots_nonskel)

print(str(len(spots_skel)) + '/' + str(len(spots_cyt)) + ' spots are cytoskeletal.')

region_skel = np.where(dt_skel <= DIST_THRESHOLD, 1, 0)
region_skel[maxp_nuc > 0] = 0
reg_skel_ind = np.argwhere(region_skel)

# draw the skeletonized masks
fig, ax = plt.subplots(1,4)
fig.set_size_inches(18, 4)
ax[0].imshow(skel_zlines, cmap='binary_r')
ax[1].imshow(skel_tubules, cmap='binary_r')
ax[2].imshow(merged_mask, cmap='gnuplot2')
ax[2].plot(spots_skel[:,1], spots_skel[:,0], 'm.', ms=1.5, mew=0)
ax[2].plot(spots_nonskel[:,1], spots_nonskel[:,0], 'y.', ms=1.5, mew=0)
ax[3].imshow(region_skel, cmap='binary_r')
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_skeletons^' + gene + '.png'), dpi=600)
plt.close()

# define all t-junction cross archetypes
x_archetypes = [
    np.array([[0,1,0],[2,3,2],[0,1,0]]),
    np.array([[0,1,0],[2,3,2],[0,0,1]]),
    np.array([[0,1,0],[0,3,2],[2,0,1]]),
    np.array([[0,1,0],[2,3,2],[0,1,1]]),
    np.array([[0,1,0],[2,3,2],[2,0,1]]),
    np.array([[2,1,0],[2,3,2],[0,0,1]]),
    np.array([[1,0,2],[0,3,0],[2,0,1]]),
    np.array([[1,0,2],[2,3,0],[2,0,1]]),
    np.array([[1,0,2],[2,3,0],[0,0,1]]),
    np.array([[1,0,0,2],[0,1,2,0],[0,2,1,0],[2,0,0,1]]),
    np.array([[1,0,2,0],[0,1,2,0],[0,2,1,0],[2,0,0,1]]),
    np.array([[1,0,2,0],[0,1,2,0],[0,2,1,0],[0,2,0,1]]),
    np.array([[0,0,2,0],[1,1,2,0],[0,2,1,0],[0,2,0,1]]),
    np.array([[0,0,2,0],[1,1,2,0],[0,2,1,1],[0,2,0,0]]),
]

# # draw the archetypes
# fig, ax = plt.subplots(4,math.ceil(len(x_archetypes)/4))
# # fig.set_size_inches(14, 4)
# for i, arch in enumerate(x_archetypes):
#     ax[i//4, i%4].imshow(arch, vmax=3, interpolation='nearest', cmap='gnuplot2')
# plt.tight_layout()
# plt.savefig('cross_archetypes.png', dpi=600)
# plt.close()

all_crosses = []

# make all possible permutations of the archetype
for arch in x_archetypes:
    permutations = [arch]

    # rotations
    for i in range(3):
        x_rot = np.rot90(arch, i+1)
        if not any(np.array_equal(x_rot, z) for z in permutations):
            permutations.append(x_rot)
    
    # reflections
    for x in deepcopy(permutations):
        x_refl = np.flip(x, axis=0)
        if not any(np.array_equal(x_refl, z) for z in permutations):
            permutations.append(x_refl)
    
    # feature swaps
    for x in deepcopy(permutations):
        x[x == 2] = 4
        x[x == 1] = 2
        x[x == 4] = 1
        if not any(np.array_equal(x, z) for z in permutations):
            permutations.append(x)

    # add them to all_crosses
    for x in permutations:
        if not any(np.array_equal(x, z) for z in all_crosses):
            all_crosses.append(x)

# # draw out all the cross shapes
# fig, ax = plt.subplots(math.ceil(len(all_crosses)/8), 8)
# fig.set_size_inches(8, math.ceil(len(all_crosses)/8))
# for i, arch in enumerate(all_crosses):
#     ax[i//8, i%8].imshow(arch, vmax=3, interpolation='nearest', cmap='gnuplot2')
#     ax[i//8, i%8].set_xticks([])
#     ax[i//8, i%8].set_yticks([])
# plt.tight_layout()
# plt.savefig('all_crosses.png', dpi=300)
# plt.close()

# find all t-junctions in the image
cross_positions = find_subarrays_crosscorr(merged_mask, all_crosses)
print('number of crosses found: ', str(len(cross_positions)))
cross_x = [pos[0] for pos in cross_positions]
cross_y = [pos[1] for pos in cross_positions]

# make a distance transform for t-junctions
img_tjunc = np.zeros_like(merged_mask)
for pos in cross_positions:
    img_tjunc[tuple(int(p) for p in pos)] = 1
dt_tjunc = distance_transform_edt(np.logical_not(img_tjunc), sampling=(dim_xy, dim_xy))

fig, ax = plt.subplots(1,2)
fig.set_size_inches(9,4)
ax[0].imshow(img_tjunc, cmap='binary_r')
ax[0].scatter(cross_y, cross_x, s=3, c='none', edgecolors='#00ff00', linewidths=0.1)
ax[1].imshow(dt_tjunc, cmap='binary_r', vmax=100*dim_xy)
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_junctions^' + gene + '.png'), dpi=600)
plt.close()

# get spot distances to t-junction
true_dists_tjunc = []
for spot in spots_skel:
    true_dists_tjunc.append(dt_tjunc[tuple(spot.astype(int))])

# randomize positions of spots within the cytoskeletal region
indices = list(reg_skel_ind)
rand_dists_tjunc = []
N_RANDOMIZATIONS = 10000
for n in range(N_RANDOMIZATIONS):
    print('\rRandomizing spot positions: ' + str(n+1) + '/' + str(N_RANDOMIZATIONS) + '...', end='')
    rand_spots = random.sample(indices, len(spots_skel))
    for spot in rand_spots:
        rand_dists_tjunc.append(dt_tjunc[tuple(spot.astype(int))])
    
    # if n < 5:
    #     fig, ax = plt.subplots()
    #     ax.imshow(merged_mask, cmap='gnuplot2')
    #     rs = np.row_stack(rand_spots)
    #     ax.plot(rs[:,1], rs[:,0], 'm.', ms=1.5, mew=0)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.tight_layout()
    #     plt.savefig('rand_spots_' + str(n) + '.png', dpi=300)
    #     plt.close()

print('Done.')
max_dist = max(max(true_dists_tjunc), max(rand_dists_tjunc))

# plot histogram of spot distances to t-junction
fig, ax = plt.subplots()
ax.hist(true_dists_tjunc, bins=50, range=(0,max_dist), density=True, color='r', alpha=0.6)
ax.hist(rand_dists_tjunc, bins=50, range=(0,max_dist), density=True, color='k', alpha=0.6)
ax.set_xlabel('Distance to t-junction (um)')
ax.set_ylabel('Probability density')
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_dist_hist^' + gene + '.png'), dpi=300)
plt.close()

# plot CDF of spot distances to t-junction
fig, ax = plt.subplots()
ax.hist(true_dists_tjunc, bins=1000, range=(0,max_dist), density=True, histtype='step', cumulative=True, color='r')
ax.hist(rand_dists_tjunc, bins=1000, range=(0,max_dist), density=True, histtype='step', cumulative=True, color='k')
ax.set_xlabel('Distance to t-junction (um)')
ax.set_ylabel('Probability density')
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_dist_hist_cdf^' + gene + '.png'), dpi=300)
plt.close()

# draw the t-junction calls over the merged mask
fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
ax.imshow(merged_mask, vmax=3, cmap='gnuplot2')
ax.scatter(cross_y, cross_x, s=3, c='none', edgecolors='#00ff00', linewidths=0.1)
ax.plot(spots_skel[:,1], spots_skel[:,0], 'm.', ms=1.5, mew=0)
ax.plot(spots_nonskel[:,1], spots_nonskel[:,0], 'y.', ms=1.5, mew=0)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(p_out, 'plots', prefix + '_junc_overlay^' + gene + '.png'), dpi=600)
plt.close()

# rank sum test
U, p = ss.mannwhitneyu(true_dists_tjunc, rand_dists_tjunc, alternative='less')
print(U, p)

with open(os.path.join(p_out, 'stats', prefix + '_mannwhitney^' + gene + '.txt'), 'w') as outfile:
    outfile.write('\t'.join([prefix, str(U), str(p)]) + '\n')

# output all spot data
with open(os.path.join(p_out, 'dists', prefix + '_expt_distances^' + gene + '.txt'), 'w') as outfile:
    outfile.writelines([str(d)+'\n' for d in true_dists_tjunc])
with open(os.path.join(p_out, 'dists', prefix + '_rand_distances^' + gene + '.txt'), 'w') as outfile:
    outfile.writelines([str(d)+'\n' for d in rand_dists_tjunc])