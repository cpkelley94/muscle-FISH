# python 2.7.14, HiPerGator
"""
Goals:

1. transcript density per volume of fiber
2. ratio of nuclear to cytoplasmic transcripts
3. distance of cytoplasmic transcripts to nearest nucleus
"""
import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from czifile import CziFile
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from skimage import feature, exposure, filters, morphology
from xml.etree import ElementTree
import argparse
import csv
import matplotlib.cm as cm
import numpy as np
import os
import scipy.ndimage as ndi
import scope_utils

from skimage.filters import try_all_threshold

def count_spots(threshold):
    return len(feature.blob_log(imgs_rna[chan], 1., 1., num_sigma=1, threshold=threshold))

def update_z(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_facecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_edge(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_edgecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_grassfire(frame):
    dapi_xy.set_data(img_dapi[:,:,frame])
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = spots_xy.get_facecolor()
    for i, spot in enumerate(spots_xyz):
        alpha = scope_utils.gauss_1d(frame, 1., spot[2]/dims['Z'], 1.)
        colors[i,3] = alpha
    spots_xy.set_facecolor(colors)
    return dapi_xy, im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_compartments(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors_nuc = []
    colors_cyt = []
    colors_peri = []
    for spot in spots_by_region['nuc']:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors_nuc.append([0.3, 0.3, 1., alpha])
    for spot in spots_by_region['cyt']:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors_cyt.append([1., 0.3, 0.3, alpha])
    for spot in spots_by_region['peri']:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors_peri.append([0.3, 1., 0.3, alpha])
    spots_nuc_xy.set_facecolor(colors_nuc)
    spots_cyt_xy.set_facecolor(colors_cyt)
    spots_peri_xy.set_facecolor(colors_peri)
    return im_xy, spots_nuc_xy, spots_cyt_xy, spots_peri_xy  # this return structure is a requirement for the FuncAnimation() callable

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('czi', type=str, nargs=1, help='Path to .CZI image file.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('genes', type=str, nargs='*', help='Gene names, in order of appearance in .CZI file.')
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('-f', '--fiber-threshold', help='Threshold value for fiber segmentation (default: Li\'s method)', default=None)
parser.add_argument('-1', '--spot1-threshold', help='Threshold value for spot detection in first FISH channel (default: 0.018).', default=0.018)
parser.add_argument('-2', '--spot2-threshold', help='Threshold value for spot detection in second FISH channel (default: 0.018).', default=0.018)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations.')

# initialize flags
is_old_image = False
use_last_fiber_mask = False

# parse arguments
args = vars(parser.parse_args())
p_czi = args['czi'][0]
outdir = args['outdir'][0]
genes = args['genes']
should_plot = args['plot']
t_dapi = args['dapi_threshold']
t_fiber = args['fiber_threshold']
t_spot1 = float(args['spot1_threshold'])
t_spot2 = float(args['spot2_threshold'])
t_spots = [t_spot1, t_spot2]

if t_dapi:
    t_dapi = float(t_dapi)
if t_fiber:
    if t_fiber == 'last':
        use_last_fiber_mask = True
    else:
        t_fiber = float(t_fiber)

'''
# define gene table
gene_table = {1:['ttn', 'hnrnp'],
              2:['hist', 'actn'],
              3:['dmd', 'atp'],
              4:['gapdh', 'ribo'],
              5:['myom', 'actb'],
              6:['polr2a'],
              7:['vcl']}
all_genes = set([g for l in gene_table.itervalues() for g in l])
aliases = {'actn2':'actn'}
'''

# open the image
with CziFile(p_czi) as czi_file:
    meta = czi_file.metadata()
    img = czi_file.asarray()
img_name = os.path.splitext(os.path.basename(p_czi))[0]

if img_name[0].isdigit():
    is_old_image = True

'''
# get the list of genes present in the image
if img_name[0].isdigit():
    gkey = int(img_name[0])
    genes = gene_table[gkey]
    is_old_image = True
else:
    fnsplit = img_name.split('_')
    genes = []
    for word in fnsplit:
        if word in aliases:
            word = aliases[word]
        if word in all_genes:
            genes.append(word)
        else:
            break

# filename corrections
if not img_name[0].isdigit() and 'actb' in genes and 'myom' in genes:
    genes = ['myom', 'actb']
'''

print 'Analyzing `' + p_czi + '`...\n'
print 'Genes: ' + ', '.join(genes)


# split channels
img_dapi = scope_utils.normalize_image(img[0, 0, -1, 0, :, :, :, 0].transpose(1,2,0))
imgs_rna = [scope_utils.normalize_image(img[0, 0, i, 0, :, :, :, 0].transpose(1,2,0)) for i in range(len(genes))]

# if should_plot:
#     scope_utils.animate_zstacks(imgs_rna + [img_dapi], vmax=[0.2, 0.2, 1])

'''
# deconvolve DAPI
img_dapi_deconv = scope_utils.deconvolve_widefield(img_dapi, meta, 405, 405, psfsize_xy=10, psfsize_z=10, plot_psf=True)
scope_utils.animate_zstacks([img_rna, img_dapi, img_dapi_deconv])
'''

# get voxel dimensions
mtree = ElementTree.fromstring(meta)
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
voxel_vol = dims['X'] * dims['Y'] * dims['Z']  # [um^3]

# get channel wavelengths
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    name = track.get('Name')
    wavelen = int(float(track.find('Attenuators').find('Attenuator').find('Wavelength').text)*1E9)
    tracks.append(wavelen)

print 'Tracks: ' + ', '.join(map(str, tracks))

# mark channels to use for fiber prediction
fiber_chans = [False if is_old_image and any([tracks[i] == 561, genes[i].lower() == 'ribo']) else True for i in range(len(genes))]
if not any(fiber_chans):
    # if there are no good channels for fiber segmentation, just use first
    fiber_chans[0] = True

# segment fiber using information from all RNA channels
img_rna_average = np.mean(np.array([im for i, im in enumerate(imgs_rna) if fiber_chans[i]]), axis=0)
img_rna_blurxy = filters.gaussian(img_rna_average, (10, 10, 1))
threshs = []
img_fiber_binary = np.zeros_like(img_rna_blurxy)
img_fiber_bool = np.zeros_like(img_rna_blurxy)

if type(t_fiber) != float:
    for z in range(img_rna_blurxy.shape[2]):
        thresh = filters.threshold_li(img_rna_blurxy[:,:,z])
        threshs.append(thresh)
        img_fiber_binary[:,:,z] = np.where(img_rna_blurxy[:,:,z] > thresh, 1, 0)
else:
    img_fiber_binary = np.where(img_rna_blurxy > t_fiber, 1, 0)
    threshs.append(t_fiber)

print '\nFiber thresholds:'
print threshs

if use_last_fiber_mask:
    # only use the last frame to define the fiber volume
    # this is a last resort option to deal with glare in the RNA channel
    for z in range(img_fiber_binary.shape[2]):
        img_fiber_binary[:,:,z] = img_fiber_binary[:,:,-1]
    print 'Override: using final frame mask for entire z-stack.'

# keep only largest object in the image
fiber_labeled, n_labels = morphology.label(img_fiber_binary, return_num=True)
label_sizes = {i:np.count_nonzero(np.where(fiber_labeled == i, 1, 0)) for i in range(1, n_labels+1)}
fiber_idx = max(label_sizes, key=label_sizes.get)
img_fiber_only = np.where(fiber_labeled == fiber_idx, 1, 0)

'''
# remove 1-pixel x-y border
border_mask = np.ones_like(img_fiber_only)
border_mask[0,:,:] = 0
border_mask[-1,:,:] = 0
border_mask[:,0,:] = 0
border_mask[:,-1,:] = 0
img_fiber_only = np.where(np.logical_and(img_fiber_only.astype(bool), border_mask.astype(bool)), 1, 0)
'''

# calculate volume of fiber
volume = np.sum(img_fiber_only) * voxel_vol
print '\nfiber = ' + str(float(100.*np.sum(img_fiber_only))/img_fiber_only.size) + '% of FOV'
print 'volume = ' + str(volume) + ' um^3\n'

if should_plot:
    scope_utils.animate_zstacks([np.clip(img_rna_average, a_min=0, a_max=np.amax(img_rna_average)/30.), img_fiber_binary, img_fiber_only], titles=['average of RNAs', 'threshold_li', 'fiber prediction'], gif_name=os.path.join(outdir, 'anim', img_name+'_fiber.gif'))

# threshold nuclei using Otsu's method
img_dapi_blurxy = filters.gaussian(img_dapi, (20, 20, 2))

if not t_dapi:
    thresh_dapi = filters.threshold_otsu(img_dapi_blurxy)*0.75
else:
    thresh_dapi = t_dapi
print 'DAPI threshold: ' + str(thresh_dapi)

img_dapi_binary = np.where(img_dapi > thresh_dapi, 1, 0)
img_dapi_binary = morphology.remove_small_objects(img_dapi_binary.astype(bool), min_size=2048)  # without this, 1-pixel noise becomes huge abberant perinuclear region
img_dapi_binary = morphology.remove_small_holes(img_dapi_binary.astype(bool), min_size=2048)

# remove nuclei that are not part of main fiber segment
nuclei_labeled, n_nuc = morphology.label(img_dapi_binary, return_num=True)
for i in range(1, n_nuc+1):
    overlap = np.logical_and(np.where(nuclei_labeled == i, True, False), img_fiber_only.astype(bool))
    if np.count_nonzero(overlap) == 0:
        # print 'non-fiber nucleus: ' + str(i)
        nuclei_labeled[nuclei_labeled == i] = 0

# define nuclear, perinuclear, and cytoplasmic region
region_nuc = np.where(nuclei_labeled > 0, 1, 0)  # initialize for nuclear segmentation
for n in range(1):  # ~ 0.5 um in z
    region_nuc = morphology.binary_erosion(region_nuc)  # contract x, y, z
for n in range(10):  # ~ 0.5 um in x, y
    for z in range(region_nuc.shape[2]):
        region_nuc[:,:,z] = morphology.binary_erosion(region_nuc[:,:,z])  # contract x, y only

img_nuc_dilated = np.where(nuclei_labeled > 0, 1, 0)  # initialize for perinuclear segmentation
for n in range(4):  # ~ 2 um in z
    img_nuc_dilated = morphology.binary_dilation(img_nuc_dilated)  # expand x, y, z
for n in range(40):  # ~ 2 um in x, y
    for z in range(img_nuc_dilated.shape[2]):
        img_nuc_dilated[:,:,z] = morphology.binary_dilation(img_nuc_dilated[:,:,z])  # expand x, y only

region_peri = np.where(np.logical_and(img_nuc_dilated > region_nuc, np.logical_or(img_fiber_only.astype(bool), img_dapi_binary.astype(bool))), 1, 0)
region_cyt = np.where(img_fiber_only > img_nuc_dilated, 1, 0)
regions = {'nuc':region_nuc, 'cyt':region_cyt, 'peri':region_peri}

if should_plot:
    scope_utils.animate_zstacks([region_nuc, region_peri, region_cyt, region_cyt + 2*region_peri + 3*region_nuc], titles=['nuclear', 'perinuclear', 'sarcoplasmic', 'all compartments'], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], gif_name=os.path.join(outdir, 'anim', img_name+'_regions.gif'))

# generate distance transform from nucleus region
grassfire = distance_transform_edt(np.logical_not(region_nuc), sampling=(dims['X'], dims['Y'], dims['Z']))
grid_x = [dims['X']*xi for xi in range(grassfire.shape[0])]
grid_y = [dims['Y']*yi for yi in range(grassfire.shape[1])]
grid_z = [dims['Z']*zi for zi in range(grassfire.shape[2])]

if should_plot:
    scope_utils.animate_zstacks([img_dapi, grassfire], gif_name=os.path.join(outdir, 'anim', img_name+'_dist_transform.gif'))

# iterate over RNAs in image
for chan, gene in enumerate(genes):
    print '--------------------------------'

    if gene == 'ribo':
        # can't resolve spots, skip this gene
        print 'Skipping gene `' + gene + '`: cannot resolve spots.'
        continue

    print 'Analyzing ' + gene + ' gene (' + str(chan+1) + '/' + str(len(genes)) + ')'

    # find FISH spots in 3D
    spots = feature.blob_log(imgs_rna[chan], 1., 1., num_sigma=1, threshold=t_spots[chan])
    print str(spots.shape[0]) + ' blobs detected.'

    # eliminate spots detected outside the fiber
    spots_masked = []
    for spot in spots:
        spot_pos = tuple(spot[0:3].astype(int))
        if img_fiber_only[spot_pos]:
            spots_masked.append(spot)
    spots_masked = np.row_stack(spots_masked)
    print str(spots_masked.shape[0]) + ' spots detected within fiber.'

    # animate spot detection
    '''
    if should_plot:
        fig, ax = plt.subplots(1,2)
        im_xy = ax[0].imshow(imgs_rna[chan][:,:,0], vmin=np.amin(imgs_rna[chan]), vmax=np.amax(imgs_rna[chan])/10., cmap='binary_r')
        ax[1].set_facecolor('k')
        spots_xy = ax[1].scatter(spots_masked[:,1], -1.*spots_masked[:,0], s=5, c='k', edgecolors='none')
        anim = FuncAnimation(fig, update_z, frames=imgs_rna[chan].shape[2], interval=100, blit=False)
        # plt.show()
        plt.close()
    '''
    if should_plot:
        fig, ax = plt.subplots()
        im_xy = ax.imshow(imgs_rna[chan][:,:,0], vmin=np.amin(imgs_rna[chan]), vmax=np.amax(imgs_rna[chan])/10., cmap='binary_r')
        spots_xy = ax.scatter(spots_masked[:,1], spots_masked[:,0], s=12, c='none', edgecolors='k')
        anim = FuncAnimation(fig, update_z_edge, frames=imgs_rna[chan].shape[2], interval=100, blit=False)
        # plt.show()
        anim.save(os.path.join(outdir, 'anim', img_name + '_rings^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
        plt.close()

    # get distance of FISH spots to nearest nucleus
    spots_xyz = np.array([np.multiply(s[:3], [dims['X'], dims['Y'], dims['Z']]) for s in spots_masked if not np.isnan(s[0])])
    spots_dist = interpn((grid_x, grid_y, grid_z), grassfire, spots_xyz)
    max_dist = np.amax(spots_dist)

    # output spot coordinates distances to file
    out_dist_list = [['x (um)', 'y (um)', 'z (um)', 'd_nuc (um)']]
    for i, pos in enumerate(spots_xyz):
        out_dist_list.append(list(pos) + [spots_dist[i]])

    with open(os.path.join(outdir, 'dists', img_name + '_dists^' + gene + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(out_dist_list)

    if should_plot:
        fig, ax = plt.subplots()
        ax.hist(spots_dist, bins=100, density=True)
        ax.set_xlabel('Distance from nucleus')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'dists', img_name + '_nuc_dist^' + gene + '.png'), dpi=300)
        # plt.show()
        plt.close()

        cmap = cm.get_cmap('viridis_r')
        grassfire_colors = [cmap(d/max_dist) for d in spots_dist]
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(10, 4)
        dapi_xy = ax[0].imshow(img_dapi[:,:,0], vmin=np.amin(img_dapi), vmax=np.amax(img_dapi)/3., cmap='bone')
        im_xy = ax[1].imshow(imgs_rna[chan][:,:,0], vmin=np.amin(imgs_rna[chan]), vmax=np.amax(imgs_rna[chan])/10., cmap='gist_heat')
        ax[2].set_facecolor('k')
        spots_xy = ax[2].scatter(spots_xyz[:,1], -1.*spots_xyz[:,0], s=5, c=grassfire_colors, edgecolors='none')
        ax[2].set_aspect('equal')
        plt.tight_layout()
        anim = FuncAnimation(fig, update_z_grassfire, frames=imgs_rna[chan].shape[2], interval=100, blit=False)
        anim.save(os.path.join(outdir, 'anim', img_name + '_grassfire^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
        # plt.show()
        plt.close()

    # categorize spots by compartment
    spots_nuc = []
    spots_cyt = []
    spots_peri = []

    for spot in spots_masked:
        spot_pos = tuple(spot[0:3].astype(int))
        if region_nuc[spot_pos]:
            spots_nuc.append(spot)
        elif region_peri[spot_pos]:
            spots_peri.append(spot)
        elif region_cyt[spot_pos]:
            spots_cyt.append(spot)
        else:
            # reason for this is unknown
            print 'warning: spot at position ' + str(spot_pos) + ' not assigned to a compartment.'

    spots_by_region = {}
    spots_by_region.update({'nuc':np.row_stack(spots_nuc)})
    spots_by_region.update({'cyt':np.row_stack(spots_cyt)})
    spots_by_region.update({'peri':np.row_stack(spots_peri)})
    region_names = sorted(list(spots_by_region.keys()))

    # compare compartments
    counts = {}
    vols = {}
    for reg in region_names:
        counts.update({reg:len(spots_by_region[reg])})
        vols.update({reg:np.sum(regions[reg])*voxel_vol})

    # output results to stdout
    print 'nuclear:      ' + str(counts['nuc']) + ' spots, ' + str(float(counts['nuc'])/vols['nuc']) + ' spots/um^3, ' + str(100.*float(counts['nuc'])/np.sum(counts.values())) + str('% of spots')
    print 'perinuclear:  ' + str(counts['peri']) + ' spots, ' + str(float(counts['peri'])/vols['peri']) + ' spots/um^3, ' + str(100.*float(counts['peri'])/np.sum(counts.values())) + str('% of spots')
    print '(nuc + peri:  ' + str(counts['peri'] + counts['nuc']) + ' spots, ' + str(float(counts['peri'] + counts['nuc'])/(vols['peri'] + vols['nuc'])) + ' spots/um^3, ' + str(100.*float(counts['peri'] + counts['nuc'])/np.sum(counts.values())) + str('% of spots') + ')'
    print 'sarcoplasmic: ' + str(counts['cyt']) + ' spots, ' + str(float(counts['cyt'])/vols['cyt']) + ' spots/um^3, ' + str(100.*float(counts['cyt'])/np.sum(counts.values())) + str('% of spots')

    # output results to file
    with open(os.path.join(outdir, 'counts', img_name + '_counts^' + gene + '.csv'), 'w') as outfile:
        out_list = [['region', 'vol (um^3)', 'count', 'density (spots/um^3)', 'counts_ratio']]
        for reg in region_names:
            out_list.append([reg, vols[reg], counts[reg], float(counts[reg])/vols[reg], float(counts[reg])/np.sum(counts.values())])
        writer = csv.writer(outfile)
        writer.writerows(out_list)

    if should_plot:
        fig, ax = plt.subplots(1,2)
        im_xy = ax[0].imshow(imgs_rna[chan][:,:,0], vmin=np.amin(imgs_rna[chan]), vmax=np.amax(imgs_rna[chan])/10., cmap='binary_r')
        ax[1].set_facecolor('k')
        spots_nuc_xy = ax[1].scatter(spots_by_region['nuc'][:,1], -1.*spots_by_region['nuc'][:,0], s=5, c='k', edgecolors='none')
        spots_cyt_xy = ax[1].scatter(spots_by_region['cyt'][:,1], -1.*spots_by_region['cyt'][:,0], s=5, c='k', edgecolors='none')
        spots_peri_xy = ax[1].scatter(spots_by_region['peri'][:,1], -1.*spots_by_region['peri'][:,0], s=5, c='k', edgecolors='none')
        ax[1].set_aspect('equal')
        anim = FuncAnimation(fig, update_z_compartments, frames=imgs_rna[chan].shape[2], interval=100, blit=False)
        anim.save(os.path.join(outdir, 'anim', img_name + '_rna_compartments^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
        # plt.show()
        plt.close()
