# python 3.6.5, HiPerGator
"""
Goals:

1. transcript density per volume of fiber
2. ratio of nuclear to cytoplasmic transcripts
3. distance of cytoplasmic transcripts to nearest nucleus
"""
import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
# from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
from mpl_toolkits import mplot3d
from numpy.linalg import norm
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from skimage import feature, exposure, filters, morphology, measure
from xml.etree import ElementTree
import argparse
import csv
import matplotlib.cm as cm
import numpy as np
import os
import scipy.ndimage as ndi
import trimesh

# custom libraries
import scope_utils3 as su
import muscle_fish as mf


#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

def count_spots(threshold):
    return len(feature.blob_log(imgs_rna[chan], 1., 1., num_sigma=1, threshold=threshold))

def update_z(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_facecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_edge(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_edgecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_grassfire(frame):
    dapi_xy.set_data(img_dapi[:,:,frame])
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors = spots_xy.get_facecolor()
    for i, spot in enumerate(spots_xyz):
        alpha = su.gauss_1d(frame, 1., spot[2]/dims['Z'], 1.)
        colors[i,3] = alpha
    spots_xy.set_facecolor(colors)
    return dapi_xy, im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_compartments(frame):
    im_xy.set_data(imgs_rna[chan][:,:,frame])
    colors_nuc = []
    colors_cyt = []
    colors_peri = []
    for spot in spots_by_region['nuc']:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors_nuc.append([0.3, 0.3, 1., alpha])
    for spot in spots_by_region['cyt']:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors_cyt.append([1., 0.3, 0.3, alpha])
    for spot in spots_by_region['peri']:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors_peri.append([0.3, 1., 0.3, alpha])
    spots_nuc_xy.set_facecolor(colors_nuc)
    spots_cyt_xy.set_facecolor(colors_cyt)
    spots_peri_xy.set_facecolor(colors_peri)
    return im_xy, spots_nuc_xy, spots_cyt_xy, spots_peri_xy  # this return structure is a requirement for the FuncAnimation() callable


#--  COMMAND LINE ARGUMENTS  --------------------------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('img', type=str, nargs=1, help='Path to image file (CZI or OME-TIFF).')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('genes', type=str, nargs='*', help='Gene names, in order of appearance in image file.')
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('-f', '--fiber-threshold', help='Threshold value for fiber segmentation (default: Li\'s method)', default=None)
parser.add_argument('-1', '--spot-threshold1', help='Threshold value for spot detection in FISH channel 1 (default: 0.02)', default=0.02)
parser.add_argument('-2', '--spot-threshold2', help='Threshold value for spot detection in FISH channel 2 (default: 0.02)', default=0.02)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations.')

# initialize flags
is_old_image = False

# parse arguments
args = vars(parser.parse_args())
p_img = args['img'][0]
outdir = args['outdir'][0]
genes = args['genes']
should_plot = args['plot']
t_dapi = args['dapi_threshold']
t_fiber = args['fiber_threshold']
t_spot1 = args['spot_threshold1']
t_spot2 = args['spot_threshold2']

if t_dapi:
    t_dapi = float(t_dapi)
if t_fiber and t_fiber != 'last':
    t_fiber = float(t_fiber)
if t_spot1 is not None:
    t_spot1 = float(t_spot1)
if t_spot2 is not None:
    t_spot2 = float(t_spot2)
t_spot = [t_spot1, t_spot2]


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

# determine image filetype and open
print('Analyzing `' + p_img + '`...\n')
print('Genes: ' + ', '.join(genes))

img, img_name, mtree = mf.open_image(p_img)

if img_name[0].isdigit():
    is_old_image = True

img_dapi = su.normalize_image(img[-1,:,:,:])
imgs_rna = [su.normalize_image(img[i,:,:,:]) for i in range(len(genes))]

if should_plot:
    su.animate_zstacks(imgs_rna + [img_dapi], vmax=[0.2]*len(imgs_rna) + [1], gif_name=os.path.join(outdir, 'anim', img_name+'_channels.gif'))

# get voxel dimensions
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
voxel_vol = dims['X'] * dims['Y'] * dims['Z']  # [um^3]
dims_xyz = np.array([dims['X'], dims['Y'], dims['Z']])

# get channel wavelengths
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    name = track.get('Name')
    wavelen = int(float(track.find('Attenuators').find('Attenuator').find('Wavelength').text)*1E9)
    tracks.append(wavelen)

print('Tracks: ' + ', '.join(map(str, tracks)))


#--  FIBER SEGMENTATION  ------------------------------------------------------#

# mark channels to use for fiber prediction
fiber_chans = [False if is_old_image and any([tracks[i] == 561, genes[i].lower() == 'ribo']) else True for i in range(len(genes))]
if not any(fiber_chans):
    # if there are no good channels for fiber segmentation, just use first
    fiber_chans[0] = True

# segment fiber using information from all RNA channels
img_rna_average = np.mean(np.array([im for i, im in enumerate(imgs_rna) if fiber_chans[i]]), axis=0)
img_fiber_only = mf.threshold_fiber(img_rna_average, t_fiber=t_fiber, verbose=True)

# calculate volume of fiber
volume = np.sum(img_fiber_only) * voxel_vol
print('\nfiber = ' + str(float(100.*np.sum(img_fiber_only))/img_fiber_only.size) + '% of FOV')
print('volume = ' + str(volume) + ' um^3\n')

if should_plot:
    su.animate_zstacks([np.clip(img_rna_average, a_min=0, a_max=np.amax(img_rna_average)/30.), img_fiber_only], titles=['average of RNAs', 'fiber prediction'], gif_name=os.path.join(outdir, 'anim', img_name+'_fiber.gif'))


#--  SEGMENTATION OF NUCLEAR, PERINUCLEAR, AND SARCOPLASMIC COMPARTMENTS  -----#

nuclei_labeled = mf.threshold_nuclei(img_dapi, t_dapi=t_dapi, fiber_mask=img_fiber_only, verbose=True)
nuclei_binary = np.where(nuclei_labeled > 0, 1, 0)

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

region_peri = np.where(np.logical_and(img_nuc_dilated > region_nuc, np.logical_or(img_fiber_only.astype(bool), nuclei_binary.astype(bool))), 1, 0)
region_cyt = np.where(img_fiber_only > img_nuc_dilated, 1, 0)
regions = {'nuc':region_nuc, 'cyt':region_cyt, 'peri':region_peri}

if should_plot:
    su.animate_zstacks([region_nuc, region_peri, region_cyt, region_cyt + 2*region_peri + 3*region_nuc], titles=['nuclear', 'perinuclear', 'sarcoplasmic', 'all compartments'], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], gif_name=os.path.join(outdir, 'anim', img_name+'_regions.gif'))


#--  GENERATION OF NUCLEAR MESHES AND DISTANCE TRANSFORM  ---------------------#

# generate distance transform from nucleus region
grassfire = distance_transform_edt(np.logical_not(region_nuc), sampling=dims_xyz)
grid_x = [dims['X']*xi for xi in range(grassfire.shape[0])]
grid_y = [dims['Y']*yi for yi in range(grassfire.shape[1])]
grid_z = [dims['Z']*zi for zi in range(grassfire.shape[2])]

if should_plot:
    su.animate_zstacks([img_dapi, grassfire], gif_name=os.path.join(outdir, 'anim', img_name+'_dist_transform.gif'))

# generate mesh corresponding to the surface of each nucleus
print('Generating nuclear surface meshes...')
region_nuc_labeled, n_nuc = morphology.label(region_nuc, return_num=True)
nuc_props = measure.regionprops(region_nuc_labeled)
region_nuc_labeled = region_nuc_labeled - 1  # shift so bg = -1 and nuclei = [0:n_nuc]
nuc_meshes = []
centroids = []
for i in range(n_nuc):
    nuc = np.where(region_nuc_labeled == i, 1, 0)
    nuc_fade = filters.gaussian(nuc.astype(float), (3, 3, 1))

    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(nuc_fade, 0.5, spacing=dims_xyz)
    except ValueError:
        # mesh generation fails for aberrant small DAPI signals
        # skip this
        nuc_meshes.append(None)
        centroids.append(None)
        continue

    mesh = trimesh.Trimesh(verts, faces, normals)
    nuc_meshes.append(mesh)
    centroid = np.array(nuc_props[i].centroid)*dims_xyz
    centroids.append(centroid)
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    tris = Triangulation(verts[:,0], verts[:,1], faces)
    ax.plot_trisurf(tris, verts[:,2])
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    plt.savefig('test_nuc_' + str(i) + '.png', dpi=300)
    plt.close()
    '''


#-- RNA DETECTION AND ANALYSIS  -----------------------------------------------#

# iterate over RNAs in image
for chan, gene in enumerate(genes):
    print('--------------------------------')

    if gene == 'ribo':
        # can't resolve spots, skip this gene
        print('Skipping gene `' + gene + '`: cannot resolve spots.')
        continue

    print('Analyzing ' + gene + ' gene (' + str(chan+1) + '/' + str(len(genes)) + ')')


    #--  FISH SPOT DETECTION  -------------------------------------------------#

    print('Finding FISH spots...')
    # img_rna_corr = mf.fix_bleaching(imgs_rna[chan], mask=img_fiber_only, draw=True, imgprefix=img_name+'^'+gene)
    # spots_masked, spot_data = mf.find_spots_snrfilter(img_rna_corr, sigma=2, snr=t_snr[chan], t_spot=0.025, mask=img_fiber_only, imgprefix=img_name)
    spots_masked = mf.find_spots(imgs_rna[chan], t_spot=t_spot[chan], mask=img_fiber_only)

    # with open(os.path.join(outdir, img_name + '_spot_data_intensities^' + gene + '.csv'), 'w') as spot_file:
    #     writer = csv.writer(spot_file)
    #     writer.writerows(spot_data)

    print(str(spots_masked.shape[0]) + ' spots detected within fiber.')

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


    #--  NUCLEAR DISTANCE MEASUREMENT  ----------------------------------------#

    # get distance of FISH spots to nearest nucleus
    spots_xyz = np.array([s[:3]*dims_xyz for s in spots_masked if not np.isnan(s[0])])
    spots_dist = interpn((grid_x, grid_y, grid_z), grassfire, spots_xyz)
    max_dist = np.amax(spots_dist)

    # assign nuclear FISH spots to a nucleus and calculate relative distance from center
    for i in range(len(spots_dist)):
        if spots_dist[i] == 0:  # spot in a nucleus
            # identify nucleus
            pos = spots_xyz[i,:] / dims_xyz
            pos_idx = tuple(pos.astype(int))
            label = region_nuc_labeled[pos_idx]

            if label == -1:
                # not in nucleus, must be right on border, call outside
                continue

            if not nuc_meshes[label]:
                # mesh generation failed for this nucleus
                # move the spot outside the plotting range
                spots_dist[i] = -3
                continue

            # calculate distance from centroid by ray casting
            r_spot = spots_xyz[i,:]
            r_0 = centroids[label]
            rel_pos = r_spot - r_0
            unit_vec = rel_pos / norm(rel_pos)
            locations, _, _ = nuc_meshes[label].ray.intersects_location(np.array([r_0]), np.array([unit_vec]))

            try:
                r_intersect = locations[0]
            except IndexError:
                # no intersection was found
                # probably because the nucleus was cut off in the image
                # move the spot outside the plotting range
                spots_dist[i] = -2
                continue

            intranuc_dist = norm(r_spot - r_0) / norm(r_intersect - r_0)
            if intranuc_dist > 1:
                intranuc_dist = 1.

            spots_dist[i] = intranuc_dist - 1

            '''
            # draw nucleus with spot, centroid, and ray casting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            verts = nuc_meshes[label].vertices
            tris = Triangulation(verts[:,0], verts[:,1], nuc_meshes[label].faces)
            ax.plot_trisurf(tris, verts[:,2], color=(0,0,0,0.4))
            ax.plot([r_0[0], r_intersect[0]], [r_0[1], r_intersect[1]], [r_0[2], r_intersect[2]], 'k-')
            ax.plot(*(list(map(lambda x: [x], r_0)) + ['bo']))
            ax.plot(*(list(map(lambda x: [x], r_spot)) + ['ro']))
            ax.plot(*(list(map(lambda x: [x], r_intersect)) + ['go']))
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            ax.set_zlabel('z (um)')
            plt.savefig('test_spot_' + str(i) + '_nuc_' + str(label) + '.png', dpi=300)
            plt.close()
            '''

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


    #--  ASSIGNMENT OF SPOTS TO COMPARTMENTS  ---------------------------------#

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
            print('warning: spot at position ' + str(spot_pos) + ' not assigned to a compartment.')

    spots_by_region = {}
    try:
        spots_by_region.update({'nuc':np.row_stack(spots_nuc)})
    except ValueError:
        spots_by_region.update({'nuc':np.empty((0,3))})
    try:
        spots_by_region.update({'cyt':np.row_stack(spots_cyt)})
    except ValueError:
        spots_by_region.update({'cyt':np.empty((0,3))})
    try:
        spots_by_region.update({'peri':np.row_stack(spots_peri)})
    except ValueError:
        spots_by_region.update({'peri':np.empty((0,3))})
    region_names = sorted(list(spots_by_region.keys()))

    # compare compartments
    counts = {}
    vols = {}
    for reg in region_names:
        counts.update({reg:len(spots_by_region[reg])})
        vols.update({reg:np.sum(regions[reg])*voxel_vol})


    #--  OUTPUT FILE OPERATIONS  ----------------------------------------------#

    # output results to stdout
    print('nuclear:      ' + str(counts['nuc']) + ' spots, ' + str(float(counts['nuc'])/vols['nuc']) + ' spots/um^3, ' + str(100.*float(counts['nuc'])/np.sum(list(counts.values()))) + str('% of spots'))
    print('perinuclear:  ' + str(counts['peri']) + ' spots, ' + str(float(counts['peri'])/vols['peri']) + ' spots/um^3, ' + str(100.*float(counts['peri'])/np.sum(list(counts.values()))) + str('% of spots'))
    print('(nuc + peri:  ' + str(counts['peri'] + counts['nuc']) + ' spots, ' + str(float(counts['peri'] + counts['nuc'])/(vols['peri'] + vols['nuc'])) + ' spots/um^3, ' + str(100.*float(counts['peri'] + counts['nuc'])/np.sum(list(counts.values()))) + str('% of spots') + ')')
    print('sarcoplasmic: ' + str(counts['cyt']) + ' spots, ' + str(float(counts['cyt'])/vols['cyt']) + ' spots/um^3, ' + str(100.*float(counts['cyt'])/np.sum(list(counts.values()))) + str('% of spots'))

    # output results to file
    with open(os.path.join(outdir, 'counts', img_name + '_counts^' + gene + '.csv'), 'w') as outfile:
        out_list = [['region', 'vol (um^3)', 'count', 'density (spots/um^3)', 'counts_ratio']]
        for reg in region_names:
            out_list.append([reg, vols[reg], counts[reg], float(counts[reg])/vols[reg], float(counts[reg])/np.sum(list(counts.values()))])
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
