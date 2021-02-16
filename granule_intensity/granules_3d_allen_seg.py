# python 3.6.5, HiPerGator
"""
Goals:

1. find distribution of FISH spot intensities for single transcripts
2. estimate number of transcripts per perinuclear cluster
3. compare transcript overlap between same nucleus and (control) different nuclei
"""

import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
# from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from mpl_toolkits import mplot3d
from numpy.linalg import norm
from scipy.interpolate import interpn
from scipy.ndimage import interpolation
from scipy.ndimage.morphology import distance_transform_edt
from skimage import feature, exposure, filters, morphology, measure
from xml.etree import ElementTree
import argparse
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import csv
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import scipy.ndimage as ndi
import scipy.optimize as optimize
import seaborn as sns
import tifffile
import trimesh

# custom libraries
import scope_utils3 as su
import muscle_fish as mf

np.setbufsize(1E7)

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

def update_z_spot(fm):
    # update fish frame
    im.set_data(imgs_rna[chan][x_int-10:x_int+10, y_int-10:y_int+10, fm])
    if plot_circ:
        alpha = np.exp(-1.*((float(fm)-z)**2.)/(2.*(sig_z**2.)))
        circ.set_edgecolor((1., 0., 0., alpha))
        return im, circ

    return im

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

def transform_mask(tf, m2):
    cent2 = np.array(measure.regionprops(m2)[0].centroid).astype(int)
    m2_shift_center = interpolation.shift(m2, shift=np.append((np.array(m2.shape[:2])/2.).astype(int)-cent2[:2], [0]), order=0)
    m2_rotate = interpolation.rotate(m2_shift_center, angle=tf[3], order=0, reshape=False)
    m2_shift_back = interpolation.shift(m2_rotate, shift=np.append(cent2[:2]-(np.array(m2.shape[:2])/2.).astype(int), [0]), order=0)
    m2_final = interpolation.shift(m2_shift_back, shift=tf[:3], order=0)
    return m2_final

def transform_image(tf, ctr, img):
    # ctr = np.array(measure.regionprops(img)[0].centroid).astype(int)
    img_shift_center = interpolation.shift(img, shift=np.append((np.array(img.shape[:2])/2.).astype(int)-ctr[:2], [0]), order=3)
    img_rotate = interpolation.rotate(img_shift_center, angle=tf[3], order=3, reshape=False)
    img_shift_back = interpolation.shift(img_rotate, shift=np.append(ctr[:2]-(np.array(img.shape[:2])/2.).astype(int), [0]), order=3)
    img_final = interpolation.shift(img_shift_back, shift=tf[:3], order=3)
    return img_final

def align_masks(mask1, mask2):
    '''
    Align two binary integer arrays. Allow floating point transforms in x and y, 
    but integer only in z. Allow floating point rotations with interpolation.
    '''

    def optim_func(tf, m1, m2):
        m2_tf = interpolation.shift(interpolation.rotate(m2, angle=tf[3], order=0, reshape=False), shift=[tf[0], tf[1], int(tf[2])], order=0)
        # print(tf, -1*np.count_nonzero(np.logical_and(m1.astype(bool), m2_tf.astype(bool))))
        return -1*np.count_nonzero(np.logical_and(m1.astype(bool), m2_tf.astype(bool)))

    centroid1 = np.array(measure.regionprops(mask1)[0].centroid).astype(int)
    centroid2 = np.array(measure.regionprops(mask2)[0].centroid).astype(int)
    feret1 = measure.regionprops(mask1)[0].major_axis_length
    feret2 = measure.regionprops(mask2)[0].major_axis_length
    max_diam = max(feret1, feret2)
    rad = int(0.6*max_diam)

    if (centroid1[0] - rad < 0) or (centroid1[0] + rad > mask1.shape[0]) or (centroid1[1] - rad < 0) or (centroid1[1] + rad > mask1.shape[1]):
        # feature in mask1 is too close to boundary
        return False
    if (centroid2[0] - rad < 0) or (centroid2[0] + rad > mask2.shape[0]) or (centroid2[1] - rad < 0) or (centroid2[1] + rad > mask2.shape[1]):
        # feature in mask2 is too close to boundary
        return False

    mask1_crop = mask1[centroid1[0]-rad:centroid1[0]+rad, centroid1[1]-rad:centroid1[1]+rad, :]
    mask2_crop = mask2[centroid2[0]-rad:centroid2[0]+rad, centroid2[1]-rad:centroid2[1]+rad, :]

    # su.animate_zstacks([mask1_crop, mask2_crop], cmaps=['binary_r', 'binary_r'], gif_name='anim/test_crop.gif')  

    # optimize z
    z_arr = np.empty(((mask2.shape[2]*2)+1, 2))
    for i, z_shift in enumerate(range(-1*mask2.shape[2], mask2.shape[2]+1)):
        z_arr[i,0] = z_shift
        z_arr[i,1] = optim_func([0, 0, z_shift, 0], mask1_crop, mask2_crop)
        # z_dict[z_shift] = optim_func([0, 0, z_shift, 0], mask1_crop, mask2_crop)
    z_func_min = np.amin(z_arr[:,1])
    z_opt = z_arr[np.argwhere(z_arr[:,1] == z_func_min)[0], 0][0]

    # optimize angle
    cur_opt_angle = 0.
    interval = 3.
    window = 5
    angle_arr = np.empty(((window*2)+1, 2))
    shift_dir = 'none'
    tried_flipped = False
    while True:
        angles_to_test = [cur_opt_angle+(interval*i) for i in range(-1*window, window+1)]
        for i, theta in enumerate(angles_to_test):
            angle_arr[i,0] = theta
            angle_arr[i,1] = optim_func([0, 0, z_opt, theta], mask1_crop, mask2_crop)
            # angle_dict[optim_func([0, 0, z_opt, theta], mask1_crop, mask2_crop)] = theta
        angle_func_min = np.amin(angle_arr[:,1])
        
        if np.all(angle_arr[:,1] == angle_func_min):
            # all values are the same, optimization complete
            if not tried_flipped:
                # try 180 degree rotation
                angle_opt = cur_opt_angle
                func_min_opt = angle_func_min
                cur_opt_angle += 180
                interval = 1.
                tried_flipped = True
                continue
            else:
                if func_min_opt < angle_func_min:
                    break
                else:
                    angle_opt = cur_opt_angle
                    break 
        elif angle_arr[0,1] == angle_func_min:
            # best angle is at the edge of the window
            if shift_dir == 'right':
                # we just shifted the other way last time
                # to prevent infinite loop, take the halfway point and optimize around that
                cur_opt_angle = (angles_to_test[0] + angles_to_test[window])/2.
                interval /= 2.
                shift_dir = 'none'
                continue
            else:
                # recenter the window and try again
                cur_opt_angle = angle_arr[0,0]
                shift_dir = 'left'
                # print('left')
                continue
        elif angle_arr[-1,1] == angle_func_min:
            # best angle is at the edge of the window
            if shift_dir == 'left':
                # we just shifted the other way last time
                # to prevent infinite loop, take the halfway point and optimize around that
                cur_opt_angle = (angles_to_test[-1] + angles_to_test[window])/2.
                interval /= 2.
                shift_dir = 'none'
                continue
            else:
                # recenter the window and try again
                cur_opt_angle = angle_arr[-1,0]
                shift_dir = 'right'
                # print('right')
                continue
        else:
            # find the best point and zoom in on it
            shift_dir = 'none'
            cur_opt_angle = angle_arr[np.argwhere(angle_arr[:,1] == angle_func_min)[0], 0]
            interval /= 4.
            continue

    # numerical optimization method attempts

    # x_guess, y_guess, z_guess = centroid1 - centroid2
    # print(centroid1 - centroid2)
    # x_ext, y_ext, z_ext = [(-1*min(mask1.shape[i], mask2.shape[i]), min(mask1.shape[i], mask2.shape[i])) for i in range(3)]
    # res = optimize.minimize(optim_func, x0=[0., 0., 0., 0.], args=(mask1_crop, mask2_crop), method='BFGS', options={'eps':[1., 1., 1., 0.25], 'gtol':100.})
    # res = optimize.basinhopping(optim_func, x0=[0., 0., 0., 0.], stepsize=5, niter=25, minimizer_kwargs={'args':(mask1_crop, mask2_crop), 'method':'BFGS', 'options':{'eps':[1., 1., 1., 0.25], 'gtol':100., 'maxiter':200}})
    # res = optimize.minimize(optim_func, x0=[x_guess, y_guess, z_guess, 0], args=(mask1, mask2), bounds=[x_ext, y_ext, z_ext, (-180, 180)])
    # res = optimize.minimize(optim_func, x0=[x_guess, y_guess, z_guess, 0], args=(mask1, mask2), method='nelder-mead')
    # transform = res.x
    # transform[:3] += centroid1 - centroid2

    transform = centroid1 - centroid2
    transform[2] += z_opt
    transform = np.append(transform, angle_opt)
    mask2_tf = transform_mask(transform, mask2)
    aligned_masks = mask1 + 2*mask2_tf
    z_overlap = [True if np.any(aligned_masks[:,:,z] == 3) else False for z in range(aligned_masks.shape[2])]
    zrange = (np.amin(np.argwhere(z_overlap)), np.amax(np.argwhere(z_overlap)) + 1)
    overlap = float(np.count_nonzero(aligned_masks[:,:,zrange[0]:zrange[1]] == 3))/float(np.count_nonzero(aligned_masks[:,:,zrange[0]:zrange[1]] > 0))
    
    print(transform)
    print(overlap)
    print(zrange)

    return transform, overlap, zrange, centroid1, centroid2, rad


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
parser.add_argument('-b', '--blobs', type=str, nargs=2, help='Binary mask TIFF files highlighting spots identified by Allen Segmenter. Expected paths used by default.', default=None)
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
p_blobs = args['blobs']

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

if not p_blobs:
    p_blobs = []
    for g in genes:
        p_blobs.append(os.path.join(outdir, 'masks', img_name + '_mask^' + g + '.tiff'))

if img_name[0].isdigit():
    is_old_image = True

img_dapi = su.normalize_image(img[-1,:,:,:])
imgs_rna = [su.normalize_image(img[i,:,:,:]) for i in range(len(genes))]
imgs_rna_unnormed = [img[i,:,:,:] for i in range(len(genes))]

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

# open masks
blob_masks = []
for p in p_blobs:
    blob_masks.append(tifffile.imread(p).transpose(1,2,0))


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
for n in range(2):  # ~ 1 um in z
    img_nuc_dilated = morphology.binary_dilation(img_nuc_dilated)  # expand x, y, z
for n in range(20):  # ~ 1 um in x, y
    for z in range(img_nuc_dilated.shape[2]):
        img_nuc_dilated[:,:,z] = morphology.binary_dilation(img_nuc_dilated[:,:,z])  # expand x, y only

region_peri = np.where(np.logical_and(img_nuc_dilated > region_nuc, np.logical_or(img_fiber_only.astype(bool), nuclei_binary.astype(bool))), 1, 0)
region_cyt = np.where(img_fiber_only > img_nuc_dilated, 1, 0)
regions = {'nuc':region_nuc, 'cyt':region_cyt, 'peri':region_peri}

region_nuc_peri = np.where(region_peri + region_nuc >= 1, 1, 0)
labeled_nuc_peri, n_nucperi = morphology.label(region_nuc_peri, return_num=True)

if should_plot:
    su.animate_zstacks([region_nuc, region_peri, region_cyt, region_cyt + 2*region_peri + 3*region_nuc], titles=['nuclear', 'perinuclear', 'sarcoplasmic', 'all compartments'], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], gif_name=os.path.join(outdir, 'anim', img_name+'_regions.gif'))


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
        

    #--  FIND DISTRIBUTIONS OF SPOT INTENSITIES IN COMPARTMENTS  --------------#
    
    intensities_cyt = []
    intensities_peri = []
    sizes_cyt = []
    sizes_peri = []

    blobs_cyt = np.zeros_like(blob_masks[chan])
    blobs_peri = np.zeros_like(blob_masks[chan])

    # norm_factor = np.median(imgs_rna_unnormed[chan][img_fiber_only.astype(bool)])
    
    labeled_blobs, n_blobs = morphology.label(morphology.remove_small_objects(blob_masks[chan], min_size=8), connectivity=1, return_num=True)
    print('\n' + str(n_blobs) + ' blobs found in Allen Segmenter mask.')

    # # create thin shell perinuclear mask
    # nuc_dil = morphology.binary_dilation(nuclei_binary)  # expand x, y, z
    # region_peri_shell = np.where(np.logical_and(nuc_dil > nuclei_binary, np.logical_or(img_fiber_only.astype(bool), nuclei_binary.astype(bool))), 1, 0)

    # region_peri_bool = region_peri_shell.astype(bool)
    region_peri_bool = region_nuc_peri.astype(bool)
    # region_peri_bool = region_peri.astype(bool)
    region_cyt_bool = region_cyt.astype(bool)

    # classify spots and get intensities
    print('Classifying blobs and calculating intensities... 0%', end='')
    intensity_map = np.full_like(labeled_blobs, np.nan, dtype=np.float32)
    # intensity_map = np.zeros_like(labeled_blobs, dtype=np.float32)
    for l in range(1, n_blobs+1):
        this_blob = (labeled_blobs == l)
        intens = np.sum(imgs_rna_unnormed[chan][this_blob])
        intensity_map[this_blob] = np.log(intens)
        # intensity_map[this_blob] = intens
        # intensity_map[this_blob] = np.random.random() # label each spot with different color

        # categorize spots by compartment
        # if np.all(region_peri_bool[this_blob]):
        #     # blob is fully enclosed in perinuclear region
        #     blobs_peri = blobs_peri + this_blob
        #     intensities_peri.append(np.sum(imgs_rna[chan][this_blob]))
        #     sizes_peri.append(np.count_nonzero(this_blob))
        if np.all(region_peri_bool[this_blob]):
            # blob overlaps with perinuclear shell
            blobs_peri = blobs_peri + this_blob
            intensities_peri.append(intens)
            sizes_peri.append(np.count_nonzero(this_blob))
        elif np.all(region_cyt_bool[this_blob]):
            # blob is fully enclosed in sarcoplasm
            blobs_cyt = blobs_cyt + this_blob
            intensities_cyt.append(intens)
            sizes_cyt.append(np.count_nonzero(this_blob))
        print('\rClassifying blobs and calculating intensities... ' + str(round(100.*l/n_blobs)) + '%', end='')
    
    print('\nBlob classification complete.')

    # # get spot intensities in sarcoplasm
    # labeled_blobs_cyt, n_cyt = morphology.label(blobs_cyt, return_num=True)
    # for l in range(1, n_cyt):
    #     this_blob = (labeled_blobs_cyt == l)
    #     intensities_cyt.append(np.sum(imgs_rna[chan][this_blob]))
    #     sizes_cyt.append(np.count_nonzero(this_blob))

    # # get spot intensities in perinuclear space
    # labeled_blobs_peri, n_peri = morphology.label(blobs_peri, return_num=True)
    # for l in range(1, n_peri):
    #     this_blob = (labeled_blobs_peri == l)
    #     intensities_peri.append(np.sum(imgs_rna[chan][this_blob]))
    #     sizes_peri.append(np.count_nonzero(this_blob))

    # for l in range(1, n_blobs+1):
    #     this_blob = (labeled_blobs == l)
    #     intens = np.sum(imgs_rna[chan][this_blob])
    #     intensity_map[this_blob] = np.log(intens)
    #     # intensity_map[this_blob] = intens
    #     # intensity_map[this_blob] = np.random.random() # label each spot with different color
    
    # draw animation of spot intensities
    both_blobs = blobs_cyt + 2*blobs_peri
    cm_magma = cm.get_cmap('magma')
    cm_magma.set_bad(color='k')
    su.animate_zstacks([imgs_rna[chan], both_blobs, intensity_map], cmaps=['binary_r', 'gnuplot2', cm_magma], vmin=[0., 0., np.nanmin(intensity_map)], vmax=[np.amax(imgs_rna[chan])/3., 2., np.nanmax(intensity_map)], gif_name=os.path.join(outdir, 'anim', img_name + '_blob_intensities^' + gene + '.gif'))

    # compare distributions of perinuclear vs cytoplasmic spot intensities
    mean_peri = np.mean(intensities_peri)
    err_peri = np.std(intensities_peri)
    mean_cyt = np.mean(intensities_cyt)
    err_cyt = np.std(intensities_cyt)
    median_peri = np.median(intensities_peri)
    median_cyt = np.median(intensities_cyt)

    # bootstrap to find 95% CI of median
    bs_peri = bs.bootstrap(np.array(intensities_peri), stat_func=bs_stats.median)
    lower_peri = bs_peri.lower_bound
    upper_peri = bs_peri.upper_bound
    bs_cyt = bs.bootstrap(np.array(intensities_cyt), stat_func=bs_stats.median)
    lower_cyt = bs_cyt.lower_bound
    upper_cyt = bs_cyt.upper_bound

    # report statistics
    print('\nmean cyt.:  ' + str(mean_cyt) + ' +/- ' + str(err_cyt))
    print('mean peri.: ' + str(mean_peri) + ' +/- ' + str(err_peri))

    print('\nmedian cyt.:  ' + str(median_cyt) + ' (' + str(lower_cyt) + ', ' + str(upper_cyt) + ')')
    print('median peri.:  ' + str(median_peri) + ' (' + str(lower_peri) + ', ' + str(upper_peri) + ')')
    
    if should_plot:
        fig, ax = plt.subplots()
        ax.hist(intensities_peri, bins=50, range=(0,100))
        # ax.set_xlim([0, 200])
        ax.set_xlabel('Spot intensity')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(outdir, 'plots', img_name + '_intens_hist_peri^' + gene + '.png'), dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.hist(intensities_cyt, bins=50, range=(0,100))
        # ax.set_xlim([0, 200])
        ax.set_xlabel('Spot intensity')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(outdir, 'plots', img_name + '_intens_hist_cyt^' + gene + '.png'), dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.hist(intensities_peri, bins=np.logspace(-0.5,4,40))
        ax.set_xscale('log')
        ax.set_xlabel('Spot intensity')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(outdir, 'plots', img_name + '_intens_hist_peri_log^' + gene + '.png'), dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.hist(intensities_cyt, bins=np.logspace(-0.5,4,40))
        ax.set_xscale('log')
        ax.set_xlabel('Spot intensity')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(outdir, 'plots', img_name + '_intens_hist_cyt_log^' + gene + '.png'), dpi=300)
        plt.close()

    with open(os.path.join(outdir, 'spot_intensities', img_name + '_intens_peri^' + gene + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([intensities_peri])

    with open(os.path.join(outdir, 'spot_intensities', img_name + '_intens_cyt^' + gene + '.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows([intensities_cyt])
    
    '''

    df = pd.DataFrame([['peri', intens] for intens in intensities_peri] + [['cyt', intens] for intens in intensities_cyt], columns=['compartment', 'intensity'])

    # swarm plot
    ax = sns.swarmplot(x='compartment', y='intensity', data=df)
    ax.set_ylabel('FISH spot intensity')
    plt.tight_layout()
    plt.savefig('intensity_swarm_' + gene + '_linear.pdf', dpi=300)
    # plt.show()
    plt.close()

    fig, log_ax = plt.subplots()
    log_ax.set_yscale("log") # log first
    # log_ax.set_ylim(4E-6, 2E0)
    sns.swarmplot(x='compartment', y='intensity', data=df, ax=log_ax)
    plt.tight_layout()
    plt.savefig('intensity_swarm_' + gene + '_log.pdf', dpi=300)
    # plt.show()
    plt.close()
    '''

'''
#--  PERINUCLEAR GRANULE ANALYSIS  --------------------------------------------#

# iterate over all nuclei and compare transcripts 1 and 2
init = False
for i in range(n_nucperi):
    t1 = imgs_rna[0][labeled_nuc_peri == i+1]
    t2 = imgs_rna[1][labeled_nuc_peri == i+1]

    heatmap, xedges, yedges = np.histogram2d(np.log10(t1), np.log10(t2), range=[[-3, 0], [-3, 0]], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    if not init:
        avg_heatmap = deepcopy(heatmap)
        init = True
    else:
        avg_heatmap += heatmap

    fig, ax = plt.subplots()
    cmap = cm.get_cmap('inferno')
    ax.set_facecolor(cmap(0))
    ax.imshow(heatmap.T, cmap='inferno', extent=extent, origin='lower', norm=LogNorm())
    ax.set_xlabel('Gene 1 (FISH intensity)')
    ax.set_ylabel('Gene 2 (FISH intensity)')
    plt.savefig('anim/test' + str(i) + '.png', dpi=600)
    plt.close()

fig, ax = plt.subplots()
cmap = cm.get_cmap('inferno')
ax.set_facecolor(cmap(0))
ax.imshow(avg_heatmap.T, cmap='inferno', extent=extent, origin='lower', norm=LogNorm())
ax.set_xlabel('Gene 1 (FISH intensity)')
ax.set_ylabel('Gene 2 (FISH intensity)')
plt.savefig('anim/average_heatmap.png', dpi=600)
plt.close()

# build a null distribution by mixing transcript localizations between nuclei
border_pix = np.ones_like(labeled_nuc_peri)
border_pix[1:-1, 1:-1] = 0
for i in range(n_nucperi):
    for j in range(n_nucperi):
        if i >= j:
            continue
        elif np.any(np.logical_and(labeled_nuc_peri == i+1, border_pix.astype(bool))):
            # nucleus i overlaps image boundary
            continue
        elif np.any(np.logical_and(labeled_nuc_peri == j+1, border_pix.astype(bool))):
            # nucleus j overlaps image boundary
            continue
        
        print(i, j)
        nuc_mask_i = np.where(labeled_nuc_peri == i+1, 1, 0)
        nuc_mask_j = np.where(labeled_nuc_peri == j+1, 1, 0)
        transform, overlap, zrange, centroid, centroid2, box_dim = align_masks(nuc_mask_i, nuc_mask_j)
        # nuc_mask_j_tf = transform_mask(transform, nuc_mask_j)
        # aligned_masks = nuc_mask_i + 2*nuc_mask_j_tf
        # su.animate_zstacks([nuc_mask_i, nuc_mask_j, nuc_mask_j_tf, aligned_masks], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], vmax=[1,1,1,3], gif_name='anim/nuclei_alignment_' + str(i) + '_' + str(j) + '.gif')

        # print('done, waiting')
        # input()

        img_rna_2j = transform_image(transform, centroid2, imgs_rna[1])
        img_rna_1i_crop = imgs_rna[0][centroid[0]-box_dim:centroid[0]+box_dim, centroid[1]-box_dim:centroid[1]+box_dim, zrange[0]:zrange[1]]
        img_rna_2j_crop = img_rna_2j[centroid[0]-box_dim:centroid[0]+box_dim, centroid[1]-box_dim:centroid[1]+box_dim, zrange[0]:zrange[1]]

        su.animate_zstacks([img_rna_1i_crop, img_rna_2j_crop], titles=['nuc_1, rna_1', 'nuc_2, rna_2'], cmaps=['binary_r']*2, gif_name='anim/test_align_rna_' + str(i) + '_' + str(j) + '.gif')
'''