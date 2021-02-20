# python 3.6.5
"""
Goals:
1. open CZI image and separate channels
2. segment muscle fiber and nuclei by automated threshold selection
3. detect HCR FISH spots by Laplacian of Gaussian
4. calculate transcript density within nuclear, perinuclear, and cytoplasmic compartments
"""
import matplotlib
matplotlib.use('Agg')  # for plotting on cluster

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import morphology
import argparse
import csv
import numpy as np
import os

# custom libraries
import scope_utils3 as su
import muscle_fish as mf


#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

def update_z(frame):
    '''Frame updater for FuncAnimation.
    '''
    im_xy.set_data(img_rna[:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_facecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_edge(frame):
    '''Frame updater for FuncAnimation.
    '''
    im_xy.set_data(img_rna[:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = su.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_edgecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_compartments(frame):
    '''Frame updater for FuncAnimation.
    '''
    im_xy.set_data(img_rna[:,:,frame])
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
parser.add_argument('czi', type=str, nargs=1, help='Path to .CZI image file.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('gene', type=str, nargs=1, help='Gene name.')
parser.add_argument('-c', '--channel', help='Wavelength of channel containing FISH signal (default first channel)', default=None)
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('-f', '--fiber-threshold', help='Threshold value for fiber segmentation (default: Li\'s method)', default=None)
parser.add_argument('-s', '--snr', help='Signal-to-noise threshold for spot detection in FISH channel (default: automatic)', default=None)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations')

# parse arguments
args = vars(parser.parse_args())
p_img = args['czi'][0]
outdir = args['outdir'][0]
gene = args['gene'][0]
fish_channel = args['channel']
t_dapi = args['dapi_threshold']
t_fiber = args['fiber_threshold']
t_snr = args['snr']
should_plot = args['plot']

# flag declarations
use_last_fiber_mask = False

if t_dapi:
    t_dapi = float(t_dapi)
if t_fiber and t_fiber != 'last':
    t_fiber = float(t_fiber)
if t_snr is not None:
    t_snr = float(t_snr)


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

print('Analyzing `' + p_img + '`...\n')
print('Gene: ' + gene)

img, img_name, mtree = mf.open_image(p_img)

# get voxel dimensions
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

print('\nTracks: ' + ', '.join(map(str, tracks)))

# split channels
img_dapi = su.normalize_image(img[-1,:,:,:])
print('DAPI: ' + str(tracks[-1]))

if fish_channel:
    try:
        img_rna = su.normalize_image(img[tracks.index(int(fish_channel)),:,:,:])
        print('FISH: ' + fish_channel)
    except ValueError:
        raise ValueError('FISH channel not recognized. Must match one of the above tracks.')
else:
    img_rna = su.normalize_image(img[0,:,:,:])
    print('FISH: ' + str(tracks[0]))

if should_plot:
    su.animate_zstacks([img_dapi, np.clip(img_rna, a_min=0, a_max=np.amax(img_rna)/5.)], titles=['DAPI', 'FISH, gain=5'], gif_name=os.path.join(outdir, 'anim', img_name+'_channels.gif'))


#--  FIBER SEGMENTATION  ------------------------------------------------------#

print('\nSegmenting fiber from background...')
img_fiber_only = mf.threshold_fiber(img_rna, t_fiber=t_fiber, verbose=True)

# calculate volume of fiber
volume = np.sum(img_fiber_only) * voxel_vol
print('\nfiber = ' + str(float(100.*np.sum(img_fiber_only))/img_fiber_only.size) + '% of FOV')
print('volume = ' + str(volume) + ' um^3')

if should_plot:
    su.animate_zstacks([np.clip(img_rna, a_min=0, a_max=np.amax(img_rna)/5.), img_fiber_only], titles=['FISH, gain=5', 'fiber prediction'], gif_name=os.path.join(outdir, 'anim', img_name+'_fiber.gif'))


#--  SEGMENTATION OF NUCLEAR, PERINUCLEAR, AND SARCOPLASMIC COMPARTMENTS  -----#

nuclei_labeled = mf.threshold_nuclei(img_dapi, t_dapi=t_dapi, fiber_mask=img_fiber_only, verbose=True)
nuclei_binary = np.where(nuclei_labeled > 0, 1, 0)

print('\nDefining nuclear compartment...')
region_nuc = np.where(nuclei_labeled > 0, 1, 0)
for n in range(1):  # ~ 0.5 um in z
    region_nuc = morphology.binary_erosion(region_nuc)  # contract x, y, z
for n in range(10):  # ~ 0.5 um in x, y
    for z in range(region_nuc.shape[2]):
        region_nuc[:,:,z] = morphology.binary_erosion(region_nuc[:,:,z])  # contract x, y only

print('Defining perinuclear compartment...')
img_nuc_dilated = np.where(nuclei_labeled > 0, 1, 0)
for n in range(4):  # ~ 2 um in z
    img_nuc_dilated = morphology.binary_dilation(img_nuc_dilated)  # expand x, y, z
for n in range(40):  # ~ 2 um in x, y
    for z in range(img_nuc_dilated.shape[2]):
        img_nuc_dilated[:,:,z] = morphology.binary_dilation(img_nuc_dilated[:,:,z])  # expand x, y only

region_peri = np.where(np.logical_and(img_nuc_dilated > region_nuc, np.logical_or(img_fiber_only.astype(bool), nuclei_binary.astype(bool))), 1, 0)

print('Defining cytoplasmic compartment...')
region_cyt = np.where(img_fiber_only > img_nuc_dilated, 1, 0)
regions = {'nuc':region_nuc, 'cyt':region_cyt, 'peri':region_peri}

if should_plot:
    su.animate_zstacks([region_nuc, region_peri, region_cyt, region_cyt + 2*region_peri + 3*region_nuc], titles=['nuclear', 'perinuclear', 'sarcoplasmic', 'all compartments'], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], gif_name=os.path.join(outdir, 'anim', img_name+'_regions.gif'))


#--  FISH SPOT DETECTION  -----------------------------------------------------#

img_rna_corr = mf.fix_bleaching(img_rna, mask=img_fiber_only, draw=False, imgprefix=img_name)

print('Finding FISH spots...')
spots_masked, spot_data = mf.find_spots_snrfilter(img_rna_corr, sigma=2, snr=t_snr, t_spot=0.025, mask=img_fiber_only, imgprefix=img_name)
print(str(spots_masked.shape[0]) + ' spots detected within fiber.')

# write spot intensities to file
with open(img_name + '_spot_data_intensities.csv', 'w') as spot_file:
    writer = csv.writer(spot_file)
    writer.writerows(spot_data)

# animate spot detection
if should_plot:
    fig, ax = plt.subplots(1,2)
    im_xy = ax[0].imshow(img_rna[:,:,0], vmin=np.amin(img_rna), vmax=np.amax(img_rna)/10., cmap='binary_r')
    ax[1].set_facecolor('k')
    spots_xy = ax[1].scatter(spots_masked[:,1], -1.*spots_masked[:,0], s=5, c='k', edgecolors='none')
    anim = FuncAnimation(fig, update_z, frames=img_rna.shape[2], interval=100, blit=False)
    anim.save(os.path.join(outdir, 'anim', img_name + '_spots^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
    # plt.show()
    plt.close()

if should_plot:
    fig, ax = plt.subplots()
    im_xy = ax.imshow(img_rna[:,:,0], vmin=np.amin(img_rna), vmax=np.amax(img_rna)/10., cmap='binary_r')
    spots_xy = ax.scatter(spots_masked[:,1], spots_masked[:,0], s=12, c='none', edgecolors='k')
    anim = FuncAnimation(fig, update_z_edge, frames=img_rna.shape[2], interval=100, blit=False)
    anim.save(os.path.join(outdir, 'anim', img_name + '_spot_overlay^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
    # plt.show()
    plt.close()


#--  ASSIGNMENT OF SPOTS TO COMPARTMENTS  -------------------------------------#

print('Categorizing FISH spots by compartment...')
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
        # spot was not assigned to a compartment
        # we have not observed this to occur
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


#--  OUTPUT FILE OPERATIONS  --------------------------------------------------#

# output results to stdout
print('')
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
    im_xy = ax[0].imshow(img_rna[:,:,0], vmin=np.amin(img_rna), vmax=np.amax(img_rna)/10., cmap='binary_r')
    ax[1].set_facecolor('k')
    spots_nuc_xy = ax[1].scatter(spots_by_region['nuc'][:,1], -1.*spots_by_region['nuc'][:,0], s=5, c='k', edgecolors='none')
    spots_cyt_xy = ax[1].scatter(spots_by_region['cyt'][:,1], -1.*spots_by_region['cyt'][:,0], s=5, c='k', edgecolors='none')
    spots_peri_xy = ax[1].scatter(spots_by_region['peri'][:,1], -1.*spots_by_region['peri'][:,0], s=5, c='k', edgecolors='none')
    ax[1].set_aspect('equal')
    anim = FuncAnimation(fig, update_z_compartments, frames=img_rna.shape[2], interval=100, blit=False)
    anim.save(os.path.join(outdir, 'anim', img_name + '_rna_compartments^' + gene + '.gif'), writer='imagemagick', fps=8, dpi=300)
    # plt.show()
    plt.close()
