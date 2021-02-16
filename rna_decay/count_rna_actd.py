# python 2.7.14, HiPerGator
"""
Goals:

1. transcript density per volume of fiber
2. ratio of nuclear to cytoplasmic transcripts
"""
import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
# from czifile import CziFile
from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import feature, exposure, filters, morphology
from xml.etree import ElementTree
import argparse
import csv
import numpy as np
import os
import scipy.ndimage as ndi
import scope_utils

#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

def update_z(frame):
    im_xy.set_data(img_rna[:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_facecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_edge(frame):
    im_xy.set_data(img_rna[:,:,frame])
    colors = []
    for spot in spots_masked:
        alpha = scope_utils.gauss_1d(frame, 1., spot[2], spot[3])
        colors.append([1., 0., 0., alpha])
    spots_xy.set_edgecolor(colors)
    return im_xy, spots_xy  # this return structure is a requirement for the FuncAnimation() callable

def update_z_compartments(frame):
    im_xy.set_data(img_rna[:,:,frame])
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


#--  COMMAND LINE ARGUMENTS  --------------------------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('czi', type=str, nargs=1, help='Path to .CZI image file.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('gene', type=str, nargs=1, help='Gene name.')
parser.add_argument('-c', '--channel', help='Wavelength of channel containing FISH signal (default first channel)', default=None)
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('-f', '--fiber-threshold', help='Threshold value for fiber segmentation (default: Li\'s method)', default=None)
parser.add_argument('-s', '--spot-threshold', help='Threshold value for spot detection in FISH channel (default: 0.018)', default=0.018)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations')

# parse arguments
args = vars(parser.parse_args())
p_img = args['czi'][0]
outdir = args['outdir'][0]
gene = args['gene'][0]
fish_channel = args['channel']
t_dapi = args['dapi_threshold']
t_fiber = args['fiber_threshold']
t_spot = float(args['spot_threshold'])
should_plot = args['plot']

# flag declarations
use_last_fiber_mask = False

if t_dapi:
    t_dapi = float(t_dapi)
if t_fiber:
    if t_fiber == 'last':
        use_last_fiber_mask = True
    else:
        t_fiber = float(t_fiber)


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

print 'Analyzing `' + p_img + '`...\n'
print 'Gene: ' + gene

# determine image filetype and open
if p_img.lower().endswith('.czi'):
    img_name = os.path.splitext(os.path.basename(p_img))[0]

    # open CZI format using `czifile` library
    from czifile import CziFile
    with CziFile(p_img) as czi_file:
        meta = czi_file.metadata()
        mtree = ElementTree.fromstring(meta)
        img_czi = czi_file.asarray()
    img = img_czi[0,0,:,0,:,:,:,0].transpose(0,2,3,1)  # c, x, y, z

elif any([p_img.lower().endswith(ext) for ext in ['.ome.tiff', '.ome.tif']]):
    img_name = os.path.splitext(os.path.splitext(os.path.basename(p_img))[0])[0]

    # open OME-TIFF format using `bioformats` library (requires Java)
    import javabridge
    import bioformats
    javabridge.start_vm(class_path=bioformats.JARS)

    # iterate over z-stack until end of file
    slices = []
    for z in count():
        try:
            slice = bioformats.load_image(p_img, z=z)
            slices.append(slice)
        except javabridge.jutil.JavaException:
            # final z-slice was read, stop reading
            javabridge.kill_vm()
            break

    img_ome = np.stack(slices)
    img = img_ome.transpose(3,1,2,0)  # c, x, y, z

    # look for metadata .XML file with same filename
    p_meta = os.path.splitext(os.path.splitext(p_img)[0])[0] + '.xml'
    try:
        mtree = ElementTree.parse(p_meta)
    except IOError:
        # metadata file not found
        raise IOError('CZI metadata XML not found at expected path "' + p_meta + '" (required for OME-TIFF)')

else:
    raise ValueError('Image filetype not recognized. Allowed:  .CZI, .OME.TIFF')

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

print '\nTracks: ' + ', '.join(map(str, tracks))

# split channels
img_dapi = scope_utils.normalize_image(img[-1,:,:,:])
print 'DAPI: ' + str(tracks[-1])

if fish_channel:
    try:
        img_rna = scope_utils.normalize_image(img[tracks.index(int(fish_channel)),:,:,:])
        print 'FISH: ' + fish_channel
    except ValueError:
        raise ValueError('FISH channel not recognized. Must match one of the above tracks.')
else:
    img_rna = scope_utils.normalize_image(img[0,:,:,:])
    print 'FISH: ' + str(tracks[0])

if should_plot:
    scope_utils.animate_zstacks([img_dapi, np.clip(img_rna, a_min=0, a_max=np.amax(img_rna)/5.)], titles=['DAPI', 'FISH, gain=5'], gif_name=os.path.join(outdir, 'anim', img_name+'_channels^'+gene+'.gif'))


#--  FIBER SEGMENTATION  ------------------------------------------------------#

# segment fiber using information from all RNA channels
print '\nSegmenting fiber from background...'
imgs_rna = np.array([im for im in img[:-1]])
img_rna_average = scope_utils.normalize_image(np.mean(np.array(imgs_rna), axis=0))
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

print 'Fiber thresholds:'
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

# calculate volume of fiber
volume = np.sum(img_fiber_only) * voxel_vol
print '\nfiber = ' + str(float(100.*np.sum(img_fiber_only))/img_fiber_only.size) + '% of FOV'
print 'volume = ' + str(volume) + ' um^3'

if should_plot:
    scope_utils.animate_zstacks([np.clip(img_rna, a_min=0, a_max=np.amax(img_rna)/5.), img_fiber_binary, img_fiber_only], titles=['FISH, gain=5', 'threshold_li', 'fiber prediction'], gif_name=os.path.join(outdir, 'anim', img_name+'_fiber^'+gene+'.gif'))


#--  NUCLEI SEGMENTATION  -----------------------------------------------------#

# threshold nuclei using Otsu's method
print '\nSegmenting nuclei...'
img_dapi_blurxy = filters.gaussian(img_dapi, (20, 20, 2))
if not t_dapi:
    thresh_dapi = filters.threshold_otsu(img_dapi_blurxy)*0.75
    print 'DAPI threshold = ' + str(thresh_dapi)
    img_dapi_binary = np.where(img_dapi > thresh_dapi, 1, 0)
else:
    # override by optional script argument
    print 'DAPI threshold = ' + str(t_dapi)
    img_dapi_binary = np.where(img_dapi > t_dapi, 1, 0)

img_dapi_binary = morphology.remove_small_objects(img_dapi_binary.astype(bool), min_size=2048)  # without this, 1-pixel noise becomes huge abberant perinuclear region
img_dapi_binary = morphology.remove_small_holes(img_dapi_binary.astype(bool), min_size=2048)

# remove nuclei that are not part of main fiber segment
print 'Removing nuclei that are not connected to main fiber segment...'
nuclei_labeled, n_nuc = morphology.label(img_dapi_binary, return_num=True)
for i in range(1, n_nuc+1):
    overlap = np.logical_and(np.where(nuclei_labeled == i, True, False), img_fiber_only.astype(bool))
    if np.count_nonzero(overlap) == 0:
        nuclei_labeled[nuclei_labeled == i] = 0


#--  SEGMENTATION OF NUCLEAR, PERINUCLEAR, AND SARCOPLASMIC COMPARTMENTS  -----#

print '\nDefining nuclear compartment...'
region_nuc = np.where(nuclei_labeled > 0, 1, 0)
for n in range(1):  # ~ 0.5 um in z
    region_nuc = morphology.binary_erosion(region_nuc)  # contract x, y, z
for n in range(10):  # ~ 0.5 um in x, y
    for z in range(region_nuc.shape[2]):
        region_nuc[:,:,z] = morphology.binary_erosion(region_nuc[:,:,z])  # contract x, y only

print 'Defining perinuclear compartment...'
img_nuc_dilated = np.where(nuclei_labeled > 0, 1, 0)
for n in range(4):  # ~ 2 um in z
    img_nuc_dilated = morphology.binary_dilation(img_nuc_dilated)  # expand x, y, z
for n in range(40):  # ~ 2 um in x, y
    for z in range(img_nuc_dilated.shape[2]):
        img_nuc_dilated[:,:,z] = morphology.binary_dilation(img_nuc_dilated[:,:,z])  # expand x, y only

region_peri = np.where(np.logical_and(img_nuc_dilated > region_nuc, np.logical_or(img_fiber_only.astype(bool), img_dapi_binary.astype(bool))), 1, 0)

print 'Defining cytoplasmic compartment...'
region_cyt = np.where(img_fiber_only > img_nuc_dilated, 1, 0)
regions = {'nuc':region_nuc, 'cyt':region_cyt, 'peri':region_peri}

if should_plot:
    scope_utils.animate_zstacks([region_nuc, region_peri, region_cyt, region_cyt + 2*region_peri + 3*region_nuc], titles=['nuclear', 'perinuclear', 'sarcoplasmic', 'all compartments'], cmaps=['binary_r', 'binary_r', 'binary_r', 'gnuplot2'], gif_name=os.path.join(outdir, 'anim', img_name+'_regions^'+gene+'.gif'))


#--  FISH SPOT DETECTION  -----------------------------------------------------#

print 'Finding FISH spots...'
spots = feature.blob_log(img_rna, 1., 1., num_sigma=1, threshold=t_spot)
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

print 'Categorizing FISH spots by compartment...'
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
print ''
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
