# python 3.6.5, HiPerGator
"""
Goals:
1. open CZI image and separate channels
2. segment muscle fiber and nuclei by automated threshold selection
3. detect HCR FISH spots by Laplacian of Gaussian
4. export nuclei and fiber segmentations to .npy files
"""

import argparse
import csv
import numpy as np
import os

# custom libraries
import scope_utils3 as su
import muscle_fish as mf


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
parser.add_argument('--transpose', action='store_true', help='Transpose fiber images.')

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
transpose = args['transpose']

if t_dapi:
    t_dapi = float(t_dapi)
if t_fiber and t_fiber != 'last':
    t_fiber = float(t_fiber)
if t_spot1 is not None:
    t_spot1 = float(t_spot1)
if t_spot2 is not None:
    t_spot2 = float(t_spot2)
t_snr = [t_spot1, t_spot2]


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

# determine image filetype and open
print('Analyzing `' + p_img + '`...\n')
print('Genes: ' + ', '.join(genes))

img, img_name, mtree = mf.open_image(p_img)

if transpose:
    img = img.transpose(0, 2, 1, 3)

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

with open(os.path.join(outdir, img_name+'_dims.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows([['x', dims['X']], ['y', dims['Y']], ['z', dims['Z']]])

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

print('fiber vol: ' + str(voxel_vol*np.count_nonzero(img_fiber_only)))

np.save(os.path.join(outdir, img_name+'_fiber.npy'), img_fiber_only)

if should_plot:
    su.animate_zstacks([np.clip(img_rna_average, a_min=0, a_max=np.amax(img_rna_average)/30.), img_fiber_only], titles=['average of RNAs', 'fiber prediction'], gif_name=os.path.join(outdir, 'anim', img_name+'_fiber.gif'))


#--  SEGMENTATION OF NUCLEI  --------------------------------------------------#

nuclei_labeled = mf.threshold_nuclei(img_dapi, t_dapi=t_dapi, fiber_mask=img_fiber_only, verbose=True)
nuclei_binary = np.where(nuclei_labeled > 0, 1, 0)

np.save(os.path.join(outdir, img_name+'_nuclei.npy'), nuclei_binary)

if should_plot:
    su.animate_zstacks([img_dapi, nuclei_binary], titles=['DAPI', 'nuclei'], cmaps=['binary_r', 'binary_r'], gif_name=os.path.join(outdir, 'anim', img_name+'_nuclei.gif'))


#-- RNA DETECTION AND ANALYSIS  -----------------------------------------------#

# iterate over RNAs in image
output_table = []
for chan, gene in enumerate(genes):
    print('--------------------------------')

    if gene == 'ribo':
        # can't resolve spots, skip this gene
        print('Skipping gene `' + gene + '`: cannot resolve spots.')
        continue

    print('Analyzing ' + gene + ' gene (' + str(chan+1) + '/' + str(len(genes)) + ')')


    #--  FISH SPOT DETECTION  -------------------------------------------------#

    print('Finding FISH spots...')
    spots_masked, spot_data = mf.find_spots_snrfilter(imgs_rna[chan], sigma=2, snr=t_snr[chan], t_spot=0.025, mask=img_fiber_only, imgprefix=img_name)

    print(str(spots_masked.shape[0]) + ' spots detected within fiber.')
    output_table.append([gene, len(spots_masked)])

with open(os.path.join(outdir, img_name+'_rna_counts.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(output_table)

