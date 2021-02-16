from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_3d
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicsimageio.writers import OmeTiffWriter
from scipy.ndimage import distance_transform_edt
from skimage import morphology, feature, measure
import argparse
import numpy as np
import os
# import tifffile

# custom libraries
import scope_utils3 as su
import muscle_fish as mf

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('img', type=str, nargs=1, help='Path to image file (CZI or OME-TIFF).')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('genes', type=str, nargs='*', help='Gene names, in order of appearance in image file.')
parser.add_argument('-1', '--threshold1', help='Threshold for detection of RNA spots for gene 1.', default=0.02)
parser.add_argument('-2', '--threshold2', help='Threshold for detection of RNA spots for gene 2.', default=0.02)

# parse arguments
args = vars(parser.parse_args())
p_img = args['img'][0]
outdir = args['outdir'][0]
genes = args['genes']
t_spot1 = float(args['threshold1'])
t_spot2 = float(args['threshold2'])

t_spots = [t_spot1, t_spot2]


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

# determine image filetype and open
print('Analyzing `' + p_img + '`...\n')
print('Genes: ' + ', '.join(genes))

img, img_name, mtree = mf.open_image(p_img, z_first=True)
imgs_rna = [su.normalize_image(img[i,:,:,:]) for i in range(len(genes))]

# get voxel dimensions
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
voxel_vol = dims['X'] * dims['Y'] * dims['Z']  # [um^3]
dims_xyz = np.array([dims['X'], dims['Y'], dims['Z']])


#--  SEGMENT SPOTS  -----------------------------------------------------------#
for i, g in enumerate(genes):
    print('\nProcessing gene ' + g + '...\n')
    print('Smoothing image...')
    img_rna_smooth = image_smoothing_gaussian_3d(imgs_rna[i], sigma=1)
    s3_param = [[1, 0.75*t_spots[i]], [2, 0.75*t_spots[i]]]
    print('Identifying spots...')
    bw = dot_3d_wrapper(img_rna_smooth, s3_param)

    # watershed
    minArea = 4
    Mask = morphology.remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False) 
    labeled_mask = measure.label(Mask)
    
    print('Performing watershed segmentation...')
    peaks = feature.peak_local_max(imgs_rna[i],labels=labeled_mask, min_distance=2, indices=False)
    Seed = morphology.binary_dilation(peaks, selem=morphology.ball(1))
    Watershed_Map = -1*distance_transform_edt(bw)
    seg = morphology.watershed(Watershed_Map, measure.label(Seed), mask=Mask, watershed_line=True)
    seg = morphology.remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)

    print('Exporting mask...')
    outname = img_name + '_mask^' + g + '.tiff'
    p_out = os.path.join(outdir, 'masks', outname)

    # tifffile.imwrite(p_out, data=seg)

    seg = seg > 0
    out=seg.astype(np.uint8)
    out[out>0]=255
    writer = OmeTiffWriter(p_out, overwrite_file=True)
    writer.save(out)

    # output animation
    print('Exporting animation...')
    anim_name = img_name + '_mask^' + g + '.gif'
    su.animate_zstacks([imgs_rna[i].transpose(1,2,0), seg.astype(int).transpose(1,2,0)], vmax=[0.2, 1], gif_name=os.path.join(outdir, 'anim', anim_name))