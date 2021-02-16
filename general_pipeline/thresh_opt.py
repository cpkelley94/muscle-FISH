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
from skimage import feature, exposure, filters, morphology
from xml.etree import ElementTree
import argparse
import csv
import numpy as np
import os
import scipy.ndimage as ndi
import scope_utils

from skimage.filters import try_all_threshold

def count_spots(threshold):
    return len(feature.blob_log(img_rna, 1., 1., num_sigma=1, threshold=threshold))

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

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('czi', type=str, nargs=1, help='Path to .CZI image file.')

# parse arguments
args = vars(parser.parse_args())
p_czi = args['czi'][0]

# open the image
with CziFile(p_czi) as czi_file:
    meta = czi_file.metadata()
    img = czi_file.asarray()
img_name = os.path.splitext(os.path.basename(p_czi))[0]

# split channels
img_dapi = scope_utils.normalize_image(img[0, 0, 1, 0, :, :, :, 0].transpose(1,2,0))
img_rna = scope_utils.normalize_image(img[0, 0, 0, 0, :, :, :, 0].transpose(1,2,0))

# optimize threshold for spot detection on the fly
t = np.linspace(0.005, 0.05, num=100)
num_blobs = np.empty_like(t, dtype=int)

print 'starting spot detection threshold optimization'
pool = Pool(processes=8)
num_blobs = pool.map(count_spots, t)

num_blobs_smooth = ndi.gaussian_filter1d(num_blobs, sigma=1)
dn_dt = np.gradient(num_blobs_smooth)

plt.plot(t, num_blobs, 'b-')
plt.plot(t, num_blobs_smooth, 'r-')
plt.xlabel('t')
plt.ylabel('num blobs')
plt.savefig(img_name + '_topt.png', dpi=300)
plt.close()

plt.plot(t, dn_dt, 'k-')
plt.xlabel('t')
plt.ylabel('dn/dt')
plt.savefig(img_name + '_dndt.png', dpi=300)
plt.close()

print 'optimized threshold = ' + str(t[np.argmax(dn_dt)])
