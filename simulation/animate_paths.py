import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from scipy import misc as scimisc
from skimage import morphology
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import copy
import csv
import numpy as np
import os
import pandas as pd
import random
import scipy.stats as ss
import tifffile
import trimesh

# custom libraries
import scope_utils3 as su
import muscle_fish as mf

class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb()) 
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 

def draw_frame(t):
    '''
    FuncAnimation callable to animate RNA movement in simulation.
    '''
    # remove old transport events
    global transport_events
    for plot in transport_events:
        plot[0].remove()
    transport_events = []

    # update positions
    x = [rna_dict['x'][r][t] for r in cols]
    y = [rna_dict['y'][r][t] for r in cols]
    xy = np.column_stack((y, x))
    scatter.set_offsets(xy)

    # update colors
    c = [spot_cmap[rna_dict['state'][r][t]] for r in cols]
    scatter.set_facecolors(c)

    # draw new active transport events
    t_idx = times.index(t)
    for i, state in enumerate([rna_dict['state'][r][t] for r in cols]):
        if state == 2 or state == 3:
            # draw slow-directed
            x1 = rna_dict['x'][i][times[t_idx-1]]
            x2 = rna_dict['x'][i][t]
            y1 = rna_dict['y'][i][times[t_idx-1]]
            y2 = rna_dict['y'][i][t]
            event = ax.plot([y1, y2], [x1, x2], ls='-', lw=3, c=spot_cmap[state], solid_capstyle='round', zorder=1000)
            transport_events.append(event)

    return scatter


# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_name', type=str, nargs=1, help='Name of image.')
parser.add_argument('indir', type=str, nargs=1, help='Directory of input files for simulation.')
parser.add_argument('gene', type=str, nargs=1, help='Gene name.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory of output files.')

# parse arguments
args = vars(parser.parse_args())
img_name = args['image_name'][0]
indir = args['indir'][0]
gene = args['gene'][0]
outdir = args['outdir'][0]

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

# open position and state paths
rna_dict = {}
dims = ['x', 'y', 'z', 'state']
files = [p_x, p_y, p_z, p_state]

for d, f in zip(dims, files):
    print('Loading `' + d + '` data from ' + f + ' ...')
    with open(f, 'r') as infile:
        reader = csv.reader(infile)
        indata = [row for row in reader]
    
    n_rnas = len(indata[-1]) - 1
    indata_flush = []
    for row in indata:
        num_to_extend = n_rnas + 1 - len(row)
        if d == 'state':
            indata_flush.append([round(float(r)) for r in row] + ([-2]*num_to_extend))  # -2 = unborn
        else:
            indata_flush.append([round(float(r)) for r in row] + ([-20]*num_to_extend))
    
    flush_nparr = np.array(indata_flush)

    cols = list(range(n_rnas))
    df = pd.DataFrame(data=flush_nparr[:,1:], index=flush_nparr[:,0], columns=cols, dtype=int)

    rna_dict.update({d:copy.deepcopy(df)})
    times = list(flush_nparr[:,0])

# draw final frame of spot movement
fiber_2d = np.amax(mask_fiber.astype(int), axis=2)
nuc_2d = np.amax(mask_nuc.astype(int), axis=2)

spot_cmap = {-2:'k', -1:'#333333', 0:'#008ffd', 1:'#75e900', 2:'#eede0a', 3:'#f075f5'}

fig, ax = plt.subplots()
ax.imshow(fiber_2d, vmax=10, cmap='binary_r')
ax.imshow(nuc_2d, cmap=su.cmap_NtoW)
scatter = ax.scatter([rna_dict['y'][r][times[-1]] for r in cols], [rna_dict['x'][r][times[-1]] for r in cols], c=[spot_cmap[rna_dict['state'][r][times[-1]]] for r in cols], s=3)

transport_events = []
for i, state in enumerate([rna_dict['state'][r][times[-1]] for r in cols]):
    if state == 2 or state == 3:
        # draw directed transport as line segment
        x1 = rna_dict['x'][i][times[-2]]
        x2 = rna_dict['x'][i][times[-1]]
        y1 = rna_dict['y'][i][times[-2]]
        y2 = rna_dict['y'][i][times[-1]]
        event = ax.plot([y1, y2], [x1, x2], ls='-', lw=3, c=spot_cmap[state], solid_capstyle='round', zorder=1000)
        transport_events.append(event)

ax.set_xlim([0, fiber_2d.shape[1]])
ax.set_ylim([0, fiber_2d.shape[0]])
plt.savefig(os.path.join(outdir, img_name + '_final_frame^' + gene + '.png'), dpi=300)
plt.close()

# draw animation of spot movement
fig, ax = plt.subplots()
ax.imshow(fiber_2d, vmax=10, cmap='binary_r')
ax.imshow(nuc_2d, cmap=su.cmap_NtoW)
scatter = ax.scatter([rna_dict['y'][r][0] for r in cols], [rna_dict['x'][r][0] for r in cols], c=[spot_cmap[rna_dict['state'][r][0]] for r in cols], s=3)
ax.set_xlim([0, fiber_2d.shape[1]])
ax.set_ylim([0, fiber_2d.shape[0]])
transport_events = []

anim = FuncAnimation(fig, draw_frame, frames=times, interval=16.667, blit=False)
moviewriter = FasterFFMpegWriter(fps=60)
anim.save(os.path.join(outdir, img_name + '_rna_anim^' + gene + '.mp4'), writer=moviewriter, dpi=300)
plt.close()