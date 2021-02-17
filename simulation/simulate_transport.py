import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from scipy import misc as scimisc
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from scipy.optimize import minimize, curve_fit
from skimage import morphology
from matplotlib import pyplot as plt
import argparse
import copy
import csv
import gzip
import numpy as np
import os
import random
import scipy.stats as ss
import tifffile
import trimesh

# custom libraries
import scope_utils3 as su
import muscle_fish as mf

class RNA:
    def __init__(self, _id, init_pos=[0.,0.,0.]):
        global dims, k_deg, t
        self._id = _id
        self.init_pos = np.array(copy.deepcopy(init_pos))  # initial position, in image coords
        self.init_t = t

        self.pos = copy.deepcopy(self.init_pos)
        self.D_slow = self.sample_D_slow()  # diffusion coefficient, um^2 s^-1
        self.D_fast = self.sample_D_fast()  # diffusion coefficient, um^2 s^-1
        self.k_deg = k_deg
        self.is_dead = False
        self.p_death = 1. - np.exp(-1.*self.k_deg*tstep)  # from integration of exponential distribution
        self.hit_bounds = False

        # set initial state
        # 0 = slow-diffusion, 1 = fast-diffusion, 2 = slow-directed, 3 = fast-directed
        if fast_diff_only:
            self.state = 1
        else:
            self.state = 0 

    def sample_D_slow(self):
        global log_kde_low_diff
        return np.power(10, log_kde_low_diff.resample(size=1)[0][0]) 

    def sample_D_fast(self):
        global log_kde_high_diff
        return np.power(10, log_kde_high_diff.resample(size=1)[0][0])

    def step(self):
        global tstep, mask_allowed, transition_mat, log_kde_fast_dist, log_kde_slow_dist

        # test for state transition
        state_probs = transition_mat[self.state]
        self.state = np.random.choice([0, 1, 2, 3], p=state_probs)

        # advance RNA position according to rules of current state
        if self.state == 0:  # slow-diffusion
            # adjust each position component using a Gaussian random variable
            dpos = np.array([random.gauss(0, np.sqrt(2.*self.D_slow*tstep)/dims[d]) for d in ['x', 'y', 'z']])
            test_pos = np.add(self.pos, dpos)

            # if the RNA is out of bounds, retry until in bounds
            while is_out_of_bounds(test_pos, mask_allowed):
                dpos = np.array([random.gauss(0, np.sqrt(2.*self.D_slow*tstep)/dims[d]) for d in ['x', 'y', 'z']])
                test_pos = np.add(self.pos, dpos)
                self.hit_bounds = True
            
            self.pos = test_pos

        elif self.state == 1:  # fast-diffusion
            # adjust each position component using a Gaussian random variable
            dpos = np.array([random.gauss(0, np.sqrt(2.*self.D_fast*tstep)/dims[d]) for d in ['x', 'y', 'z']])
            test_pos = np.add(self.pos, dpos)

            # if the RNA is out of bounds, retry until in bounds
            while is_out_of_bounds(test_pos, mask_allowed):
                dpos = np.array([random.gauss(0, np.sqrt(2.*self.D_fast*tstep)/dims[d]) for d in ['x', 'y', 'z']])
                test_pos = np.add(self.pos, dpos)
                self.hit_bounds = True
            
            self.pos = test_pos
        
        elif self.state == 2:  # slow-directed
            # sample from distance distribution
            travel_dist = np.power(10, log_kde_slow_dist.resample(size=1)[0][0])

            # transport RNA
            test_pos = move_directed(self.pos, travel_dist)
            
            # if the RNA is out of bounds, retry until in bounds
            while is_out_of_bounds(test_pos, mask_allowed):
                # print(self.state, 'BOOP', self.pos, test_pos)
                travel_dist = np.power(10, log_kde_slow_dist.resample(size=1)[0][0])
                test_pos = move_directed(self.pos, travel_dist)
                self.hit_bounds = True
            
            self.pos = test_pos

        elif self.state == 3:  # fast-directed
            # sample from distance distribution
            travel_dist = np.power(10, log_kde_fast_dist.resample(size=1)[0][0])

            # transport RNA
            test_pos = move_directed(self.pos, travel_dist)
            
            # if the RNA is out of bounds, retry until in bounds
            while is_out_of_bounds(test_pos, mask_allowed):
                # print(self.state, 'BOOP', self.pos, test_pos)
                travel_dist = np.power(10, log_kde_fast_dist.resample(size=1)[0][0])
                test_pos = move_directed(self.pos, travel_dist)
                self.hit_bounds = True
            
            self.pos = test_pos
        
        else:  # unknown state
            raise ValueError('unknown state `' + str(self.state) + '` for RNA ' + str(self._id))

        # check for RNA decay event
        if self.check_death():
            self.kill()

        return self.pos
    
    def check_death(self):
        if random.random() < self.p_death:
            return True
        else:
            return False
    
    def kill(self):
        global t
        self.is_dead = True
        self.state = -1
        self.final_t = t
        self.final_pos = copy.deepcopy(self.pos)
        return True

def spawn_rna(_id, xlist, plist):
    '''
    Create an instance of RNA and give it a starting position in image coordinates.
    '''
    init_pos = random.choices(xlist, weights=plist, k=1)[0]
    # print('RNA spawned at position ' + str(init_pos))
    return RNA(_id, init_pos=init_pos)

def move_directed(pos, dist):
    global e_axial, e_radial, dims

    # choose direction of movement
    if random.random() < 0.5:
        # travel along axial direction
        if random.random() < 0.5:
            direction = np.array(e_axial)  # parallel to fiber
        else:
            direction = -1.*np.array(e_axial)  # antiparallel to fiber
    else:
        # travel along radial direction
        phi = 2.*np.pi*random.random()
        direction = np.multiply(e_radial, (np.cos(phi), np.cos(phi), np.sin(phi)))

    # transport RNA
    dpos = np.divide(direction*dist, [dims[d] for d in ['x', 'y', 'z']])
    new_pos = np.add(pos, dpos)
    return new_pos

def is_out_of_bounds(pos, allowed_region):
    '''
    Check if an RNA position falls outside the allowed region of the image.
    '''
    # test if position is outside image boundaries
    for e_i, ub in zip(pos, allowed_region.shape):
        if e_i < 0 or e_i > ub-1:
            return True
    
    # test if position falls outside allowed region of image
    ipos = tuple([int(round(e_i)) for e_i in pos])
    if allowed_region[ipos]:
        return False
    else:
        return True

def get_fiber_axes(fiber_mask):
    
    '''def line_opt(params, distmat):
        m, b = params
        x = np.linspace(0, distmat.shape[0]-1, num=1000)
        y = m*x + b

        intens = interpn((list(range(distmat.shape[0])), list(range(distmat.shape[1]))), distmat, list(zip(x, y)), bounds_error=False)
        score = -1.*np.nanmean(intens)

        print(m, b, score)

        return score

    # flatten to 2D and skeletonize
    mask_2d = np.amax(fiber_mask, axis=2)
    medial_axis, dists_within_fiber = morphology.medial_axis(mask_2d, return_distance=True)

    # extend distance matrix to border
    dists_from_fiber = distance_transform_edt(np.logical_not(dists_within_fiber > 0))
    dists = dists_within_fiber - dists_from_fiber
    dists = dists - np.amin(dists)
    # medial_axis = morphology.medial_axis(mask_2d)
    # dists = distance_transform_edt(np.logical_not(medial_axis))
    # print(dists)

    # fit a line to the distance matrix
    res = minimize(line_opt, x0=[1., 1000.], bounds=[(-100, 100), (-np.inf, np.inf)], args=(dists))
    m, b = res.x'''

    def line_opt(x, m, b):
        return m*x + b

    # flatten to 2D and skeletonize
    mask_2d = np.amax(fiber_mask, axis=2)
    medial_axis, dists = morphology.medial_axis(mask_2d, return_distance=True)
    xdata, ydata = np.nonzero(medial_axis)

    # fit a line to the distance matrix
    popt, pcov = curve_fit(line_opt, xdata, ydata)
    m, b = popt

    # draw line over image
    x = np.array([0, mask_2d.shape[0]])
    y = m*x + b
    fig, ax = plt.subplots()
    ax.imshow(dists, cmap='binary_r')
    ax.plot(y, x, 'r-')
    ax.set_xlim([0, mask_2d.shape[1]])
    ax.set_ylim([0, mask_2d.shape[0]])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, img_name+'_fiber_medial_axis.pdf'), dpi=300)
    plt.close()
    
    e_ax = (1./np.sqrt(m**2. + 1), m/np.sqrt(m**2. + 1), 0)
    e_rad = (m/np.sqrt(m**2. + 1), -1./np.sqrt(m**2. + 1), 0)

    return e_ax, e_rad
    
def calculate_transitions(states):
    # define state transition matrix
    p01 = 0.
    p02 = 0.070100143061516444
    p03 = 0.027181688125894135
    p00 = 1.-(p01+p02+p03)

    # p11 = np.exp(-1.*tstep/60.)
    # p12 = 0.070100143061516444
    # p13 = 0.027181688125894135
    # p10 = 1.-(p11+p12+p13)

    p11 = 1.  # fast-diffusion persists until decay
    p12 = 0.
    p13 = 0.
    p10 = 0.

    full_tm = np.array([
        [p00, p01, p02, p03],
        [p10, p11, p12, p13],
        [1,0,0,0],
        [1,0,0,0]
    ])

    tm = copy.deepcopy(full_tm)
    for s in list(set([0,1,2,3]).difference(states)):  # states must be a set
        tm[:,s] = np.zeros(4)
        for i in range(len(full_tm)):
            tm[i,0] = 1. - np.sum(tm[i,1:])
    
    return tm


#--  INITIALIZATION  ----------------------------------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_name', type=str, nargs=1, help='Name of image.')
parser.add_argument('indir', type=str, nargs=1, help='Directory of input files for simulation.')
parser.add_argument('gene', type=str, nargs=1, help='Gene name.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory of output files.')
parser.add_argument('-s', '--states', type=str, help='Allowed mobility states (0 = slow-diffusion, 1 = fast-diffusion, 2 = slow-directed, 3 = fast-directed).', default='023')
parser.add_argument('-p', '--params', type=str, help='Path to gene params file.', default='gene_params.csv')
parser.add_argument('-l', '--length', type=float, help='Length of simulation (hr).', default=1000)
parser.add_argument('-t', '--timestep', type=float, help='Length of timestep (s).', default=10)
parser.add_argument('-a', '--sample', type=int, help='Number of timesteps between output rows. If 1, save in animation-friendly format.', default=1)
parser.add_argument('-e', '--events', type=str, help='Changes to simulation conditions, in the format `t(hr):states,...` . eg. 200:0,500:023', default='')
parser.add_argument('--compress', action='store_true', help='Compress output files with gzip.')

# parse arguments
args = vars(parser.parse_args())
img_name = args['image_name'][0]
indir = args['indir'][0]
gene = args['gene'][0]
outdir = args['outdir'][0]
states_str = args['states']
p_params = args['params']
sim_len = args['length']
tstep = args['timestep']
sample_freq = args['sample']
event_str = args['events']
compressed = args['compress']

# set file opener
if compressed:
    wopen = gzip.open
    wsuf = '.gz'
    wmode = 'wt'
    amode = 'at'
else:
    wopen = open
    wsuf = ''
    wmode = 'w'
    amode = 'a'

# parse events
events = []
for ev in event_str.split(','):
    if ev:
        try:
            t, s = ev.split(':')
            events.append([3600.*float(t), set([int(c) for c in s])])
        except:
            raise ValueError('event string must be formatted as `t(hr):states,...` (eg. 200:0,500:023)')
events.sort(key=lambda x: x[0])

# describe simulation in stdout
print('------------------------------------------------')
print('            RNA TRANSPORT SIMULATION            ')
print('------------------------------------------------')
print('')
print('FIBER:   ' + img_name)
print('GENE:    ' + gene)
print('STATES:  ' + states_str)
print('LENGTH:  ' + str(sim_len) + ' hr')
print('STEP:    ' + str(tstep) + ' s')
print('')

# make output directories if not already present
if not os.path.exists(outdir):
    os.makedirs(outdir)

# set paths for fiber data
p_fiber = os.path.join(indir, img_name + '_fiber.npy')
p_nuc = os.path.join(indir, img_name + '_nuclei.npy')
p_dims = os.path.join(indir, img_name + '_dims.csv')

BUFFER_LEN = 1000*sample_freq

# parse allowed states
allowed_states = set([int(c) for c in states_str])
fast_diff_only = False
if not 0 in allowed_states:
    if states_str == '1':
        # only allow fast-diffusion - this is a valid simulation
        fast_diff_only = True
    else:
        raise ValueError('state 0 (`slow-diffusion`) is required.')
elif len(allowed_states.difference(set([0,1,2,3]))) > 0:
    raise ValueError('only states 0, 1, 2, and 3 are allowed.')
disallowed_states = set([0,1,2,3]).difference(allowed_states)

# open gene params file
with open(p_params, 'r') as infile:
    reader = csv.reader(infile)
    halflives = {}
    densities = {}
    for row in reader:
        if not row[0].startswith('#'):
            halflives[row[0].lower()] = float(row[1])
            densities[row[0].lower()] = float(row[2])

halflife = halflives[gene.lower()]*3600.
density = densities[gene.lower()]

print('CALCULATIONS: \n')
print('Half-life (s): ' + str(halflife))
print('Cytoplasmic density (molecules/um^3): ' + str(density))

# open fiber and nuclei masks
mask_fiber = np.load(p_fiber)
mask_nuc = np.load(p_nuc)
# mask_fiber = np.logical_or(mask_fiber.astype(bool), mask_nuc.astype(bool)).astype(int)
mask_nuc_eroded = morphology.binary_erosion(mask_nuc.astype(bool))

# construct allowed region for RNA mobility
mask_allowed = np.logical_and(mask_fiber.astype(bool), np.logical_not(mask_nuc_eroded).astype(bool))
labeled_mask_allowed, n_regions = morphology.label(mask_allowed, return_num=True)
voxel_counts = []
for l in range(1, n_regions+1):
    voxel_counts.append(np.count_nonzero(labeled_mask_allowed == l))
mask_allowed = (labeled_mask_allowed == np.argmax(voxel_counts) + 1)
tifffile.imwrite(os.path.join(outdir, img_name+'_allowed_region.tiff'), data=mask_allowed.transpose(2,0,1).astype(np.uint8)*255, compress=6, photometric='minisblack')
su.animate_zstacks([mask_fiber, mask_nuc, mask_allowed], titles=['fiber', 'nuclei', 'allowed'], gif_name=os.path.join(outdir, img_name+'_masks^'+gene+'.gif'))

# create spawning probability matrix
labeled_mask_nuc, n_nuc = morphology.label(mask_nuc, return_num=True)
labeled_mask_nuc[mask_nuc_eroded == 1] = 0
labeled_mask_nuc[mask_allowed == 0] = 0
voxels_by_label = {l:np.count_nonzero(labeled_mask_nuc == l) for l in range(1, n_nuc+1)}

spawn_probs = np.zeros_like(mask_nuc, dtype=np.float32)
sp_x = []
sp_P = []
for l in range(1, n_nuc+1):
    prob = 1./(float(n_nuc)*voxels_by_label[l])
    l_mask = (labeled_mask_nuc == l)
    spawn_probs[l_mask] = prob

    sp_x_extension = list(np.column_stack(np.nonzero(l_mask)))
    sp_x.extend(sp_x_extension)
    sp_P.extend([prob for i in range(len(sp_x_extension))])

su.animate_zstacks([mask_nuc, spawn_probs], titles=['nuclei', 'spawn_probabilities'], gif_name=os.path.join(outdir, img_name+'spawnprobs^'+gene+'.gif'))

# get image dimensions
with open(p_dims, 'r') as infile:
    reader = csv.reader(infile)
    dims = {row[0]:float(row[1]) for row in reader}
voxel_vol = dims['x'] * dims['y'] * dims['z']

# calculate RNA steady state levels
fiber_volume = voxel_vol*np.count_nonzero(mask_fiber)
avg_num_rnas = fiber_volume*density

print('Fiber volume (um^3): ' + str(fiber_volume))
print('Average num RNAs: ' + str(avg_num_rnas))

# calculate production and degradation rates (Poisson)
k_deg = np.log(2)/halflife  # s^-1
k_prod = k_deg*avg_num_rnas  # molec s^-1

print('Production rate (molec s^-1): ' + str(k_prod))
print('Degradation rate (s^-1): ' + str(k_deg))

# load parameter distributions and apply KDE
meas_high_diff = np.loadtxt('distributions/high_diffusion.txt')
meas_low_diff = np.loadtxt('distributions/low_diffusion.txt')
meas_fast_dist = np.loadtxt('distributions/fast_distance.txt')
meas_slow_dist = np.loadtxt('distributions/slow_distance.txt')
meas_fast_vel = np.loadtxt('distributions/fast_velocity.txt')
meas_slow_vel = np.loadtxt('distributions/slow_velocity.txt')

log_kde_high_diff = ss.gaussian_kde(np.log10(meas_high_diff))
log_kde_low_diff = ss.gaussian_kde(np.log10(meas_low_diff))
log_kde_fast_dist = ss.gaussian_kde(np.log10(meas_fast_dist))
log_kde_slow_dist = ss.gaussian_kde(np.log10(meas_slow_dist))
log_kde_fast_vel = ss.gaussian_kde(np.log10(meas_fast_vel))
log_kde_slow_vel = ss.gaussian_kde(np.log10(meas_slow_vel))

transition_mat = calculate_transitions(allowed_states)

# get cylindrical coordinates of fiber
e_axial, e_radial = get_fiber_axes(mask_fiber)


#--  SIMULATION  --------------------------------------------------------------#

timesteps = np.arange(0, sim_len*3600., tstep)
live_rnas = {}
dead_rnas = {}
live_rna_ids = []
cur_rna = 0

output_vars = ['x', 'y', 'z', 'state']
output_buffer_dict = {v:[] for v in output_vars}

# clear output files
for v in output_vars:
    wopen(os.path.join(outdir, img_name+'_paths_'+v+'^'+gene+'.csv'+wsuf), wmode).close()

# run simulation
print('\nSTARTING SIMULATION...\n')
trace = []

for j, t in enumerate(timesteps):
    # keep user informed
    if j % 1000 == 0:
        print(str(round(t/3600., 2)) + ' hours simulated.')
    
    # check for change to simulation conditions
    if events:
        if t > events[0][0]:
            t_event, new_states = events.pop(0)
            transition_mat = calculate_transitions(new_states)  # recalculate transition probabilities
            print('Allowed states changed to ' + str(new_states) + ' at time ' + str(round(t/3600., 2)) + ' hr.')

    # advance existing RNAs
    for _id in live_rna_ids:
        live_rnas[_id].step()

    # purge decayed RNAs
    for _id in live_rna_ids:
        if live_rnas[_id].is_dead == True:
            dead_rna = live_rnas.pop(_id)  # remove from live RNAs container
            dead_rnas.update({_id:dead_rna})  # add to dead RNAs container

    # spawn new RNAs
    num_to_spawn = ss.poisson.rvs(k_prod*tstep)  # Poisson process, mean = k_prod*tstep
    new_rna_objects = [spawn_rna(cur_rna+i, sp_x, sp_P) for i in range(num_to_spawn)]
    new_rna_ids = [cur_rna+i for i in range(num_to_spawn)]
    new_rnas = {_id:ob for _id, ob in zip(new_rna_ids, new_rna_objects)}
    live_rnas.update(new_rnas)
    live_rna_ids = list(live_rnas.keys())

    # housekeeping
    cur_rna += num_to_spawn
    trace.append([t, len(live_rnas), len(dead_rnas)])

    # add traces to output buffer
    if sample_freq == 1:
        # save animation-friendly format
        for i, v in enumerate(output_vars):
            output = [str(t)]
            output.extend([''] * cur_rna)
            for _id, rna in live_rnas.items():
                if i < 3:  # position
                    output[_id+1] = str(rna.pos[i])
                else:  # state
                    output[_id+1] = str(rna.state)
            for _id, rna in dead_rnas.items():
                if i < 3:  # position
                    output[_id+1] = str(rna.final_pos[i])
                else:  # state
                    output[_id+1] = str(rna.state)
            out_line = ','.join(output) + '\n'
            output_buffer_dict[v].append(out_line)
    else:
        if (j % sample_freq) == 0:
            # save in unordered format, live RNAs only
            id_to_index = {_id:i+1 for i, _id in enumerate(live_rna_ids)}
            for i, v in enumerate(output_vars):
                output = [str(t)]
                output.extend([''] * len(live_rna_ids))
                for _id, rna in live_rnas.items():
                    if i < 3:  # position
                        output[id_to_index[_id]] = str(rna.pos[i])
                    else:  # state
                        output[id_to_index[_id]] = str(rna.state)
                out_line = ','.join(output) + '\n'
                output_buffer_dict[v].append(out_line)

    # cash out buffer to file if we've reached BUFFER_LEN
    if (j+1) % BUFFER_LEN == 0:
        for v in output_vars:
            with wopen(os.path.join(outdir, img_name+'_paths_'+v+'^'+gene+'.csv'+wsuf), amode) as outfile:
                outfile.writelines(output_buffer_dict[v])

        # clear buffer
        output_buffer_dict = {v:[] for v in output_vars}

# finish output
for v in output_vars:
    with wopen(os.path.join(outdir, img_name+'_paths_'+v+'^'+gene+'.csv'+wsuf), amode) as outfile:
        outfile.writelines(output_buffer_dict[v])
output_buffer_dict = {v:[] for v in output_vars}


#--  OUTPUT  ------------------------------------------------------------------#

# draw trace of RNA count
trace_np = np.array(trace)
fig, ax = plt.subplots(1, 2)
ax[0].plot(trace_np[:,0], trace_np[:,1], 'b')
ax[1].plot(trace_np[:,0], trace_np[:,2], 'r')
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('Number of live RNAs')
ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('Number of dead RNAs')
plt.tight_layout()
plt.savefig(os.path.join(outdir, img_name+'_trace^'+gene+'.pdf'), dpi=300)






'''x = np.linspace(-3.5, -1, num=200)
y = log_kde_high_diff.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_high_diff), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Diffusion coefficient (um^2/s))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_high_diff.pdf'), dpi=300)

x = np.linspace(-7, -1, num=200)
y = log_kde_low_diff.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_low_diff), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Diffusion coefficient (um^2/s))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_low_diff.pdf'), dpi=300)


x = np.linspace(-1, 2, num=200)
y = log_kde_fast_dist.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_fast_dist), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Distance (um))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_fast_dist.pdf'), dpi=300)

x = np.linspace(-1, 1, num=200)
y = log_kde_slow_dist.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_slow_dist), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Distance (um))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_slow_dist.pdf'), dpi=300)


x = np.linspace(-1, 1, num=200)
y = log_kde_fast_vel.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_fast_vel), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Velocity (um/s))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_fast_vel.pdf'), dpi=300)

x = np.linspace(-2, 0, num=200)
y = log_kde_slow_vel.evaluate(x)

fig, ax = plt.subplots()
ax.hist(np.log10(meas_slow_vel), bins=10, normed=True)
ax.plot(x, y, 'r')
ax.set_xlabel('log(Velocity (um/s))')
ax.set_ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'dist_slow_vel.pdf'), dpi=300)'''

'''
p01 = 1.-np.power(1-0.03719599427753934, tstep/60.)
p02 = 0.070100143061516444
p03 = 0.027181688125894135
p00 = 1.-(p01+p02+p03)

p11 = np.exp(-1.*tstep/60.)
p12 = 0.070100143061516444
p13 = 0.027181688125894135
p10 = 1.-(p11+p12+p13)

transition_mat = {
    0:[p00, p01, p02, p03],
    1:[p10, p11, p12, p13],
    2:[1,0,0,0],
    3:[1,0,0,0]
}
'''