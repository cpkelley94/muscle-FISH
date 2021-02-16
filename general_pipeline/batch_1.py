import argparse
import os
import subprocess

# create parser and parse arguments passed in the command line
parser = argparse.ArgumentParser()
parser.add_argument('file_table', type=str, nargs=1, help='Path to the .txt file listing all CZI images and arguments.')
parser.add_argument('outdir', type=str, nargs=1, help='Name of the output directory.')
parser.add_argument('--run', action='store_true', help='Start SLURM processes.')

args = vars(parser.parse_args())
p_ftable = args['file_table'][0]
outdir = args['outdir'][0]
should_run = args['run']

# make output directories if not already present
if not os.path.exists(os.path.join(outdir, 'anim')):
    os.makedirs(os.path.join(outdir, 'anim'))
if not os.path.exists(os.path.join(outdir, 'counts')):
    os.makedirs(os.path.join(outdir, 'counts'))

# file list should be in format [img_path, gene, channel, t_dapi, t_fiber, t_spot]
with open(p_ftable, 'r') as listfile:
    file_list = listfile.readlines()

# create a bash script to analyze each image
sh_paths = []
for i, line in enumerate(file_list):
    p_img, gene, t_dapi, t_fiber, t_spot = line.replace('\r','').replace('\n','').split('\t')
    img_name = os.path.splitext(os.path.basename(p_img))[0]

    cmd = 'module load python/3.6.5\n'
    cmd += 'python 1_py3.py "' + p_img + '" ' + outdir + ' ' + gene
    if not t_dapi == '.':
        cmd += ' -d ' + t_dapi
    if not t_fiber == '.':
        cmd += ' -f ' + t_fiber
    if not t_spot == '.':
        cmd += ' -s ' + t_spot
    cmd += ' --plot\n'

    sh_path = outdir + '_' + str(i).zfill(2) + '.sh'
    sh_paths.append(['1_' + str(i).zfill(2), sh_path])

    with open(sh_path, 'w') as sh_file:
        sh_file.write(cmd)

for name, p in sh_paths:
    cmd = 'chmod +x ' + p
    subprocess.Popen(cmd, shell=True)

if should_run:
    # slurmify the bash scripts and queue them in slurm
    for name, p in sh_paths:
        cmd = 'slurmify-run ' + p + ' -n ' + name + ' -m 24 -t 1 --burst'
        subprocess.Popen(cmd, shell=True)
