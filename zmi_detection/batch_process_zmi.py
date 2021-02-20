import argparse
import os
import subprocess

# create parser and parse arguments passed in the command line
parser = argparse.ArgumentParser()
parser.add_argument('file_table', type=str, nargs=1, help='Path to the .txt file listing all CZI images and arguments.')
parser.add_argument('outdir', type=str, nargs=1, help='Name of the output directory.')

args = vars(parser.parse_args())
p_ftable = args['file_table'][0]
outdir = args['outdir'][0]

# make output directories if not already present
if not os.path.exists(os.path.join(outdir, 'plots')):
    os.makedirs(os.path.join(outdir, 'plots'))
if not os.path.exists(os.path.join(outdir, 'stats')):
    os.makedirs(os.path.join(outdir, 'stats'))
if not os.path.exists(os.path.join(outdir, 'dists')):
    os.makedirs(os.path.join(outdir, 'dists'))

# open file list
with open(p_ftable, 'r') as listfile:
    file_list = listfile.readlines()

# create a bash script to analyze each image
sh_paths = []
for i, line in enumerate(file_list):
    pref, gene = line.replace('\r','').replace('\n','').split('\t')

    cmd = 'module load python/3.6.5\n'
    cmd += 'python detect_zmi.py '
    cmd += os.path.join('images', 'ZL', pref + '_struct_segmentation.tiff') + ' '
    cmd += os.path.join('images', 'MT', pref + '_struct_segmentation.tiff') + ' '
    cmd += os.path.join('images', pref + '_nuclei.tiff') + ' '
    cmd += os.path.join('images', pref + '_spots.txt') + ' '
    cmd += gene + ' '
    cmd += '-o ' + outdir + '\n'

    sh_path = pref + '.sh'
    sh_paths.append([pref, sh_path])

    with open(sh_path, 'w') as sh_file:
        sh_file.write(cmd)

for name, p in sh_paths:
    cmd = 'chmod +x ' + p
    subprocess.Popen(cmd, shell=True)
