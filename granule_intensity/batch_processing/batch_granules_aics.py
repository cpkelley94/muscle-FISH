'''Set up job scripts for batch processing.
'''

import argparse
import os
import subprocess

# create parser and parse arguments passed in the command line
parser = argparse.ArgumentParser()
parser.add_argument('file_table', type=str, nargs=1, help='Path to the .txt file listing all CZI images and arguments.')

args = vars(parser.parse_args())
p_ftable = args['file_table'][0]

# file list should be in format [img_path, out_dir, gene1, gene2, t_dapi, t_fiber, t_spot1, t_spot2]
with open(p_ftable, 'r') as listfile:
    file_list = listfile.readlines()

# create a bash script to analyze each image
sh_paths = []
for i, line in enumerate(file_list):
    p_img, outdir, gene1, gene2, t_dapi, t_fiber, t_spot1, t_spot2 = line.replace('\r','').replace('\n','').split('\t')
    img_name = os.path.splitext(os.path.basename(p_img))[0]

    cmd = 'module load python/3.6.5\n'
    cmd += 'python fish_analysis_granule_intensity.py "' + p_img + '" ' + outdir + ' ' + gene1
    if not gene2 == '.':
        cmd += ' ' + gene2
    if not t_dapi == '.':
        cmd += ' -d ' + t_dapi
    if not t_fiber == '.':
        cmd += ' -f ' + t_fiber
    if not t_spot1 == '.':
        cmd += ' -1 ' + t_spot1
    if not t_spot2 == '.':
        cmd += ' -2 ' + t_spot2
    cmd += ' --plot\n'

    sh_path = outdir + '_' + str(i).zfill(2) + '.sh'
    sh_paths.append([outdir[0] + '_' + str(i).zfill(2), sh_path])

    with open(sh_path, 'w') as sh_file:
        sh_file.write(cmd)

for name, bp in sh_paths:
    cmd = 'chmod +x ' + bp
    subprocess.call(cmd, shell=True)
