import csv
import os

flist = []
cur_dir = os.getcwd()
cond_dirs = ['No_Treat', 'Noco', 'Wash']
out_dirs = ['1_notreat', '2_noco', '3_wash']
cond_paths = [os.path.join(cur_dir, 'images', c) for c in cond_dirs]

for i, d in enumerate(cond_paths):
    for f in sorted(os.listdir(d)):
        if f.endswith('.czi') and 'still' not in f:
            p_img = os.path.join(d, f)
            out_dir = out_dirs[i]
            flist.append([p_img, out_dir, '.', '.', '.', '.', '.', '.'])

with open('3_file_list.txt', 'w') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(flist)
