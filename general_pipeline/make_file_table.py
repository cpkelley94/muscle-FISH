import csv
import os

flist = []
d = os.path.join(os.getcwd(), 'images')

for f in sorted(os.listdir(d)):
    if f.endswith('.czi') and 'still' not in f:
        p_img = os.path.join(d, f)
        flist.append([p_img, '.', '.', '.', '.', '.'])

with open('1_file_list.txt', 'w') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(flist)
