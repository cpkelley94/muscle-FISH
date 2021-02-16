import csv
import os

dirs = [dir for dir in os.listdir('images') if os.path.isdir(os.path.join('images', dir))]

for dir in dirs:
    flist = []
    d = os.path.join(os.getcwd(), 'images', dir)

    for f in sorted(os.listdir(d)):
        if f.endswith('.czi') and 'still' not in f:
            p_img = os.path.join(d, f)
            if f.startswith('_'):
                channels = ['.']
            else:
                if int(f[0]) < 5:
                    channels = ['561', '633']
                else:
                    channels = ['.']
            for c in channels:
                flist.append([p_img, '.', c, '.', '.', '.'])

    with open('file_list_' + dir + '.txt', 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerows(flist)

    del flist[:]
