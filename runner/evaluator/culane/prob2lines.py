import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tqdm


def getLane(probmap, pts, cfg = None):
    thr = 0.3
    coordinate = np.zeros(pts)
    cut_height = 0
    if cfg.cut_height:
        cut_height = cfg.cut_height
    for i in range(pts):
        line = probmap[round(cfg.img_height-i*20/(590-cut_height)*cfg.img_height)-1]
        if np.max(line)/255 > thr:
            coordinate[i] = np.argmax(line)+1
    if np.sum(coordinate > 0) < 2:
        coordinate = np.zeros(pts)
    return coordinate


def prob2lines(prob_dir, out_dir, list_file, cfg = None):
    lists = pd.read_csv(list_file, sep=' ', header=None,
                        names=('img', 'probmap', 'label1', 'label2', 'label3', 'label4'))
    pts = 18

    for k, im in enumerate(lists['img'], 1):
        existPath = prob_dir + im[:-4] + '.exist.txt'
        outname = out_dir + im[:-4] + '.lines.txt'
        prefix = '/'.join(outname.split('/')[:-1])
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        f = open(outname, 'w')

        labels = list(pd.read_csv(existPath, sep=' ', header=None).iloc[0])
        coordinates = np.zeros((4, pts))
        for i in range(4):
            if labels[i] == 1:
                probfile = prob_dir + im[:-4] + '_{0}_avg.png'.format(i+1)
                probmap = np.array(Image.open(probfile))
                coordinates[i] = getLane(probmap, pts, cfg)

                if np.sum(coordinates[i] > 0) > 1:
                    for idx, value in enumerate(coordinates[i]):
                        if value > 0:
                            f.write('%d %d ' % (
                                round(value*1640/cfg.img_width)-1, round(590-idx*20)-1))
                    f.write('\n')
        f.close()
