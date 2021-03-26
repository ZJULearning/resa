import os.path as osp
import numpy as np
import torchvision
import utils.transforms as tf
from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, 'seg_label/list', cfg)

    def transform_train(self):
        input_mean = self.cfg.img_norm['mean']
        train_transform = torchvision.transforms.Compose([
            tf.GroupRandomRotation(),
            tf.GroupRandomHorizontalFlip(),
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return train_transform


    def init(self):
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                self.img.append(line_split[0])
                self.img_list.append(self.img_path + line_split[0])
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5]),
                              int(line_split[6]), int(line_split[7])
                              ]))
