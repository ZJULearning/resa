import os.path as osp
import numpy as np
import torchvision
import utils.transforms as tf
from .base_dataset import BaseDataset
from .registry import DATASETS
import cv2
import torch


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, cfg=cfg)

    def init(self):
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                self.img.append(line_split[0])
                self.img_list.append(self.img_path + line_split[0])
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5])]))

    def transform_train(self):
        train_transform = torchvision.transforms.Compose([
            tf.GroupRandomRotation(degree=(-2, 2)),
            tf.GroupRandomHorizontalFlip(),
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return train_transform

    # def __getitem__(self, idx):
    #     img = cv2.imread(self.img_list[idx]).astype(np.float32)
    #     label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
    #     label = label.squeeze()

    #     img = img[self.cfg.cut_height:, :, :]
    #     label = label[self.cfg.cut_height:, :]

    #     exist = self.exist_list[idx]

    #     if self.transform:
    #         img, label = self.transform((img, label))

    #     img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
    #     label = torch.from_numpy(label).contiguous().long()
    #     meta = {'file_name': self.img[idx]}

    #     data = {'img': img, 'label': label,
    #             'exist': exist, 'meta': meta}
    #     return data
