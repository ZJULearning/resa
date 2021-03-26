import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import utils.transforms as tf
from .registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, img_path, data_list, list_path='list', cfg=None):
        self.cfg = cfg
        self.img_path = img_path
        self.list_path = osp.join(img_path, list_path)
        self.data_list = data_list
        self.is_testing = ('test' in data_list)
        self.img = []
        self.img_list = []
        self.label_list = []
        self.exist_list = []

        self.transform = self.transform_val() if self.is_testing else self.transform_train()

        self.init()

    def transform_train(self):
        raise NotImplementedError()

    def transform_val(self):
        val_transform = torchvision.transforms.Compose([
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return val_transform


    def init(self):
        raise NotImplementedError()


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32)
        label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label.squeeze()

        img = img[self.cfg.cut_height:, :, :]
        label = label[self.cfg.cut_height:, :]

        exist = self.exist_list[idx]

        if self.transform:
            img, label = self.transform((img, label))

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        label = torch.from_numpy(label).contiguous().long()
        meta = {'file_name': self.img[idx]}

        data = {'img': img, 'label': label,
                'exist': exist, 'meta': meta}
        return data
