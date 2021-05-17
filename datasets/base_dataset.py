import os.path as osp
import os
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
        self.is_training = ('train' in data_list)

        self.img_name_list = []
        self.full_img_path_list = []
        self.label_list = []
        self.exist_list = []

        self.transform = self.transform_train() if self.is_training else self.transform_val()

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

    def view(self, img, coords, file_path=None):
        for coord in coords:
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

        if file_path is not None:
            if not os.path.exists(osp.dirname(file_path)):
                os.makedirs(osp.dirname(file_path))
            cv2.imwrite(file_path, img)


    def init(self):
        raise NotImplementedError()


    def __len__(self):
        return len(self.full_img_path_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.full_img_path_list[idx]).astype(np.float32)
        img = img[self.cfg.cut_height:, :, :]

        if self.is_training:
            label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            exist = self.exist_list[idx]
            if self.transform:
                img, label = self.transform((img, label))
            label = torch.from_numpy(label).contiguous().long()
        else:
            img, = self.transform((img,))

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        meta = {'full_img_path': self.full_img_path_list[idx],
                'img_name': self.img_name_list[idx]}

        data = {'img': img, 'meta': meta}
        if self.is_training:
            data.update({'label': label, 'exist': exist})
        return data
