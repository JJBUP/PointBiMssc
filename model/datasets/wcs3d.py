"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from model.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_20, VALID_CLASS_IDS_200


@DATASETS.register_module()
class WCS3DDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(self,
                 split='train',
                 data_root='data/WCS3DDataset',
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 save_complete_pred=True,
                 cache=False,
                 # work_key_word="Work_1",  # todo
                 loop=1):
        super(WCS3DDataset, self).__init__()
        self.save_complete_pred = save_complete_pred
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = loop if not test_mode else 1  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        # self.work_key_word = work_key_word

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            # data_list = glob.glob(os.path.join(self.data_root, self.split, f"{self.work_key_word}*.pth"))  # todo
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                # data_list += glob.glob(os.path.join(self.data_root, split, f"{self.work_key_word}*.pth"))  # todo
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        # data_list.sort()
        data_list.sort(reverse=True)
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        if data.shape[1] > 6:
            segment = data[:, 6].reshape([-1]).astype(np.int_)
        else:
            segment = np.ones(data.shape[0]) * -1
        data_dict = dict(coord=data[:, :3], color=data[:, 3:6], segment=segment)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        ori_data = np.array([])
        if self.save_complete_pred:
            ori_data = np.concatenate([
                data_dict["coord"],
                # data_dict["color"],
                # data_dict["normal"]
            ], axis=1)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                aug(deepcopy(data_dict))
            )

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return input_dict_list, segment, ori_data

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
