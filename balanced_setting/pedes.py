import glob
import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from tools.function import get_pkl_rootpath


class PedesAttr(data.Dataset):

    def __init__(self, cfg, split, transform=None, target_transform=None, idx=None, balance = False):

        assert cfg.DATASET.NAME in ['PETA', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {cfg.DATASET.NAME} is not exist'

        data_path = get_pkl_rootpath(cfg.DATASET.NAME, cfg.DATASET.ZERO_SHOT)

        print("which pickle", data_path)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name

        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        if 'label_idx' not in dataset_info.keys():
            print(' this is for zero shot split')
            assert cfg.DATASET.ZERO_SHOT
            self.eval_attr_num = self.attr_num
        else:
            self.eval_attr_idx = dataset_info.label_idx.eval
            self.eval_attr_num = len(self.eval_attr_idx)

            assert cfg.DATASET.LABEL in ['all', 'eval', 'color'], f'key word {cfg.DATASET.LABEL} error'
            if cfg.DATASET.LABEL == 'eval':
                attr_label = attr_label[:, self.eval_attr_idx]
                self.attr_id = [self.attr_id[i] for i in self.eval_attr_idx]
                self.attr_num = len(self.attr_id)
            elif cfg.DATASET.LABEL == 'color':
                attr_label = attr_label[:, self.eval_attr_idx + dataset_info.label_idx.color]
                self.attr_id = [self.attr_id[i] for i in self.eval_attr_idx + dataset_info.label_idx.color]
                self.attr_num = len(self.attr_id)

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = cfg.DATASET.NAME
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        if self.target_transform:
            self.attr_num = len(self.target_transform)
            print(f'{split} target_label: {self.target_transform}')
        else:
            self.attr_num = len(self.attr_id)
            print(f'{split} target_label: all')

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0

        if idx is not None:
            self.img_idx = idx

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]  # [:, [0, 12]]

        if balance == True:
            if split == 'test':
                self.label_min = np.where((self.label.shape[0] - self.label.sum(0)) < self.label.sum(0), self.label.shape[0] - self.label.sum(0), self.label.sum(0))
                self.count_p_n = np.zeros([self.label.shape[1],2])
                self.label_mask_bal = np.zeros(self.label.shape)

                for j in range(self.label.shape[1]):
                    for i in range(self.label.shape[0]):
                        if self.label[i,j] == 0:
                            if self.count_p_n[j,0] < self.label_min[j]:
                                self.count_p_n[j,0] += 1
                                self.label_mask_bal[i,j] = 1
                        else:
                            if self.count_p_n[j,1] < self.label_min[j]:
                                self.count_p_n[j,1] += 1
                                self.label_mask_bal[i,j] = 1

                #print(((self.label_mask_bal * self.label) == 1).sum(0) - ((self.label_mask_bal * (1-self.label)) == 1).sum(0))
                
                self.zero_out = np.where(self.label_mask_bal.sum(1) != 0)[0]

                self.img_id = [self.img_id[i] for i in self.zero_out]
                self.label = self.label[self.zero_out]
                self.label_mask_bal = self.label_mask_bal[self.zero_out]

                self.img_num = self.label.shape[0]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]

        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname,  # noisy_weight

    def __len__(self):
        return len(self.img_id)
