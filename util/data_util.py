import numpy as np
import random

import torch

from util.voxelize import voxelize


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn_test(batch):
    coord, feat, label, voxel_count, name, idx_sort = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), label, voxel_count, torch.IntTensor(offset), name, idx_sort

def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        try:
            coord, feat, label = transform(coord, feat, label)
        except ValueError:
            print(feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max and ('train' in split):
        init_idx = np.random.randint(label.shape[0])
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]


    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_test(coord, feat, label, voxel_size=0.04, transform=None):
    if transform:
        try:
            coord, feat, label = transform(coord, feat, label)
        except ValueError:
            print(feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx, count, idx_sort  = voxelize(coord, voxel_size, mode=9)
        coord, feat = coord[uniq_idx], feat[uniq_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label, count, idx_sort