import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util.ply import write_ply, read_ply
from util.stpls import STPLS_test
from util.data_util import collate_fn_test
# random.seed(123)
# np.random.seed(123)

train_sequences = ['Synthetic_v1', 'Synthetic_v2', 'Synthetic_v3', 'RealWorldData']
cvalid_sequences = ['OCCC_points', 'RA_points', 'USC_points', 'WMSC_points']
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/stpls/stpls_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/stpls/stpls_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def changeSemLabels(cloud):
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 2) &  (cloud[:, 6:7] <= 4), 2, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 5) &  (cloud[:, 6:7] <= 6), 3, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 8), 3, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 11) &  (cloud[:, 6:7] <= 12), 4, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 14), 5, cloud[:, 6:7])

    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 7) &  (cloud[:, 6:7] <= 10), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 13), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 15) &  (cloud[:, 6:7] <= 16), 0, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 17), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >17), 0, cloud[:, 6:7])
    
    return cloud
def ply2array(ply_path):
    cloud = read_ply(ply_path)
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'], cloud['red'], cloud['green'], cloud['blue'], cloud['class'])).T
    cloud = changeSemLabels(cloud)
    return cloud

def get_logger(log_file):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    args.model_path = os.path.join(args.exp_dir, 'model', '{}.pth'.format(args.epoch))
    args.save_folder = os.path.join(args.exp_dir, 'result', args.epoch)
    os.makedirs(args.save_folder, exist_ok=True)
    log_file = os.path.join(args.exp_dir, 'result', args.epoch, 'test.log')
    
    logger = get_logger(log_file)
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from model.DFKNN.pointtransformer_seg import  pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'stpls':
        data_list = []
        for sq in train_sequences:
            data_list += os.listdir(args.data_root + sq)
        data_list = [item for item in data_list if '{}_'.format(cvalid_sequences[args.test_area]) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    # data_path = os.path.join(args.data_root, data_name + '.npy')
    
    # data = np.load(data_path)  # xyzrgbl, N*7
    data_path = os.path.join(args.data_root, "RealWorldData", data_name)
    data = ply2array(data_path)
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    args.batch_size_test = 10
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    
    val_transform = None
    val_data = STPLS_test(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_test)
    
    model.eval()
    file_list = []
    pred_save = []
    label_save = []
    for i, (coord, feat, target, voxel_count, offset, filenames, idx_sort) in enumerate(val_loader):
        coord, feat, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)

        with torch.no_grad():
            output = model([coord, feat, offset])
        output = output.max(1)[1].cpu().numpy()
        offset = np.insert(offset.cpu().numpy(), 0, 0)
        for j, filename in enumerate(filenames):
            voxel_pred = output[offset[j]:offset[j+1]]
            pred_sort = np.zeros(len(idx_sort[j]))
            pred_sort[idx_sort[j]] = np.repeat(voxel_pred, voxel_count[j])
            file_list.append(filename)
            pred_save.append(pred_sort)
            label_save.append(target[j].cpu().numpy())
 

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "filename.pickle"), 'wb') as handle:
        pickle.dump({'label': file_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    # logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
