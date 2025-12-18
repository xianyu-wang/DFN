import sys
import torch
from thop import profile

from util.stpls import STPLS
from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

data_root = r'E:\Develop\ALS_Dataset\STPLS3D\block'
test_area = 3
val_data = STPLS(split='val', data_root=data_root, test_area=test_area, voxel_size=0.5, voxel_max=800000, transform=None)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, sampler=None, collate_fn=collate_fn)

model = Model(c=6, k=6)
model = model.cuda()

for i, (coord, feat, target, offset) in enumerate(val_loader):
    coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
    if target.shape[-1] == 1:
        target = target[:, 0]
    with torch.no_grad():
        macs, params = profile(model, inputs=([coord, feat, offset],))
        print(coord.shape)
        print(offset.shape)
        print('MACs:',macs)
        print('Paras:',params)
        break