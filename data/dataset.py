from pathlib import Path
from torch.utils import data as data
from utils.util import imfrombytes, img2tensor, FileClient
import random
import torch
import os

class MIVRecurrentDataset(data.Dataset):
    def __init__(self):
        super(MIVRecurrentDataset, self).__init__()
        self.gt_root, self.lq_root = Path("datasets/MIV/GT"), Path("datasets/MIV/MIVx4")
        self.num_frame = 15

        self.keys = []
        self.support_view = []
        with open('data/meta_info_MIV_GT.txt', 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}\\{i:08d}\\{frame_num}' for i in range(int(frame_num))])
        
        # file client (io backend)
        self.file_client = None
        self.interval_list = [1]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient('disk')

        key = self.keys[index]
        clip_name, texture, view, frame_name, frame_num = key.split('\\')
        view_list = os.listdir(self.gt_root / clip_name / texture)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        frame_num = int(frame_num)
        if start_frame_idx > frame_num - self.num_frame * interval:
            start_frame_idx = random.randint(0, frame_num - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # get the neighboring LQ and GT frames
        img_lqs = []
        list_lqs = []
        img_gts = []
     
        for neighbor in neighbor_list:
            img_gt_path = self.gt_root / clip_name / texture/ view / f'{neighbor:08d}.jpg'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        for i in view_list:
            for neighbor in neighbor_list:
                img_lq_path = self.lq_root / clip_name / texture/ i / f'{neighbor:08d}.jpg'
                img_bytes = self.file_client.get(img_lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)
                img_lqs.append(img_lq)
            list_lqs.append(img_lqs)



        img_gts, list_lqs = paired_random_crop(img_gts, list_lqs, 256, 4, img_gt_path)
        

        img_gts = torch.stack(img2tensor(img_gts), dim=0)
        list_lqs = [torch.stack(img2tensor(img_lqs), dim=0) for img_lqs in list_lqs]
        list_lqs =  torch.stack(list_lqs, 0)

        return {'lqs': list_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

def paired_random_crop(img_gts, list_lqs, gt_patch_size, scale, gt_path=None):
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = list_lqs[0][0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = list_lqs[0][0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    for img_lqs in list_lqs:
        if input_type == 'Tensor':
            img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]

        else:
            img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]


    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    return img_gts, list_lqs

