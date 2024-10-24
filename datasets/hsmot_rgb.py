from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances

from random import choice, randint



class DetHSMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.split_dir = os.path.join(args.mot_path, "train", "rgb")
        self.labels_dir = os.path.join(args.mot_path, "train", "mot")

        self.labels_full = defaultdict(lambda: defaultdict(list))
        for vid in os.listdir(self.labels_dir):
            gt_path = os.path.join(self.labels_dir, vid)
            for l in open(gt_path):
                t, i, *x0y0x1y1x2y2x3y3, _, cls = l.strip().split(',')[:12]
                # t, i, *xywh, mark, label = l.strip().split(',')[:8]
                # t, i, mark, label = map(int, (t, i, mark, label))
                t, i, cls = map(int, (t, i, cls))
                # if mark == 0:
                    # continue
                # if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person labels, adjust if needed for HSMOT RGB
                    # continue
                # else:
                    # crowd = False
                x0, y0, x1, y1, x2, y2, x3, y3 = map(float, (x0y0x1y1x2y2x3y3))
                self.labels_full[vid][t].append([x0, y0, x1, y1, x2, y2, x3, y3, i, cls])

        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.split_dir, osp.splitext(vid)[0], f'{idx:06d}.png')
        img = Image.open(img_path)
        targets = {}
        w, h = img.size
        assert w > 0 and h > 0, f"invalid image {img_path} with shape {w} {h}"
        obj_idx_offset = self.video_dict[vid] * 100000

        targets['dataset'] = 'HSMOT_RGB'
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for *x0y0x1y1x2y2x3y3, id, cls in self.labels_full[vid][idx]:
            targets['boxes'].append(x0y0x1y1x2y2x3y3)
            area = 0.5 * abs(
                (x0y0x1y1x2y2x3y3[0] * x0y0x1y1x2y2x3y3[3] + x0y0x1y1x2y2x3y3[2] * x0y0x1y1x2y2x3y3[5] + x0y0x1y1x2y2x3y3[4] * x0y0x1y1x2y2x3y3[1]) -
                (x0y0x1y1x2y2x3y3[1] * x0y0x1y1x2y2x3y3[2] + x0y0x1y1x2y2x3y3[3] * x0y0x1y1x2y2x3y3[4] + x0y0x1y1x2y2x3y3[5] * x0y0x1y1x2y2x3y3[0])
            )
            targets['area'].append(area)
            targets['iscrowd'].append(False)
            targets['labels'].append(cls)
            targets['obj_ids'].append(id + obj_idx_offset)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 8)
        # targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        return img, targets

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        images, targets = self.pre_continuous_frames(vid, indices)
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        if transform is not None:
            images, targets = transform(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.size())
            gt_instances.append(gt_instances_i)
        return {
            'imgs': images,
            'gt_instances': gt_instances,
        }
        
        # return images, gt_instances
        # 重组成[(img1, tar1), (img2, tar2)...]的形式
        # return [(img, tar) for img, tar in zip(images, gt_instances)]


    def __len__(self):
        return len(self.indices)

def build_dataset2transform(args, image_set):
    hsmot_train = make_transforms_for_hsmot_rgb('train', args)
    hsmot_test = make_transforms_for_hsmot_rgb('val', args)

    dataset2transform_train = {'HSMOT_RGB': hsmot_train}
    dataset2transform_val = {'HSMOT_RGB': hsmot_test}
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetHSMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetHSMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset

def make_transforms_for_hsmot_rgb(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.RotateMotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.RotateMotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.RotateMotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.RotateMotRandomResize([800, 1000, 1200]),
                    T.RotateFixedMotRandomCrop(800, 1200),
                    T.RotateMotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.RotateMotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

