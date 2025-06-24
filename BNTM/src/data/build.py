# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from .autoaugment import AutoAugImageNetPolicy
from scipy import io
from os.path import join
import scipy
import matplotlib.pyplot as plt

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from utils import get_rank, get_world_size
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, list_dir
from .Rareplanes import Rareplanes
from .FARI1M import FAIRI1M
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp

import torchvision
from PIL import Image
def get_birds(augment: bool, train, train_dir: str, project_dir: str, test_dir: str, img_size=224):
    shape = (3, img_size, img_size)
    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    prj = 600 if img_size > 224 else 300

    transform_no_augment = transforms.Compose([
        transforms.Resize((prj, prj), Image.BILINEAR),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    if augment:
        # transform = transforms.Compose([
        #     transforms.Resize(size=(img_size, img_size)),
        #     transforms.RandomOrder([
        #         transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        #         transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
        #     ]),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # transform = transforms.Compose([
        #     transforms.Resize((600, 600), Image.BILINEAR),
        #     transforms.RandomCrop((img_size, img_size)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        transform = transforms.Compose([
            transforms.Resize((prj, prj), Image.BILINEAR),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    transform_explain = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BILINEAR),
        transforms.ToTensor(),
        normalize
    ])

    if train:
        # dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_explain)
    else:
        dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = dataset.classes
    for i in range(len(classes)):
        classes[i] = classes[i].split('.')[1]
    dataset.classes = classes
    return dataset, len(classes)




def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {config.rank} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {config.rank} successfully build val dataset")

    num_tasks = get_world_size()
    global_rank = get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, num_replicas=num_tasks,rank=global_rank,shuffle=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    
    elif config.DATA.DATASET == 'cifar10':
        dataset = datasets.CIFAR10(config.DATA.DATA_PATH, train=is_train, transform=transform)
        nb_classes = 10
    
    elif config.DATA.DATASET == 'Rareplanes':
        train_transform = transforms.Compose([transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
                                              transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_dataset = Rareplanes(transform=train_transform, train=is_train)
        nb_classes = 7
    
    elif config.DATA.DATASET == 'FAIR1M':
        train_transform = transforms.Compose([transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
                                              transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_dataset = FAIRI1M(transform=train_transform, is_train=is_train)
        nb_classes = 10

    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return train_dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
