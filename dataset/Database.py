import pickle
import numpy as np
from PIL import Image
import os

import torch
# from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
# from torchvision.transforms.functional import resize

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

join = os.path.join

# 数据集读取及处理

class Database(Dataset):
    def __init__(self, keys, args, mode='train'):
        super().__init__()
        self.patch_size = (320, 640)
        self.files = []
        self.mode = mode

        for f in os.listdir(args.data_dir):
            if f in keys:
                self.files.append(join(args.data_dir,f))

        print(f'dataset length: {len(self.files)}')

    def __len__(self):
        return len(self.files)
    
    # 对数据进行预处理，有两种方法，分别为transform_contrast和transform
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = Image.open(self.files[index].replace('image/', 'mask/'))
        label = np.asarray(label)
        label2 = label.astype(np.uint8)
        label2[label2==255]=1
        # label2[label2==160]=2
        # （H，W，C)-->(C，H，W)
        img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
        img = (img - img.min()) / (img.max() - img.min())

        img, label = self.transform(img, label2)
        return img, label

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                 patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]

        # if img.shape[0]==4:
        #     import torch
        #     import matplotlib.pyplot as plt
        #
        #     # 假设img是一个torch.Tensor类型的图像
        #     # img = torch.randn(3, 256, 256)  # 替换为你的图像
        #
        #     # 将图像转换为NumPy数组
        #     img_np = img.numpy()
        #
        #     # 调整图像通道顺序，从 (channel, height, width) 到 (height, width, channel)
        #     img_np = img_np.transpose(1, 2, 0)
        #
        #     # 显示图像
        #     plt.imshow(img_np)
        #     plt.axis('off')  # 可选，关闭坐标轴
        #     plt.show()

        label = data_dict.get('seg')[0]
        return img, label


class Database_teech(Dataset):
    def __init__(self, keys, args, mode='train'):
        super().__init__()
        self.patch_size = (args.img_size, 2*args.img_size)
        self.files = []
        self.mode = mode

        for f in os.listdir(args.data_dir):
            if f in keys:
                self.files.append(join(args.data_dir, f))

        print(f'dataset length: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    # 对数据进行预处理，有两种方法，分别为transform_contrast和transform
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = Image.open(self.files[index].replace('image/', 'mask/'))
        label = np.asarray(label)
        label2 = label.astype(np.uint8)
        label2[label2 == 1] = 1
        # label2[label2 == 160] = 2
        # （H，W，C)-->(C，H，W)
        img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
        img = (img - img.min()) / (img.max() - img.min())

        img, label = self.transform(img, label2)
        return img, label

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                 patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]


        return img, label



# class Database_pretict(Dataset):
#     def __init__(self, keys, args, mode='train'):
#         super().__init__()
#         self.patch_size = (args.img_size, args.img_size)
#         self.files = []
#         self.mode = mode
#
#         for f in os.listdir(args.data_dir):
#             if f in keys:
#                 self.files.append(join(args.data_dir, f))
#
#         print(f'dataset length: {len(self.files)}')
#
#     def __len__(self):
#         return len(self.files)
#
#     # 对数据进行预处理，有两种方法，分别为transform_contrast和transform
#     def __getitem__(self, index):
#         img = Image.open(self.files[index])
#         # label = Image.open(self.files[index].replace('image/', 'mask/'))
#         # label = np.asarray(label)
#         # label2 = label.astype(np.uint8)
#         # label2[label2 == 80] = 1
#         # label2[label2 == 160] = 2
#         # （H，W，C)-->(C，H，W)
#         img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
#         img = (img - img.min()) / (img.max() - img.min())
#
#         img = self.transform(img)
#         return img
#
#     def transform(self, img ,label ):
#         # normalize to [0, 1]
#         data_dict = {'data': img[None], 'seg': label[None, None]}
#         if self.mode == 'train':
#             aug_list = [  # CenterCropTransform(crop_size=target_size),
#                 BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
#                 GammaTransform(p_per_sample=0.5),
#                 GaussianNoiseTransform(p_per_sample=0.5),
#                 ResizeTransform(target_size=self.patch_size, order=1),  # resize
#                 MirrorTransform(axes=(1,)),
#                 SpatialTransform(patch_size=self.patch_size, random_crop=False,
#                                  patch_center_dist_from_border=self.patch_size[0] // 2,
#                                  do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
#                                  do_rotation=True, p_rot_per_sample=0.5,
#                                  angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
#                                  scale=(0.5, 1.9), p_scale_per_sample=0.5,
#                                  border_mode_data="nearest", border_mode_seg="nearest"),
#                 NumpyToTensor(),
#             ]
#
#             aug = Compose(aug_list)
#         else:
#             aug_list = [
#                 ResizeTransform(target_size=self.patch_size, order=1),
#                 NumpyToTensor(),
#             ]
#             aug = Compose(aug_list)
#
#         data_dict = aug(**data_dict)
#         img = data_dict.get('data')[0]
#         label = data_dict.get('seg')[0]
#         return img , label

class Database_pretict(Dataset,):
    def __init__(self, keys, args, size):
        super().__init__()
        self.patch_size = (320, 640)
        # self.patch_size = size

        self.files = []
        for f in os.listdir(args.data_dir):
            if f in keys:
                self.files.append(join(args.data_dir, f))
        print(f'dataset length: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        # （H，W，C)-->(C，H，W)
        img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
        img = (img - img.min()) / (img.max() - img.min())
        img = self.transform(img)
        return img

    def transform(self, img):
        # normalize to [0, 1]
        data_dict = {'data': img[None]}
        aug_list = [
            ResizeTransform(target_size=self.patch_size, order=1),
            NumpyToTensor(),
        ]
        aug = Compose(aug_list)
        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        return img
