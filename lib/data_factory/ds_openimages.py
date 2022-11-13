import os
import copy
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as tvtrans
import PIL.Image
from PIL import Image, ImageDraw
import math
PIL.Image.MAX_IMAGE_PIXELS = None

from lib import visual_service as vis

from .common import *

from ..log_service import print_log

import numpy.random as npr

@regdataset()
class openimages(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        self.mode = cfg.mode

        self.load_info = []

        if self.mode == "train":
            imdir = osp.join(self.root_dir, "train")
        elif self.mode == "val":
            imdir = osp.join(self.root_dir, "validation")
        else:
            raise ValueError

        for subdir, _, files in os.walk(imdir):
            for fi in files:
                impath = osp.join(subdir, fi)
                if not (impath.endswith(".jpg") or impath.endswith(".png")):
                    continue

                tags = subdir.split('/')[4:] + [osp.splitext(fi)[0]]
                uid = '-'.join(tags)
                info = {
                    'unique_id': uid,
                    'filename': fi,
                    'image_path': osp.join(subdir, fi),
                }
                self.load_info.append(info)

@regloader()
class DefaultLoader(object):
    def __init__(self):
        super().__init__()

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        data = tvtrans.ToTensor()(data)
        return data

@regloader()
class FixResolutionLoader(object):
    def __init__(self, resolution=512):
        self.resolution = resolution

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        if data.size[0] > self.resolution or data.size[1] > self.resolution:
            w, h = data.size
            resize_ratio = self.resolution / w if w > h else self.resolution / h
            target_size = (self.resolution, int(h * resize_ratio)) if w > h else (int(w * resize_ratio), self.resolution)
            data = data.resize(target_size, resample=PIL.Image.BICUBIC)

        content_size = data.size[::-1]

        padded_data = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        padded_data[:data.size[1], :data.size[0], :] = np.array(data, dtype=np.uint8)
        data = tvtrans.ToTensor()(padded_data)
        element['content_size'] = content_size
        return data

########################
# RandomBrush for mask #
########################

from .ds_ffhq import RandomMask

##############
# fixed mask #
##############

@regformat()
class FixedMaskFormatter(object):
    """
    This formatter is a direct replication of the original CoModGan TF code
    """
    def __init__(self):
        self.lod = 0
        self.latent_dim = 512
        # the original code always put a zero here

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2
        latent = torch.randn([512])
        mask = element['image_path'].replace('image/', 'mask/').replace('.png', '_mask.png')
        mask = np.array(PIL.Image.open(mask))>128
        mask = mask.astype(np.float32)
        return x, latent, mask, element['unique_id']

##########################################################
# inpainting formatter with random scale and random crop #
##########################################################

@regformat()
class AdvInpaintingFormatter(object):
    """
    This formatter perform the following operation.
    a) Random scale the image h w so it is in a designated range = [1, 1.2] or larger.
    i.e. If target resolution = 256, image size = [480, 640]
         imsize after resize is [np.uniform(256, max(480, 256*1.2), max(640, 256*1.2))]
    """
    def __init__(self, resolution=512, hole_range=[0, 1]):
        self.resolution = resolution
        self.hole_range = hole_range

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2 # Normalize to -1 to 1
        _, oh, ow = x.shape
        s = self.resolution
        nh = npr.randint(s, max(oh, int(s*1.2))+1)
        nw = npr.randint(s, max(ow, int(s*1.2))+1)
        ch, cw = npr.randint(0, nh-s+1), npr.randint(0, nw-s+1)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=[nh, nw], mode='bicubic', align_corners=False)
        x = x.squeeze(0)[:, ch:ch+s, cw:cw+s]
        mask = RandomMask(s, self.hole_range)[0]
        return x, mask, element['unique_id']

#############################################################
# inpainting formatter without random scale and random crop #
#############################################################

@regformat()
class FreeFormMaskFormatter(object):
    """
    This formatter perform the following operation.
    a) Rescale the image h w to the designated resolution.
    """
    def __init__(self, random_flip=True, resolution=512, hole_range=[0, 1]):
        self.random_flip = random_flip
        self.resolution = resolution
        self.hole_range = hole_range

    def __call__(self, element):
        x = (element['image']*2 - 1)
        if (self.random_flip) and (npr.rand() < 0.5):
            x = x.flip(-1)
        mask = RandomMask(self.resolution, self.hole_range)[0]
        im_h, im_w = element['content_size']
        mask[0:, im_w:] = 1.0
        mask[im_h:, 0:] = 1.0
        return x, mask, element['unique_id']

#######################
# LAMA mask generator #
#######################

from .ds_ffhq import LamaMaskFormatter

