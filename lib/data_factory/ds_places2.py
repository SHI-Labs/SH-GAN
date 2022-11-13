import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as tvtrans
import PIL.Image
from PIL import Image, ImageDraw
import math
PIL.Image.MAX_IMAGE_PIXELS = None

from .common import *

from ..log_service import print_log

import numpy.random as npr

@regdataset()
class places2(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        self.mode = cfg.mode

        tagging_normal = {
            'train' : ('data_large', '00train'),
            'challenge' : ('data_challenge', '01challenge'),
            'val'   : ('val_large',   '50val'  ),
            'test'  : ('test_large',  '90test' ),
        }

        tagging_small = {
            'strain' : ('train_large', '01strain'),
            'sval'   : ('val_large',   '51sval'  ),
            'stest'  : ('test_large',  '91stest' ),
        }

        tagging_small_512 = {
            'strain512' : ('train_512', '02strain'),
            'sval512'   : ('val_512',   '52val'  ),
            'stest512'  : ('test_512',  '92test' ),
        }

        tagging_fromtf = {
            'test_fromtf'  : ('test_fromtf/image',  '93test_fromtf' ),
        }

        self.load_info = []
        for m in self.mode.split('+'):
            if m in tagging_normal:
                imdir, maintag = tagging_normal[m]
                imdir = osp.join(self.root_dir, imdir)
            elif m in tagging_small:
                imdir, maintag = tagging_small[m]
                imdir = osp.join(self.root_dir, 'places2_small', imdir)
            elif m in tagging_small_512:
                imdir, maintag = tagging_small_512[m]
                imdir = osp.join(self.root_dir, 'places2_small', imdir)
            elif m in tagging_fromtf:
                imdir, maintag = tagging_fromtf[m]
                imdir = osp.join(self.root_dir, imdir)
            else:
                raise ValueError

            for subdir, _, files in os.walk(imdir):
                for fi in files: 
                    impath = osp.join(subdir, fi)
                    if not (impath.endswith(".jpg") or impath.endswith(".png")):
                        continue

                    tags = [maintag] + subdir.split('/')[4:] + [osp.splitext(fi)[0]]
                    uid = '-'.join(tags)
                    info = {
                        'unique_id': uid, 
                        'filename' : fi,
                        'image_path'  : osp.join(subdir, fi),
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
    """
    Loader with resolution 512
    """
    def __init__(self, resolution=512):
        self.resolution = resolution

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        data = data.resize([self.resolution, self.resolution], PIL.Image.BICUBIC)
        data = tvtrans.ToTensor()(data)
        return data

#############
# formatter #
#############

@regformat()
class DefaultFormatter(object):
    """
    This formatter is a direct replication of the original CoModGan TF code
    """
    def __init__(self, resolution=512):
        self.lod = 0 
        self.resolution = resolution
        # the original code always put a zero here

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2
        if self.lod != 0:
            c, h, w = x.shape
            y = x.view(c, h//2, 2, w//2, 2)        
            y = y.mean(dim=(2, 4), keepdim=True)
            y = y.repeat(1, 1, 2, 1, 2)
            y = y.view(c, h, w)
            x = x + (y-x)*self.lod
        mask = RandomMask(self.resolution)[0]
        return x, mask, element['unique_id']

@regformat()
class CenterMaskFormatter(object):
    """
    This formatter that use the same center mask
    """
    def __init__(self):
        self.latent_dim = 512
        # the original code always put a zero here

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2
        _, h, w = x.shape
        latent = torch.randn([512])
        mask = np.ones([h, w]).astype(np.float32)
        mask[h//4:(h//4+h//2), w//4:(w//4+w//2)] = 0
        return x, latent, mask, element['unique_id']

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
        return x, mask, element['unique_id']

###############################################
# use for evaluation with all image generated #
###############################################
# a) the dataloader that load generated image directly from loader

@regdataset()
class places2_loadgen(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        self.mode = cfg.mode
        self.gen_dir = cfg.gen_dir

        tagging_normal = {
            'train' : ('data_large', '00train'),
            'challenge' : ('data_challenge', '01challenge'),
            'val'   : ('val_large',   '50val'  ),
            'test'  : ('test_large',  '90test' ),
        }

        self.load_info = []
        for m in self.mode.split('+'):
            if m in tagging_normal:
                imdir, maintag = tagging_normal[m]
                imdir = osp.join(self.root_dir, imdir)
            else:
                raise ValueError

            for subdir, _, files in os.walk(imdir):
                for fi in files: 
                    if osp.exists(osp.join(self.gen_dir, fi.replace('.jpg', '.png'))):
                        impath = osp.join(subdir, fi)
                        if not (impath.endswith(".jpg") or impath.endswith(".png")):
                            continue

                        tags = [maintag] + subdir.split('/')[4:] + [osp.splitext(fi)[0]]
                        uid = '-'.join(tags)
                        info = {
                            'unique_id': uid, 
                            'filename' : fi,
                            'image_path'  : osp.join(subdir, fi),
                            'gen_path' : osp.join(self.gen_dir, fi.replace('.jpg', '.png')),
                        }
                        self.load_info.append(info)

@regloader()
class DoubleLoader(object):
    def __init__(self, resolution):
        self.resolution = resolution

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        data = data.resize([self.resolution, self.resolution], PIL.Image.BICUBIC)
        data = tvtrans.ToTensor()(data)
        gen = PIL.Image.open(element['gen_path']).convert('RGB')
        assert (gen.size[0] == self.resolution) and (gen.size[1] == self.resolution)
        element['gen'] = tvtrans.ToTensor()(gen)
        return data

@regformat()
class NoMaskFormatter(object):
    """
    a) The one that do not generate mask.
    b) Direct output the fake result
    """
    def __init__(self):
        pass

    def __call__(self, element):
        x = element['image']
        gen = element['gen']
        return x, gen, element['unique_id']

#######################
# LAMA mask generator #
#######################

from .ds_ffhq import LamaMaskFormatter

