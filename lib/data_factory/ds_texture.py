import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
from torch.functional import align_tensors
import torchvision.transforms as tvtrans
import PIL.Image
from PIL import Image, ImageDraw
import math
import copy
PIL.Image.MAX_IMAGE_PIXELS = None

from lib import visual_service as vis

from .common import *

from ..log_service import print_log

import numpy.random as npr

@regdataset()
class texture(ds_base):

    def get_imagelist(self, listname):
        with open(osp.join(
                self.root_dir, 'dtd', 'labels', listname+'.txt')) as f:
            list = f.readlines()
        list = [li.strip() for li in list]
        return list

    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        mode = cfg.mode
        self.mode = mode
        mode = mode.split('+')

        imlist = []
        for modei in mode:
            if modei.find('train') != 0:
                pass
            elif modei.find('val') != 0:
                pass
            elif modei.find('test') != 0:
                pass
            else:
                raise ValueError
            imlist += self.get_imagelist(modei)

        self.load_info = []
        for imref in imlist:
            texture_type, filename = osp.split(imref)
            uid = osp.splitext(filename)[0]
            info = {
                'unique_id'   : uid, 
                'filename'    : filename,
                'texture_type': texture_type,
                'image_path'  : osp.join(
                    self.root_dir, 'dtd', 'images', texture_type, filename),
            }
            self.load_info.append(info)

        # here we can choose to mixed the pattern order
        # so that the visalization on first couple of images
        # can cover a wide range of patterns
        mixed_order = getattr(
            cfg, 'mixed_order_on_texture_type', False)
        if mixed_order:
            group = {}
            for loadi in self.load_info:
                tt = loadi['texture_type']
                if tt in group:
                    group[tt].append(loadi)
                else:
                    group[tt] = [loadi]

            cnt = 0
            self.load_info = []
            while len(group) > 0:
                for pi in list(group.keys()):
                    gi = group[pi]
                    if len(gi) == 0:
                        group.pop(pi)
                        continue
                    loadi = copy.deepcopy(gi[0])
                    loadi['unique_id'] = '{:05d}_'.format(cnt) + loadi['unique_id']
                    self.load_info.append(loadi)
                    cnt += 1
                    group[pi] = gi[1:]

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
class InpaintingFormatter(object):
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
        if npr.random() < 0.5:
            x = x.flip(1)
        if npr.random() < 0.5:
            x = x.flip(2)
        mask = RandomMask(s, self.hole_range)[0]
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

def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

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
