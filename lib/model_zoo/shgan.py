import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr
import copy

from lib.model_zoo.common.get_model import get_model, register
from lib.model_zoo.common import utils
from . import get_optimizer

version = '0'
symbol = 'shgan'

from .comodgan import Encoder as Encoder_base

class one_hot_2d(object):
    """
    Convert a [bs x 2D] ID array into an one hot 
        [bs x max_dim x 2D] binary array. 
    """
    def __init__(self, 
                 max_dim=None, 
                 ignore_label=None, 
                 **kwargs):
        """
        Args:
            max_dim: An integer tells largest one_hot dimension.
                None means auto-find the largest ID as max_dim.
            ignore_label: An integer or array tells the ignore ID(s).
        """
        self.max_dim = max_dim
        if isinstance(ignore_label, int):
            self.ignore_label = [ignore_label]
        elif ignore_label is None:
            self.ignore_label = []
        else:
            self.ignore_label = ignore_label

    def __call__(self, 
                 x, 
                 mask=None):

        if mask is not None:
            x *= mask==1

        check = []
        for i, n in enumerate(np.bincount(x.flatten())):
            if (i not in self.ignore_label) and (n>0):
                check.append(i)
        
        if self.max_dim is None: 
            max_dim = check[-1]+1
        else:
            max_dim = self.max_dim
        batch_n, h, w = x.shape

        oh = np.zeros((batch_n, max_dim, h, w)).astype(np.uint8)
        for c in check:
            if c >= max_dim:
                continue
            oh[:, c, :, :] = x==c

        if mask is not None:
            # remove the unwanted one-hot zeros
            oh[:, 0, :, :] *= mask==1
        return oh

def make_cweight(half_size, half_sample, type='piecewise_linear', oddeven_aligned=True, device='cpu'):
    """
    Make a coordinate based weighting.
    Args:
        half_size: [int, int], 
            the size of height and width, 
            height will be normalized to -1 to 1
            width will be normalized to 0 to 1
        half_sample: [int, int],
            the size of height and width on sampling
        type: str, 
            tells how to sample these points:
        oddeven_aligned: bool,
            If half_sample use the oddeven align rule, and if height is even number
                it means it will not be noamlized to exact [-1, 1],
                but to [-1+2/height_sample, 1], 
                so the [0, 0] origin is at [height_sample//2+1, 0]
    Outputs:
        cweight:
            a one-hot on each location of the half_size matrix will be prepared.
            a grid sampling based on the type will be perform on half_sampling grid.
            the result is cweight
    """

    h0, w0 = half_size
    hs, ws = half_sample
    
    reference_id = np.array([i for i in range(h0*w0)]).reshape(1, h0, w0)
    reference_oh = one_hot_2d(max_dim=h0*w0)(reference_id)
    reference_oh = torch.Tensor(reference_oh).float().to(device)

    # expand so the reference is on the whole [-1, 1]^2 plane
    reference_oh = F.pad(reference_oh, pad=(w0-1, 0, 0, 0), mode='reflect')

    if oddeven_aligned and hs%2 == 0:
        h_grid = np.array([-1 + i/(hs)*2 for i in range(hs+1)])[1:]
    else:
        h_grid = np.array([-1 + i/(hs-1)*2 for i in range(hs)])
    w_grid = np.array([ 0 + i/(ws-1)   for i in range(ws)])
    w_grid, h_grid = np.meshgrid(w_grid, h_grid)
    grid = np.stack([w_grid, h_grid], axis=-1) # format '[h x w x wh]'
    grid = torch.Tensor(grid).float().unsqueeze(0).to(device)

    if type == 'piecewise_linear':
        cweight = F.grid_sample(reference_oh, grid, mode='bilinear', padding_mode='border', align_corners=True)
        cweight = cweight.squeeze(0)
    elif type == 'bicubic':
        cweight = F.grid_sample(reference_oh, grid, mode='bicubic', padding_mode='border', align_corners=True)
        cweight = cweight.squeeze(0)
    else:
        raise NotImplementedError
    return cweight

class heterogeneous_filter(nn.Module):
    def __init__(self, in_channels, out_channels, freedom, type, init='ones'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freedom = freedom
        self.type = type
        self.cw_cache = None
        self.size_cache = None

        if type in ['piecewise_linear', 'bicubic']:
            fh, fw = freedom 
            self.weight = nn.Parameter(
                torch.empty(in_channels, out_channels*fh*fw), requires_grad=True)
        else:
            raise NotImplementedError

        if init == 'ones':
            nn.init.ones_(self.weight)

    def forward(self, x):
        bs, c, h, w = x.shape
        if self.cw_cache is not None and self.size_cache == x.shape:
            cw = self.cw_cache
        elif self.type in ['piecewise_linear', 'bicubic']:
            cw = make_cweight(
                half_size=self.freedom, 
                half_sample=x.shape[2:], 
                type=self.type, 
                oddeven_aligned=True,
                device=x.device)
            self.cw_cache = cw
            self.size_cache = x.shape

        weight = self.weight.T.unsqueeze(-1).unsqueeze(-1)
        y = F.conv2d(x, weight).view(bs, c, -1, h, w)
        o = (y * cw.unsqueeze(0).unsqueeze(0)).sum(2)
        return o

class gaussian_heatmap_2d(object):
    """
    Given a [n x 2] center point coord,
        and [n x 2 x 2] gaussian sigma (std)
        Compute [h x w] np array
        which is 1 at gaussian center and difuss to
        elsewhere as the sigma shows.
    """
    def __init__(self, size, merge_type='max'):
        """
        Args:
            size: (int, int),
                h, w of the output size.
            merge_type: str,
                'max', output value will take the max one
                'add', add values from multiple gaussians.
                    (GMM is such)
        """
        self.size = size
        self.merge_type = merge_type
        h, w = size
        coordh = np.arange(0, h)[:, np.newaxis] * np.ones((1, w))
        coordw = np.arange(0, w)[np.newaxis, :] * np.ones((h, 1))
        self.coord = np.stack([coordh, coordw])
        self.speedup = True

    def __call__(self, c, v):
        """
        Args:
            c: [n x 2] float array,
                the center of each gaussian.
            v: [n x 2 x 2] float array,
                the 2x2 variance matrices on each gaussian.
        Returns:
            x: [h x w] float array,
                the output gaussian heatmap.
        """
        if c.shape[0] != v.shape[0]:
            raise ValueError
        x = np.zeros(self.size, dtype=float)
        for ci, vi in zip(c, v):
            ci = ci[:, np.newaxis, np.newaxis]
            dx = self.coord-ci

            if self.speedup:
                # for speed up
                # only update the value within -1.5sigma <-> 1.5sigma
                try:
                    _, singv, _ = np.linalg.svd(vi)
                except:
                    continue
                singvmax = np.max(singv) 
                # this is the variance on the max spread direction
                maxstd = np.sqrt(singvmax) # this is the std
                searchr = int(3*maxstd+1) 
                # from -searchr:searchr is the range of search
                ciint = ci.astype(int)
                chint, cwint = ciint[0, 0, 0], ciint[1, 0, 0]
                searchh = [max(min(i, self.size[0]), 0) \
                    for i in [chint-searchr, chint+searchr]] 
                searchw = [max(min(i, self.size[1]), 0) \
                    for i in [cwint-searchr, cwint+searchr]]
                sh, sw = searchh[1]-searchh[0], searchw[1]-searchw[0] 
                dx = dx[:, searchh[0]:searchh[1], searchw[0]:searchw[1]]
                if sh==0 or sw==0:
                    continue
                # a slide of x
                xref = x[searchh[0]:searchh[1], searchw[0]:searchw[1]]
            else:
                xref = x
                sh, sw = self.size

            try:
                vi_inv = np.linalg.inv(vi)
            except:
                continue
            dx = dx.transpose(1, 2, 0).reshape(-1, 2)
            xi = dx @ vi_inv
            xi = (xi * dx).sum(-1)
            xi = xi.reshape(sh, sw)
            xi = np.exp(-0.5*xi)
            if self.merge_type == 'max':
                # update the memory
                xref[:, :] = np.maximum(xref, xi)
            elif self.merge_type == 'add':
                xref[:, :] = xref + xi
            else:
                raise ValueError
        return x

class SHU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dfilter_freedom=[3, 2], 
                 dfilter_type='piecewise_linear', 
                 input_res = 256,
                 lowest_res = 4,
                 tail_sigma_mult = 3,
                 gaussian_at_input_res = False,):
        '''
        Args:
            tail_sigma_mult: float, 
                tells how much does the sigma extend. 
        '''
        super().__init__()
        from .stylegan import conv2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_res = input_res
        self.lowest_res = lowest_res
        self.conv0 = conv2d(in_channels*2, in_channels*2, 1, 1, 0)
        self.df1 = heterogeneous_filter(
            in_channels*2, out_channels*2, 
            freedom=dfilter_freedom, type=dfilter_type)
        torch.nn.init.normal_(self.df1.weight, mean=1/(out_channels*2), std=0.1/(out_channels*2))
        self.act = torch.nn.ReLU(inplace=True)

        self.tail_sigma_mult = tail_sigma_mult
        self.gaussian_at_input_res = gaussian_at_input_res

        self.reslist = [2**i for i in range(int(np.log2(self.lowest_res)), int(np.log2(self.input_res))+1)]
        reslistrev = self.reslist[::-1]
        self.gaussian_weight_map = {}
        for idx, resi in enumerate(reslistrev):
            if idx != 0:
                gaussianf = gaussian_heatmap_2d(size=[resi, resi//2+1])
                center = np.array([resi//2-1, 0], dtype=float)
                sigma = (resi//2)/tail_sigma_mult
                variance = np.array([
                    [sigma**2, 0],
                    [0, sigma**2],], dtype=float)
                self.gaussian_weight_map[resi] = gaussianf(c=center[None], v=variance[None])
                resi_prev = reslistrev[idx-1]
                self.gaussian_weight_map[resi_prev][
                        (resi_prev//2-resi//2):(resi_prev//2+resi//2), 0:(resi//2+1)] \
                    -= self.gaussian_weight_map[resi]
            elif gaussian_at_input_res:
                gaussianf = gaussian_heatmap_2d(size=[resi, resi//2+1])
                center = np.array([resi//2-1, 0], dtype=float)
                sigma = (resi//2)/tail_sigma_mult
                variance = np.array([
                    [sigma**2, 0],
                    [0, sigma**2],], dtype=float)
                self.gaussian_weight_map[resi] = gaussianf(c=center[None], v=variance[None])
            else:
                self.gaussian_weight_map[resi] = torch.ones([resi, resi//2+1]).float()


        for resi in self.reslist:
            self.gaussian_weight_map[resi] = torch.Tensor(self.gaussian_weight_map[resi]).float()

    def forward(self, x):
        ffted = torch.fft.rfftn(x, dim=(2, 3), norm='forward')
        # Shift is necessary because the top-left is low frequency (make it at center)
        ffted = torch.cat([
            ffted[:, :, ffted.size(2)//2+1:, :],
            ffted[:, :, :ffted.size(2)//2+1, :]], dim=2)

        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.conv0(ffted)
        ffted = self.act(ffted)
        ffted = self.df1(ffted)
        ffted = torch.complex(ffted[:, 0:self.out_channels], ffted[:, self.out_channels:])

        output = {}
        
        for resi in self.reslist:
            splited_ffted = ffted[:, :, (self.input_res//2-resi//2):(self.input_res//2+resi//2), 0:(resi//2+1)].clone()
            splited_ffted = splited_ffted * self.gaussian_weight_map[resi].to(x.device)[None, None]
            # Shift back
            splited_ffted = torch.cat([
                splited_ffted[:, :, resi-resi//2-1:, :],
                splited_ffted[:, :, :resi-resi//2-1, :]], dim=2)
            output[resi] = torch.fft.irfftn(splited_ffted, dim=(2, 3), norm='forward')# /10*resi

        return output

@register('shgan_encoder', version)
class Encoder(Encoder_base):
    def __init__(self,
                 *args,
                 **kwargs,):
        self.shu_input_res = kwargs.pop('shu_input_res')
        self.shu_lowest_res = kwargs.pop('shu_lowest_res')
        self.shu_channels = kwargs.pop('shu_channels')
        self.shu_df_freedom = kwargs.pop('shu_df_freedom')
        self.shu_df_type = kwargs.pop('shu_df_type')
        self.shu_tail_sigma_mult = kwargs.pop('shu_tail_sigma_mult')
        self.shu_gaussian_at_input_res = kwargs.pop('shu_gaussian_at_input_res')

        super().__init__(*args, **kwargs)

        self.shu = SHU(
            self.shu_channels, self.shu_channels, 
            self.shu_df_freedom, self.shu_df_type, 
            input_res=self.shu_input_res,
            lowest_res=self.shu_lowest_res,
            tail_sigma_mult = self.shu_tail_sigma_mult,
            gaussian_at_input_res = self.shu_gaussian_at_input_res)

    def forward(self, img, c=None):
        x = None
        feats = {}
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, img, feat = block(x, img)
            feats[resi] = feat

        cmap = None
        if self.mapping is not None:
            cmap = self.mapping(None, c)
        x, feat = self.b4(x, img, cmap)
        feats[4] = feat

        infeat_shu = feats[self.shu_input_res][:, -self.shu_channels:]
        oufeat_shu = self.shu(infeat_shu)

        ch = self.shu_channels
        for ki, vi in oufeat_shu.items():
            fa, fb = torch.split(feats[ki], [feats[ki].size(1)-ch, ch], dim=1)
            fb = fb + vi
            feats[ki] = torch.cat([fa, fb], dim=1)
        return x, feats
