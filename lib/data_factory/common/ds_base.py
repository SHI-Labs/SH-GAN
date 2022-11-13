import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision
import copy
import itertools

# import sys
# code_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
# sys.path.append(code_dir)

from ...cfg_holder import cfg_unique_holder as cfguh
from ...log_service import print_log

# 0730 try shared memory
import torch.distributed as dist
from multiprocessing import shared_memory
import pickle
import hashlib

class ds_base(torch.utils.data.Dataset):
    def __init__(self, 
                 cfg, 
                 loader = None, 
                 estimator = None, 
                 transforms = None, 
                 formatter = None):

        self.cfg = cfg
        self.init_load_info(cfg)
        self.loader = loader
        self.transforms = transforms
        self.formatter = formatter

        console_info = '{}: '.format(self.__class__.__name__)
        console_info += 'total {} unique images, '.format(len(self.load_info))

        # self.load_info = sorted(self.load_info, key=lambda x:x['unique_id'])
        # added 20210303 for memory trick
        try:
            load_info_order_by = self.cfg.load_info_order_by
        except:
            load_info_order_by = None

        if load_info_order_by == 'imsize_current_reverse':
            self.load_info = sorted(
                self.load_info, key=lambda x:np.prod(x['imsize_current']), reverse=True)
        elif load_info_order_by is None:
            self.load_info = sorted(self.load_info, key=lambda x:x['unique_id'])
        else:
            raise ValueError

        try:
            self.try_sample = self.cfg.try_sample
        except:
            self.try_sample = None
        if self.try_sample is not None:
            self.load_info = self.load_info[:self.try_sample]

        if estimator is not None:
            self.load_info = estimator(self.load_info)

        try:
            pick = self.cfg.pick
            if pick is not None:
                self.load_info = [i for i in self.load_info if i['filename'] in pick]
        except:
            pass

        for idx, info in enumerate(self.load_info):
            info['idx'] = idx
        try:
            self.repeat = self.cfg.repeat
        except:
            self.repeat = 1

        try:
            cache_pct = self.cfg.cache_pct
        except:
            cache_pct = 0

        # 0730 try shared memory
        try:
            self.cache_sm = self.cfg.cache_sm
        except:
            self.cache_sm = False

        if dist.is_initialized():
            self.rank = dist.get_rank() # used in __del__
        else:
            print('Warning, torch ddp not initialized so dataset.rank set to default value 0.')
            self.rank = 0

        if self.cache_sm:
            if dist.get_rank() == 0:
                import random
                cache_unique_id = pickle.dumps([random.random() for _ in range(10)])
                self.cache_unique_id = hashlib.sha256(cache_unique_id).hexdigest()
                shm = shared_memory.SharedMemory(
                    name='cache_unique_id', create=True, size=len(cache_unique_id))
                shm.buf[0:len(cache_unique_id)] = cache_unique_id[0:len(cache_unique_id)]
                dist.barrier()
            else:
                dist.barrier()
                shm = shared_memory.SharedMemory(name='cache_unique_id')
                self.cache_unique_id = hashlib.sha256(shm.buf).hexdigest()

            dist.barrier()
            if dist.get_rank() == 0:
                shm.close()
                shm.unlink()

        self.cache_cnt = None # set later
        self.__cache__(cache_pct)

        console_info += 'total {} unique sample. Cached {}. Repeat {} times.'.format(
            len(self.load_info), self.cache_cnt, self.repeat)
        print_log(console_info)

    def init_load_info(self, cfg):
        # implement by sub class
        raise ValueError

    def __len__(self):
        return len(self.load_info)*self.repeat

    def __cache__(self, pct):
        if pct == 0:
            self.cache_cnt = 0
            return
        self.cache_cnt = int(len(self.load_info)*pct)
        # for i in range(self.cache_cnt):
        #     self.load_info[i] = self.loader(self.load_info[i])

        # 0730 try shared memory
        if not self.cache_sm:
            for i in range(self.cache_cnt):
                self.load_info[i] = self.loader(self.load_info[i])
            return

        for i in range(self.cache_cnt):
            shm_name = str(self.load_info[i]['unique_id']) + '_' + self.cache_unique_id
            if i % dist.get_world_size() == dist.get_rank():
                data = pickle.dumps(self.loader(self.load_info[i]))
                datan = len(data)
                # self.print_smname_to_file(shm_name)
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=datan)
                shm.buf[0:datan] = data[0:datan]
                shm.close()
                self.load_info[i] = shm_name
            else:
                self.load_info[i] = shm_name
        dist.barrier()

    def __getitem__(self, idx):
        idx = idx%len(self.load_info)
        # element = copy.deepcopy(self.load_info[idx])

        # 0730 try shared memory
        element = self.load_info[idx]
        if isinstance(element, str):
            shm = shared_memory.SharedMemory(name=element)
            element = pickle.loads(shm.buf)
            shm.close()
        else:
            element = copy.deepcopy(element)

        if idx >= self.cache_cnt:
            element = self.loader(element)
        if self.transforms is not None:
            element = self.transforms(element)
        if self.formatter is not None:
            return self.formatter(element)
        else:
            return element

    # 0730 try shared memory
    def __del__(self):
        # Clean the shared memory
        if self.rank == 0:
            for infoi in self.load_info:
                if isinstance(infoi, str):
                    shm = shared_memory.SharedMemory(name=infoi)
                    shm.close()
                    shm.unlink()
                # dist.get_rank() # Cannot use because dist is already __del__ at this point

    def print_smname_to_file(self, smname):
        try:
            log_file = cfguh().cfg.train.log_file
        except:
            try:
                log_file = cfguh().cfg.eval.log_file
            except:
                raise ValueError
        # a trick to use the log_file path
        sm_file = log_file.replace('.log', '.smname')
        with open(sm_file, 'a') as f:
            f.write(smname + '\n')

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

from .ds_loader import get_loader
from .ds_transform import get_transform
from .ds_estimator import get_estimator
from .ds_formatter import get_formatter

@singleton
class get_dataset(object):
    def __init__(self):
        self.dataset = {}

    def register(self, ds):
        self.dataset[ds.__name__] = ds

    def __call__(self, cfg):
        t = cfg.type

        # the register is in each file
        if t == 'cityscapes':
            from .. import ds_cityscapes
        elif t == 'div2k':
            from .. import ds_div2k
        elif t == 'gandiv2k':
            from .. import ds_gandiv2k
        elif t == 'srbenchmark':
            from .. import ds_srbenchmark
        elif t == 'imagedir':
            from .. import ds_imagedir
        elif t in ['places2', 'places2_loadgen']:
            from .. import ds_places2
        elif t == 'texture':
            from .. import ds_texture
        elif t in ['ffhq', 'ffhqsimple', 'ffhqzip', 'ffhqzip_loadgen']:
            from .. import ds_ffhq
        elif t in ['imcpt2020', 'imcpt2020_auto_distance', 'imcpt2020inpainting']:
            from .. import ds_imcpt
        elif t in ['openimages']:
            from .. import ds_openimages
        else:
            raise ValueError

        loader    = get_loader()(cfg.loader)
        transform = get_transform()(cfg.transform)
        estimator = get_estimator()(cfg.estimator)
        formatter = get_formatter()(cfg.formatter)

        return self.dataset[t](
            cfg, loader, estimator, 
            transform, formatter)

def register():
    def wrapper(class_):
        get_dataset().register(class_)
        return class_
    return wrapper

# some other helpers

class collate(object):
    """
        Modified from torch.utils.data._utils.collate
        It handle list different from the default.
            List collate just by append each other.
    """
    def __init__(self):
        self.default_collate = \
            torch.utils.data._utils.collate.default_collate

    def __call__(self, batch):
        """
        Args:
            batch: [data, data] -or- [(data1, data2, ...), (data1, data2, ...)]
        This function will not be used as induction function
        """
        elem = batch[0]
        if not (elem, (tuple, list)):
            return self.default_collate(batch)
        
        rv = []
        # transposed
        for i in zip(*batch):
            if isinstance(i[0], list):
                if len(i[0]) != 1:
                    raise ValueError
                try:
                    i = [[self.default_collate(ii).squeeze(0)] for ii in i]
                except:
                    pass
                rvi = list(itertools.chain.from_iterable(i))
                rv.append(rvi) # list concat
            else:
                rv.append(self.default_collate(i))
        return rv
