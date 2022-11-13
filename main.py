import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import copy

from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import \
    get_experiment_id, \
    cfg_to_debug, \
    get_command_line_args, \
    cfg_initiates

from lib.utils import train as train_base
from lib.utils import eval as eval_base
from lib.model_zoo.shgan import version
from lib.experiments import get_experiment

class train(train_base):
    def prepare_model(self):
        # Do not load model here
        return {}

    def prepare_dataloader(self):
        # Do not load dataset here
        return {}

    def save(self, *args, **kwargs):
        pass

class eval(eval_base):
    def prepare_model(self):
        # Do not load model here
        return {}

    def prepare_dataloader(self):
        # Do not load dataset here
        return {}

if __name__ == "__main__":
    cfg = get_command_line_args()
    isresume = 'resume_path' in cfg.env

    if ('train' in cfg) and not isresume:
        cfg.train.experiment_id = get_experiment_id()

    isdebug = getattr(cfg.env, 'debug', False)

    if isdebug:
        # pass
        if not isresume:
            cfg = cfg_to_debug(cfg)
        else:
            cfg.env.gpu_device = [0]
            cfg.env.gpu_count = 1

        if 'train' in cfg:
            cfg.train.image_snapshot_ticks = 1
        if 'eval' in cfg:
            cfg.eval.dataset.try_sample = 32
            cfg.eval.batch_size_per_gpu = 32

    cfg = cfg_initiates(cfg)

    if 'train' in cfg: 
        trainer = train(cfg)
        tstage = get_experiment(cfg.train.exec_stage)()
        trainer.register_stage(tstage)
        if cfg.env.gpu_count == 1:
            trainer(0)
        else:
            mp.spawn(trainer,
                     args=(),
                     nprocs=cfg.env.gpu_count, 
                     join=True)
    else:
        evaler = eval(cfg)
        estage = get_experiment(cfg.eval.exec_stage)()
        evaler.register_stage(estage)
        if cfg.env.gpu_count == 1:
            evaler(0)
        else:
            mp.spawn(evaler,
                     args=(),
                     nprocs=cfg.env.gpu_count,
                     join=True)
