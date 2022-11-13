import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# cudnn.enabled = True
# cudnn.benchmark = True
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import pprint
import timeit
import time
import copy
import matplotlib.pyplot as plt

from .cfg_holder import cfg_unique_holder as cfguh

from .data_factory import \
    get_dataset, collate, \
    get_loader, \
    get_transform, \
    get_estimator, \
    get_formatter, \
    DistributedSampler

from .model_zoo import \
    get_model, save_state_dict, \
    get_optimizer, \
    get_scheduler

from .log_service import print_log, distributed_log_manager

from .evaluator import get_evaluator

class train_stage(object):
    """
    This is a template for a train stage,
        (can be either train or test or anything)
    Usually, it takes RANK
        one dataloader, one model, one optimizer, one lr_scheduler.
    But it is not limited to these parameters. 
    """
    def __init__(self):
        self.nested_eval_stage = None
        self.rv_keep = None

    def is_better(self, x):
        return (self.rv_keep is None) or (x>self.rv_keep)

    def __call__(self,
                 **paras):
        cfg = cfguh().cfg
        cfgt = cfg.train
        logm = distributed_log_manager()
        epochn, itern = 0, 0

        trainloader  = paras['trainloader']
        lr_scheduler = paras['lr_scheduler']
        net          = paras['net']
        RANK         = paras['RANK']

        net.train()

        epoch_time = timeit.default_timer()

        while True:
            for idx, batch in enumerate(trainloader):
                # so first element of batch (usually image) can be [tensor]
                if not isinstance(batch[0], list):
                    bs = batch[0].shape[0]
                else:
                    bs = len(batch[0])
                if cfgt.skip_partial_batch:
                    if bs != cfgt.batch_size_per_gpu:
                        continue

                # debug
                # bs = batch['cell'].shape[0]

                if cfgt.step_type == 'epoch':
                    lr = lr_scheduler[epochn]
                elif cfgt.step_type == 'iter':
                    lr = lr_scheduler[itern]
                else:
                    raise ValueError

                # this is a special init that helps the model
                # to sync. Which is a loss.backward without weight update.
                # actually, the bug is not very frequently shows.
                # if itern==0:
                #     self.main(
                #         batch=batch,
                #         lr=lr, 
                #         isinit=True,
                #         itern=itern,
                #         epochn=epochn,
                #         **paras)
                paras_new = self.main(
                    batch=batch, 
                    lr=lr,
                    isinit=False,
                    itern=itern,
                    epochn=epochn,
                    **paras)

                paras.update(paras_new)

                logm.accumulate(bs, **paras['rvinfo'])
                logm.tensorboard_train(itern, lr, **paras['rvinfo'])

                itern += 1
                if itern % cfgt.log_display == 0:
                    print_log(logm.pop(
                        RANK, 
                        itern, 
                        epochn, 
                        (idx+1)*cfgt.batch_size, 
                        lr
                    ))

                if cfgt.step_type == 'iter':
                    eval_now = 'eval' in cfg
                    eval_now &= self.nested_eval_stage is not None
                    eval_now &= cfgt.eval_every is not None
                    if eval_now:
                        eval_itern = max(itern-cfgt.eval_start, 1)
                        eval_now &= (eval_itern%cfgt.eval_every == 0)
                    if eval_now:
                        net.eval()
                        rv = self.nested_eval_stage(
                            eval_cnt=itern,
                            **paras)['eval_rv']
                        if rv is not None:
                            logm.tensorboard_eval(itern, rv)
                        net.train()
                        if self.is_better(rv):
                            self.rv_keep = rv
                            if (RANK==0):
                                self.save(net)

                    if itern >= cfgt.step_num:
                        break                    
                    if cfgt.ckpt_every is None:
                        continue
                    if (itern%cfgt.ckpt_every == 0) and (RANK==0):
                        print_log('Checkpoint... {}'.format(itern))
                        self.save(net, itern=itern)

                # loop end

            epochn += 1
            print_log('Epoch {} time:{:.2f}s.'.format(
                epochn, timeit.default_timer()-epoch_time))
            epoch_time = timeit.default_timer()

            if cfgt.step_type == 'iter':
                if itern >= cfgt.step_num:
                    break

            elif cfgt.step_type == 'epoch':
                eval_now = 'eval' in cfg
                eval_now &= self.nested_eval_stage is not None
                eval_now &= cfgt.eval_every is not None
                if eval_now:
                    eval_epochn = max(epochn-cfgt.eval_start, 1)
                    eval_now &= (eval_epochn%cfgt.eval_every == 0)
                if eval_now:
                    net.eval()
                    rv = self.nested_eval_stage(
                        eval_cnt=epochn,
                        **paras)['eval_rv']
                    net.train()
                    if self.is_better(rv):
                        self.rv_keep = rv
                        if (RANK==0):
                            self.save(net)

                if epochn >= cfgt.step_num:
                    break
                if cfgt.ckpt_every is None:
                    continue
                if (epochn%cfgt.ckpt_every == 0) and (RANK==0):
                    print_log('Checkpoint... {}'.format(epochn))
                    self.save(net, epochn=epochn)

        logm.tensorboard_close()
        return {}

    def main(self, isinit=False, **paras):
        raise NotImplementedError

    def save(self, net, itern=None, epochn=None):
        cfgt = cfguh().cfg.train
        try:
            net_symbol = net.module.symbol
        except:
            net_symbol = net.symbol
        if itern is not None:
            path = '{}_{}_iter_{}.pth'.format(
                cfgt.experiment_id, 
                net_symbol,
                itern)
            path = osp.join(cfgt.log_dir, path)
            save_state_dict(net, path)
        elif epochn is not None:
            path = '{}_{}_epoch_{}.pth'.format(
                cfgt.experiment_id, 
                net_symbol,
                epochn)
            path = osp.join(cfgt.log_dir, path)
            save_state_dict(net, path)
        else:
            path = '{}_{}_best.pth'.format(
                cfgt.experiment_id, 
                net_symbol)
            path = osp.join(cfgt.log_dir, path)
            save_state_dict(net, path)

class eval_stage(object):
    def __init__(self):
        self.evaluator = None

    def check_and_create_dir(self, path):
        if not osp.exists(path):
            if torch.distributed.get_rank() == 0:
                os.makedirs(path)
        torch.distributed.barrier()

    def __call__(self, 
                 RANK,
                 evalloader,
                 net,
                 **paras):
        cfgt = cfguh().cfg.eval
        if self.evaluator is None:
            evaluator = get_evaluator()(cfgt.evaluator)
            self.evaluator = evaluator
        else:
            evaluator = self.evaluator

        time_check = timeit.default_timer()

        for idx, batch in enumerate(evalloader): 
            rv = self.main(RANK, batch, net)
            evaluator.add_batch(**rv)
            if cfgt.output_result:
                try:
                    self.output_f(**rv, cnt=paras['eval_cnt'])
                except:
                    self.output_f(**rv)
            if idx%cfgt.log_display == cfgt.log_display-1:
                print_log('processed.. {}, Time:{:.2f}s'.format(
                    idx+1, timeit.default_timer() - time_check))
                time_check = timeit.default_timer()
            # break

        evaluator.set_sample_n(len(evalloader.dataset))
        eval_rv = evaluator.compute()
        if RANK == 0:
            evaluator.one_line_summary()
            evaluator.save(cfgt.log_dir)
        evaluator.clear_data()
        return {
            'eval_rv' : eval_rv
        }

class exec_container(object):
    """
    This is the base functor for all types of executions.
        One execution can have multiple stages, 
        but are only allowed to use the same 
        config, network, dataloader. 
    Thus, in most of the cases, one exec_container is one
        training/evaluation/demo...
    If DPP is in use, this functor should be spawn.
    """
    def __init__(self,
                 cfg,
                 **kwargs):
        self.cfg = cfg
        self.registered_stages = []
        self.RANK = None

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, 
                 RANK,
                 **kwargs):
        """
        Args:
            RANK: int,
                the rank of the stage process.
            If not multi process, please set 0.
        """
        self.RANK = RANK
        cfg = self.cfg
        cfguh().save_cfg(cfg) 
        # broadcast cfg
        dist.init_process_group(
            backend = cfg.env.dist_backend,
            init_method = cfg.env.dist_url,
            rank = RANK,
            world_size = cfg.env.gpu_count,
        )

        # need to set random seed 
        # originally it is in common_init()
        # but it looks like that the seed doesn't transfered to here.
        if isinstance(cfg.env.rnd_seed, int):
            np.random.seed(cfg.env.rnd_seed)
            torch.manual_seed(cfg.env.rnd_seed)

        time_start = timeit.default_timer()

        para = {
            'RANK':RANK,
            'itern_total':0}
        dl_para = self.prepare_dataloader()
        if not isinstance(dl_para, dict):
            raise ValueError
        para.update(dl_para)
        md_para = self.prepare_model()
        if not isinstance(md_para, dict):
            raise ValueError
        para.update(md_para)

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        # save the model
        if RANK == 0:
            pass # Temporary disabled
            # only train will save the model
            # if 'train' in cfg:
            #     self.save(**para)

        print_log(
            'Total {:.2f} seconds'.format(timeit.default_timer() - time_start))
        self.RANK = None
        dist.destroy_process_group()

    def prepare_dataloader(self):
        """
        Prepare the dataloader from config.
        """
        return {
            'trainloader' : None,
            'evalloader' : None}

    def prepare_model(self):
        """
        Prepare the model from config.
        A default prepare_model for training.
            If the desire behaviour is eval. Or the there
            are two models, or there are something special.
            Please override this function.
        """
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        istrain = 'train' in cfg

        # save the init model
        if istrain:
            if (cfg.train.save_init_model) and (self.RANK==0):
                output_model_file = osp.join(
                    cfg.train.log_dir, 
                    '{}_{}.pth.init'.format(
                        cfg.train.experiment_id, 
                        cfg.model.symbol))
                save_state_dict(net, output_model_file)

        paras = {}

        if cfg.env.cuda:
            net.to(self.RANK)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.RANK], 
                find_unused_parameters=True)

        if istrain:
            net.train() 

            lr_scheduler = get_scheduler()(
                cfg.train.lr_schduler)

            optimizer = get_optimizer()(
                net, cfg.train.optimizer)
            return {
                'net'          : net,
                'optimizer'    : optimizer,
                'lr_scheduler' : lr_scheduler,
            }
        else:
            net.eval()
            return {
                'net' : net,
            }

    def save(self, net, **kwargs):
        cfgt = cfguh().cfg.train
        try:
            net_symbol = net.module.symbol
        except:
            net_symbol = net.symbol

        output_model_file = osp.join(
            cfgt.log_dir,
            '{}_{}_last.pth'.format(
                cfgt.experiment_id, 
                net_symbol,
            )
        )
        print_log('Saving model file {0}'.format(output_model_file))
        save_state_dict(net, output_model_file)

class train(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        trainset = get_dataset()(cfg.train.dataset)
        sampler = DistributedSampler(
            dataset=trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size = cfg.train.batch_size_per_gpu, 
            sampler = sampler, 
            num_workers = cfg.train.dataset_num_workers_per_gpu, 
            drop_last = False, 
            pin_memory = False,
            collate_fn = collate(), 
        )

        if 'eval' in cfg:
            evalset = get_dataset()(cfg.eval.dataset)
            sampler = DistributedSampler(
                evalset, shuffle=False, extend=True)
            evalloader = torch.utils.data.DataLoader(
                evalset, 
                batch_size = cfg.eval.batch_size_per_gpu,
                sampler = sampler,
                num_workers = cfg.train.dataset_num_workers_per_gpu,
                drop_last = False, 
                pin_memory = False,
                collate_fn = collate(), 
            )
        else:
            evalloader = None

        return {
            'trainloader' : trainloader,
            'evalloader'  : evalloader,}

class eval(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        evalset = get_dataset()(cfg.eval.dataset)
        sampler = DistributedSampler(
            evalset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(
            evalset, 
            batch_size = cfg.eval.batch_size_per_gpu,
            sampler = sampler,
            num_workers = cfg.eval.dataset_num_workers_per_gpu,
            drop_last = False, 
            pin_memory = False,
            collate_fn = collate(), 
        )
        return {
            'trainloader' : None,
            'evalloader'  : evalloader,}

class finalize_loss(object):
    """
    A function that finalize all loss into one loss.
    Caller can choose different type of average techniques.
    This is simplified from previous version. 
    """
    def __init__(self, 
                 weight = None,
                 normalize = True,
                 **kwargs):
        """
        Args:
            weight: {str : float} or None,
                {<lossname> : <weigbt>}
                A dict tells the unnormalized weights 
                we would like to put on each loss items.
                If None, then average all items.
            normalize: bool,
                whether normalize the weight. 
        """
        self.normalize = normalize

        if weight is None:
            self.weight = None
            return

        for _, wi in weight.items():
            if wi < 0:
                raise ValueError

        if not normalize:
            self.weight = weight
        else:
            sum_weight = 0
            for _, wi in weight.items():
                sum_weight += wi
            if sum_weight == 0:
                raise ValueError           
            self.weight = {
                n : wi/sum_weight \
                    for n, wi in weight.items()}

    def __call__(self, 
                 loss_input,):
        """    
        Args:
            loss_input: {str : torch float}, 
                {
                    <lossname> : <loss>, 
                    <otherinfo> : <value>,
                }
                loss can be backpropagated.
                lossname begin with loss... will be weighted sum and 
                    backpropagated. 
                otherinfo will be ignored from backprop
        Returns:
            loss: torch float,
                The finalized loss need to be back propagated. 
            display: {str : float}, 
                {
                    Loss : <finalloss>,
                    <lossname> : <loss>,
                    <otherinfo> : <value>,
                }
        """
        display = {
            n : v.item() \
                for n, v in loss_input.items()
        }
        loss_dict = {
            n : v 
                for n, v in loss_input.items()
                    if n.find('loss')==0
        }

        if self.weight is None:
            loss = 0
            for n, v in loss_dict.items():
                loss += v
            if self.normalize:
                loss /= len(loss_dict)
            display['Loss'] = loss.item()
            return loss, display

        if sorted(loss_dict.keys()) \
                != sorted(self.weight.keys()):
            raise ValueError

        loss = 0
        for n, v in loss_dict.items():
            loss += v * self.weight[n]
        display['Loss'] = loss.item()
        return loss, display

###############
# some helper #
###############

def torch_to_numpy(*argv):
    if len(argv) > 1:
        data = list(argv)
    else:
        data = argv[0]

    if isinstance(data, torch.Tensor):
        return data.to('cpu').detach().numpy()

    elif isinstance(data, (list, tuple)):
        out = []
        for di in data:
            out.append(torch_to_numpy(di))
        return out

    elif isinstance(data, dict):
        out = {}
        for ni, di in data.items():
            out[ni] = torch_to_numpy(di)
        return out
    
    else:
        return data
