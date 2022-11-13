import timeit
import numpy as np
import os
import os.path as osp
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from .cfg_holder import cfg_unique_holder as cfguh

def print_log(*console_info):
    console_info = [str(i) for i in console_info]
    console_info = ' '.join(console_info)
    print(console_info)
    try:
        log_file = cfguh().cfg.train.log_file
    except:
        try:
            log_file = cfguh().cfg.eval.log_file
        except:
            return
    # TODO: potential bug on both have train and eval
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(console_info + '\n')

class log_manager(object):
    """
    The helper to print logs. 
    """
    def __init__(self,
                 **kwargs):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def accumulate(self, 
                   n, 
                   data,
                   **kwargs):
        """
        Args:
            n: number of items (i.e. the batchsize)
            data: {itemname : float} data (i.e. the loss values)
                which are going to be accumulated. 
        """
        if n < 0:
            raise ValueError

        for itemn, di in data.items():
            try:
                self.data[itemn] += di * n
            except:
                self.data[itemn] = di * n
            
            try:
                self.cnt[itemn] += n
            except:
                self.cnt[itemn] = n

    def print(self, rank, itern, epochn, samplen, lr):
        console_info = [
            'Rank:{}'.format(rank),
            'Iter:{}'.format(itern),
            'Epoch:{}'.format(epochn),
            'Sample:{}'.format(samplen),
            'LR:{:.4E}'.format(lr)]

        cntgroups = {}
        for itemn, ci in self.cnt.items():
            try:
                cntgroups[ci].append(itemn)
            except:
                cntgroups[ci] = [itemn]

        for ci, itemng in cntgroups.items():
            console_info.append('cnt:{}'.format(ci)) 
            for itemn in sorted(itemng):
                console_info.append('{}:{:.4f}'.format(
                    itemn, self.data[itemn]/ci))

        console_info.append('Time:{:.2f}s'.format(
            timeit.default_timer() - self.time_check))
        return ' , '.join(console_info)

    def clear(self):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def pop(self, rank, itern, epochn, samplen, lr):
        console_info = self.print(
            rank, itern, epochn, samplen, lr)
        self.clear()
        return console_info

class distributed_log_manager(object):
    """
    The helper to print logs. 
    """
    def __init__(self,
                 **kwargs):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()        
        try:
            use_tensorboard = cfguh().cfg.train.log_tensorboard
        except:
            use_tensorboard = False

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        self.tb = None
        if use_tensorboard:
            import tensorboardX
            monitoring_dir = osp.join(cfguh().cfg.train.log_dir, 'tensorboard')
            if self.rank == 0:
                if osp.isdir(monitoring_dir):
                    shutil.rmtree(monitoring_dir)
                self.tb = tensorboardX.SummaryWriter(
                    osp.join(monitoring_dir))
        self.use_tensorboard = use_tensorboard

    def accumulate(self, 
                   n, 
                   **paras):
        """
        Args:
            n: number of items (i.e. the batchsize)
            data: {itemname : float} data (i.e. the loss values)
                which are going to be accumulated. 
        """
        if n < 0:
            raise ValueError

        for itemn, di in paras.items():
            try:
                self.data[itemn] += di * n
            except:
                self.data[itemn] = di * n
            
            try:
                self.cnt[itemn] += n
            except:
                self.cnt[itemn] = n

    def print(self, rank, itern, epochn, samplen, lr):
        console_info = [
            'Rank:{}'.format(rank),
            'Iter:{}'.format(itern),
            'Epoch:{}'.format(epochn),
            'Sample:{}'.format(samplen),
            'LR:{:.4E}'.format(lr)]

        cntgroups = {}
        for itemn, ci in self.cnt.items():
            try:
                cntgroups[ci].append(itemn)
            except:
                cntgroups[ci] = [itemn]

        for ci, itemng in cntgroups.items():
            console_info.append('cnt:{}'.format(ci)) 
            for itemn in sorted(itemng):
                console_info.append('{}:{:.4f}'.format(
                    itemn, self.data[itemn]/ci))

        console_info.append('Time:{:.2f}s'.format(
            timeit.default_timer() - self.time_check))
        return ' , '.join(console_info)

    def clear(self):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def pop(self, rank, itern, epochn, samplen, lr):
        console_info = self.print(
            rank, itern, epochn, samplen, lr)
        self.clear()
        return console_info

    def sync_(self, scalar, rank):
        if rank == self.rank:
            n = torch.ones([]).float()*scalar
            n = n.to(self.rank)
            dist.broadcast(n, src=rank)
            return float(scalar)

        n = torch.zeros([]).float()
        n = n.to(self.rank)
        dist.broadcast(n, src=rank)
        return float(n.item())

    def tensorboard_train(self, itern, lr, **paras):
        if not self.use_tensorboard:
            return

        global_step = itern*self.world_size
        if self.tb is not None:
            for i in range(self.world_size):
                self.tb.add_scalar('other/lr', lr, global_step+i)

        for itemn, di in paras.items():
            datai = [
                self.sync_(di, ranki)
                    for ranki in range(self.world_size)
            ]
            if self.tb is None:
                continue
            for i in range(self.world_size):
                if itemn.find('loss') != -1:
                    self.tb.add_scalar('loss/'+itemn,  datai[i], global_step+i)
                elif itemn.find('Loss') != -1:
                    self.tb.add_scalar('Loss/'+itemn,  datai[i], global_step+i)
                else:
                    self.tb.add_scalar('other/'+itemn, datai[i], global_step+i)

    def tensorboard_eval(self, itern, rv):
        if not self.use_tensorboard:
            return
        if self.tb is None:
            return 

        global_step = itern*self.world_size
        if isinstance(rv, dict):
            for name, value in rv.items():
                self.tb.add_scalar('eval/'+name, value, global_step)
        else:
            self.tb.add_scalar('eval', rv, global_step)
        return

    def tensorboard_close(self):
        if (self.use_tensorboard) and (self.tb is not None):
            self.tb.close()

# ----- also include some small utils -----

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

def gather_result(path, 
                  exid,
                  sdir,
                  root='/home/james/PythonNotes/log'):
    """
    Function that gather result from log dir, can gather
        multiple type of experiment, multiple ids and multiply
        sub evaluation dirs. 
    Args:
        path: str or [] of str,
            experiment folder or array of folders
            (i.e., [deeplab_cityscapes, ...])
        exid: int or [] of int,
            experiment ids
            will auto search all experiment folder for exid.
            # Error will only raise if the exid has no result retrived. 
        sdir: str or [] of str,
            evaluation subfolder or subfolders
            If subfolder is not found, skip it.
        root: str,
            the root to the log dir.
    Returns:
        info: pd.DataFrame,
            table on result, 
            columns are [
                str(path), str(experiment_path), 
                str(sdir), str(session_name), 
                str(category...)]
    """
    from easydict import EasyDict as edict
    import json
    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    from .cfg_helper import experiment_folder

    cfg = edict()
    cfg.MISC_DIR = root
    info = None

    exid_found = {ei:False for ei in exid}

    for pi in path:
        for ei in exid:
            cfg.EXPERIMENT_ID = ei
            try:
                path = experiment_folder(cfg, isnew=False, mdds_override=pi)
            except:
                continue

            expath = osp.basename(path)
            for si in sdir:
                resultf = osp.join(path, si, 'result.json')
                try:
                    with open(resultf, 'r') as f:
                        jinfo = json.load(f)['result']
                except:
                    continue
                exid_found[ei] = True

                for sname, di in jinfo.items():
                    for metric, data in di.items(): 
                        if isinstance(data, dict):
                            cats = list(data.keys())
                            values = list(data.values())
                        elif isinstance(data, float):
                            cats = [metric]
                            values = [data]

                        col = ['mdds', 'experiment', 'eval', 'session', 'metric'] \
                            + cats
                        val = [pi, expath, si, sname, metric] + values 

                        if info is None:
                            info = pd.DataFrame([val], columns=col)
                        else:
                            info = pd.concat([
                                    info, 
                                    pd.DataFrame([val], columns=col),
                                ], ignore_index=True, sort=False)

    for ei, found in exid_found.items():
        if not found:
            print(ei, 'not found!')

    return info

class plotter(object):
    def __init__(self, 
                 path, 
                 exid, 
                 critname, 
                 root='/home/james/PythonNotes/log',
                 palette = None,
                 subfigsize = [7, 14],):
        """
        Function that gather result from log dir, can gather
            multiple type of experiment, multiple ids and multiply
            sub evaluation dirs. 
        Args:
            path: str or [] of str,
                experiment folder or array of folders
                (i.e., [deeplab_cityscapes, ...])
            exid: int or [] of int,
                experiment ids
                will auto search all experiment folder for exid.
                # Error will only raise if the exid has no result retrived. 
            critname: str or [] of str,
                the criteria names to plot.
            root: str,
                the root to the log dir.
        Returns:
            info: pd.DataFrame,
                table on result, 
                columns are [
                    str(path), str(experiment_path), 
                    str(sdir), str(session_name), 
                    str(category...)]
        """
        import pandas as pd
        from easydict import EasyDict as edict
        from .cfg_helper import experiment_folder

        cfg = edict()
        cfg.MISC_DIR = root
        info_array = []
        for info in [path, exid, critname]:
            if isinstance(info, (list, tuple)):
                info_array.append(list(info))
            else:
                info_array.append([info])
        path, exid, critname = info_array

        if palette is None:
            self.palette = get_default_palette()

        critname += ['Iter']
        self.log_label = []
        self.log_data = []
        skipcrit = lambda x : x.find('Rank')!=0        

        for pi in path:
            for ei in exid:
                cfg.EXPERIMENT_ID = ei
                try:
                    logpath = experiment_folder(
                        cfg, isnew=False, mdds_override=pi)
                except:
                    continue
                logfile = osp.join(logpath, 'train.log')

                with open(logfile) as f:
                    content = f.readlines()
                content = [x.strip() for x in content] 

                data = []
                for l in content:
                    if skipcrit(l):
                        continue
                    data.append(self.line_to_data(l, critname))

                # the Rank and Iter should not be none.
                if (data[0][-1] is None) or (data[0][-2] is None):
                    raise ValueError

                # if some critname has no info, then it's value should be 
                # None from the beginning to the end.
                infoidx, noneidx = [], []
                for idx, di in enumerate(data[0]):
                    if di is None:
                        noneidx.append(idx)
                    else:
                        infoidx.append(idx)
                for idx in noneidx:
                    nonearray = [i[idx] is not None for i in data]
                    if sum(nonearray) != 0:
                        raise ValueError

                # exclude the Nones columns
                critn = [critname[i] for i in infoidx]
                data  = [[ii[i] for i in infoidx] for ii in data]

                # group the value by iter
                data_swap = {}
                for datai in data:
                    try:
                        data_swap[datai[-1]].append(datai[:-1])
                    except:
                        data_swap[datai[-1]] = [datai[:-1]]
                data = data_swap

                # aggregate and find the average
                for itern in data.keys():
                    data[itern] = [sum(i)/len(i) for i in zip(*data[itern])]

                # reformate the data so that it is {str<critname> : [<value>, ...]}
                ordered_itern = sorted(list(data.keys()))
                data_swap = {'Iter': ordered_itern}
                for idx, cni in enumerate(critn[:-1]):
                    # critn[:-1] exclude 'Itern'
                    data_swap[cni] = [data[i][idx] for i in ordered_itern]
                data = data_swap

                self.log_label.append('{}--{}'.format(pi, osp.basename(logpath)))
                self.log_data.append(data)

        # some para for plots
        self.critname = critname[:-1]
        self.show_col = int(np.ceil(np.sqrt(len(self.critname))))
        self.show_row = int(np.ceil(len(self.critname)/self.show_col))
        h, w = subfigsize
        self.figsize = [w*self.show_col, h*self.show_row]

    def line_to_data(self, line, critname):
        """
        Args:
            line: str,
                one log info line.
            critname: [] of str,
                list of criteria should be find.
        Return:
            data: [] of float,
                the values, order followed critname list order
                <critname> : <value> , <critname> : <value> 
        """
        line = line.split(',')
        data = {i:None for i in critname}
        for item in line:
            iseg = item.split(':')
            k = iseg[0].strip()
            try:
                data[k] = float(iseg[1].strip())
            except:
                continue
        data = [data[i] for i in critname]
        return data

    def __call__(self, 
                 start=0, 
                 window=1,
                 show=True,
                 close=True):
        """
        Args:
            start: int,
                the start itern to plot loss. 
            window: int,
                the average window size. 
                Be careful that the window size disregard the 
                    real itern step (which is usually 10)
        """
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(
            self.show_row, self.show_col, squeeze=False, figsize=self.figsize)
        axs = axs.flatten()
        for plot_idx, cni in enumerate(self.critname): 
            axs_ref = axs[plot_idx]
            for data_idx, mdds in enumerate(self.log_label):
                log_ref = self.log_data[data_idx]

                axs_ref.set_title(cni)
                if cni in log_ref.keys():
                    x = log_ref['Iter'][start:]
                    y = log_ref[cni][start:]
                    y = np.convolve(y, np.ones((window,))/window, mode='valid')
                    x = x[-len(y):]
                    axs_ref.plot(
                        x, y,
                        color=np.array(self.palette[data_idx])/255,
                        label=mdds)
                axs_ref.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
        if show:
            plt.show()
        if close:
            plt.close()
            return None
        else:
            return axs

############
# palettes #
############

def get_default_palette():
    return {
        -1: [255, 255, 255], 
        0:  [0  , 0  , 0  ],
        1:  [230, 25 , 75 ], # red
        2:  [255, 225, 25 ], # yellow
        3:  [67 , 99 , 216], # blue
        4:  [60 , 180, 75 ], # green
        5:  [245, 130, 49 ], # orange
        6:  [191, 239, 69 ], # lime            
        7:  [66 , 212, 244], # cyan
        8:  [145, 30 , 180], # purple
        9:  [240, 50 , 230], # magenta
        10: [169, 169, 169], # gray
        11: [0  , 0  , 117], # navy
        12: [255, 250, 200], # beige
        13: [128, 0  , 0  ], # maroon
        14: [230, 190, 255], # lavender
        15: [170, 255, 195], # mint
        16: [154,  99,  36], # brown
        17: [255, 216, 177], # apricot
        18: [70 , 153, 144], # teal
        19: [250, 190, 190], # pink
        20: [128, 128,   0], # oliver
    }
