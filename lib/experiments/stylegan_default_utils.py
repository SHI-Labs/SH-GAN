import os
import os.path as osp
import numpy as np
import PIL.Image
import torch
import time

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def check_and_create_dir(path):
    while not osp.exists(path):
        if torch.distributed.get_rank() == 0:
            os.makedirs(path)
            break
        time.sleep(0.01)

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)
