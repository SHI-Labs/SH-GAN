import torch
import torch.optim as optim
import numpy as np
import itertools

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class get_optimizer(object):
    def __init__(self):
        self.optimizer = {}
        self.register(optim.SGD, 'sgd')
        self.register(optim.Adam, 'adam')

    def register(self, optim, name):
        self.optimizer[name] = optim

    def __call__(self, net, cfg):
        t = cfg.type
        params = []
        try:
            pg = net.module.parameter_group
        except:
            pg = net.parameter_group

        for k in sorted(pg.keys()):
            pgi = [i.parameters() for i in pg[k]]
            pgi = itertools.chain(*pgi)
            pgi = {'params' : pgi}
            params.append(pgi)
        return self.optimizer[t](
            params,
            lr=0,
            **cfg.args)
