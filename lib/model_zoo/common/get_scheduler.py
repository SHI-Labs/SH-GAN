import torch
import torch.optim as optim
import numpy as np

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_scheduler(object):
    def __init__(self):
        self.lr_scheduler = {}

    def register(self, lrsf, name):
        self.lr_scheduler[name] = lrsf

    def __call__(self, pipeline_cfg):
        schedulers = []        
        for ci in pipeline_cfg:
            t = ci.type
            schedulers.append(
                self.lr_scheduler[t](**ci.args))
        if len(schedulers) == 0:
            raise ValueError
        else:
            return compose(schedulers)

def register(name):
    def wrapper(class_):
        get_scheduler().register(class_, name)
        return class_
    return wrapper

class template_scheduler(object):
    def __init__(self, step):
        self.step = step

    def __getitem__(self, idx):
        raise ValueError

    def to_list(self):
        return [
            self.__getitem__(i) 
                for i in range(self.step)
        ]

@register('constant')
class constant_scheduler(template_scheduler):
    def __init__(self, lr, step):
        super().__init__(step)
        self.lr = lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr

@register('poly')
class poly_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, power, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b = self.start_lr, self.end_lr
        p, n = self.power, self.step
        return b + (a-b)*((1-idx/n)**p)

@register('linear')
class linear_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b, n = self.start_lr, self.end_lr, self.step
        return b + (a-b)*(1-idx/n)

@register('multistage')
class constant_scheduler(template_scheduler):
    def __init__(self, start_lr, milestones, gamma, step):
        super().__init__(step)
        self.start_lr = start_lr
        m = [0] + milestones + [step]
        lr_iter = start_lr
        self.lr = []
        for ms, me in zip(m[0:-1], m[1:]):
            for _ in range(ms, me):
                self.lr.append(lr_iter)
            lr_iter *= gamma

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr[idx]

class compose(object):
    def __init__(self, schedulers):
        self.lr = []
        for i in schedulers:
            self.lr += i.to_list()
        self.step = len(self.lr)

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr[idx]

    def to_list(self):
        return self.lr

    def set_lr(self, optim, new_lr, pg_lrscale=None):
        """
        Set Each parameter_groups in optim with new_lr
        New_lr can be find according to the idx.
        pg_lrscale tells how to scale each pg.
        """
        # new_lr = self.__getitem__(idx)
        if pg_lrscale is not None:
            scale_list = [
                pg_lrscale[k] for k in sorted(pg_lrscale)
            ]
        else:
            scale_list = None

        for i, pg in enumerate(optim.param_groups):
            if scale_list is None:
                pg['lr'] = new_lr
            else:
                pg['lr'] = new_lr * scale_list[i]
