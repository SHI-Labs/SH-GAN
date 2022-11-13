import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch

from ..evaluator.stylegan_metrics import metric_main

from lib.cfg_holder import cfg_unique_holder as cfguh

import os.path as osp
from lib.data_factory import \
    get_dataset, collate, DistributedSampler
from lib.model_zoo import get_model
from lib.log_service import print_log

from .stylegan_default_utils import \
    check_and_create_dir, highlight_print

from .stylegan_default import draw_functor as draw_functor_base
from .stylegan_default import eval_stage as eval_stage_base

class draw_functor(draw_functor_base):
    def __call__(self, **kwargs):
        RANK = self.RANK
        generator = kwargs['generator']
        filename = kwargs['filename'] if 'filename' in kwargs else 'demo.png'
        isinit = kwargs['isinit'] if 'isinit' in kwargs else False

        if 'input' in kwargs:
            images = [generator(**input_i).cpu() for input_i in kwargs['input']]
            # recover grid_real and grid_mask from overrided input
            grid_real = torch.cat([input_i['x'][:, 1:] for input_i in kwargs['input']])
            grid_mask = torch.cat([input_i['x'][:, 0:1] for input_i in kwargs['input']])+0.5
            grid_real, grid_mask = grid_real.detach().to('cpu'), grid_mask.detach().to('cpu')
        elif self.input is not None:
            images = [generator(**input_i).cpu() for input_i in self.input]
            # recover grid_real and grid_mask from saved input
            grid_real = torch.cat([input_i['x'][:, 1:] for input_i in self.input])
            grid_mask = torch.cat([input_i['x'][:, 0:1] for input_i in self.input])+0.5
            grid_real, grid_mask = grid_real.detach().to('cpu'), grid_mask.detach().to('cpu')
        else:
            evalset = kwargs['evalset']
            z_dim = kwargs['z_dim']
            grid_size_n = np.prod(self.grid_size)
            grid_z = torch.randn([grid_size_n, z_dim])
            grid_c = torch.randn([grid_size_n, 0])
            grid_real, grid_mask, _ = zip(*[evalset[i] for i in range(grid_size_n)])
            # To avoid a wierd bug in ZipFile VVVV
            if getattr(evalset.loader[0], 'zipfile_close', None) is not None:
                evalset.loader[0].zipfile_close()
            grid_real = torch.stack([torch.FloatTensor(i) for i in grid_real])
            grid_mask = torch.stack([torch.FloatTensor(i) for i in grid_mask]).unsqueeze(1)
            grid_real_erased = grid_real * grid_mask
            grid_x = torch.cat([grid_mask-0.5, grid_real_erased], axis=1)
            if RANK >= 0:
                grid_z, grid_c, grid_x = grid_z.to(RANK), grid_c.to(RANK), grid_x.to(RANK)
            grid_z = grid_z.split(self.batch_gpu)
            grid_c = grid_c.split(self.batch_gpu)
            grid_x = grid_x.split(self.batch_gpu)

            # XX_debug
            # ipath = "/home/james/Project/co-mod-gan/debug/Places365_val_00009088.png"
            # mpath = "/home/james/Project/co-mod-gan/debug/Places365_val_00009088_mask.png"
            # opath = "/home/james/Project/co-mod-gan/debug/Places365_val_00009088_output2.png"

            # grid_real = PIL.Image.open(ipath)
            # grid_mask = PIL.Image.open(mpath)
            # grid_real = grid_real.resize([512, 512], PIL.Image.BICUBIC)
            # grid_mask = grid_mask.resize([512, 512], PIL.Image.NEAREST)
            # grid_real = np.asarray(grid_real.convert('RGB')).transpose([2, 0, 1])
            # grid_real = (grid_real.astype(np.float32)/255 - 0.5)*2
            # grid_real = torch.FloatTensor(grid_real).unsqueeze(0).to(RANK)
            # grid_mask = np.asarray(grid_mask.convert('1'), dtype=np.float32)[np.newaxis]
            # grid_mask = torch.FloatTensor(grid_mask).unsqueeze(0).to(RANK)

            # grid_real_erased = grid_real * grid_mask
            # grid_x = torch.cat([grid_mask-0.5, grid_real_erased], axis=1)
            # grid_z = torch.zeros([1, z_dim]).to(RANK)
            # grid_c = torch.randn([1, 0]).to(RANK)

            # grid_o = generator(x=grid_x, z=grid_z, c=grid_c, noise_mode='none')
            # grid_o = grid_real * grid_mask + grid_o * (1-grid_mask)
            # grid_o = ((grid_o.detach().cpu().numpy()[0]+1)/2 * 255).transpose(1, 2, 0)
            # grid_o = np.rint(grid_o).clip(0, 255).astype(np.uint8)
            # PIL.Image.fromarray(grid_o).save(opath)
            # raise ValueError

            self.input = [{
                    'x': x,
                    'z': z, 
                    'c': c, 
                    'noise_mode': 'const',
                } for x, z, c in zip(grid_x, grid_z, grid_c)]
            images = [generator(**input_i).cpu() for input_i in self.input]

        images = torch.cat(images)
        self.output(images, os.path.join(
            self.log_dir, self.subfolder, filename),
            drange=[-1, 1], grid_size=self.grid_size)
        filename_combined = osp.splitext(filename)
        filename_combined = filename_combined[0]+'_combined'+filename_combined[1]
        images_combined = grid_real * grid_mask + images * (1-grid_mask)
        self.output(images_combined, os.path.join(
            self.log_dir, self.subfolder, filename_combined),
            drange=[-1, 1], grid_size=self.grid_size)

        if isinit:
            self.output(grid_mask, os.path.join(
                self.log_dir, self.subfolder, 'masks.png'),
                drange=[0, 1], grid_size=self.grid_size)
            self.output(grid_real, os.path.join(
                self.log_dir, self.subfolder, 'reals.png'),
                drange=[-1, 1], grid_size=self.grid_size)
            self.output(grid_real_erased, os.path.join(
                self.log_dir, self.subfolder, 'erased.png'),
                drange=[-1, 1], grid_size=self.grid_size)

##############
# eval stage #
##############

class eval_stage(eval_stage_base):
    """
    The version that evaluate using my evaluator only. 
    Which include FID, LPIPS, PSNR and SSIM
    """
    def __init__(self):
        self.draw_function = draw_functor

    def resume(self, resume_pkl, models):
        assert False, "This functionality is broken"

    def get_ddp_modules(self, RANK, G, D, **kwargs):
        ddp_modules = {
            'G_mapping'  : G.mapping, 
            'G_encoder'  : G.encoder,
            'G_synthesis': G.synthesis, 
            'D'          : D,
        }
        ddp_modules.update(kwargs)
        for name in ddp_modules.keys():
            module = ddp_modules[name]
            if len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[RANK], broadcast_buffers=False)
                module.requires_grad_(False)
            ddp_modules[name] = module
        return ddp_modules

    def __call__(self, RANK, **paras):
        cfg = cfguh().cfg
        cfge   = cfg.env
        cfgv   = cfg.eval
        cfgm_g = cfg.model_g
        cfgm_d = cfg.model_d

        # Initialize.
        device = torch.device('cuda', RANK)
        if cfge.rnd_seed is not None:
            np.random.seed(cfge.rnd_seed * cfge.gpu_count + RANK)
            torch.manual_seed(cfge.rnd_seed * cfge.gpu_count + RANK)
        torch.backends.cudnn.benchmark = cfge.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cfge.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfge.allow_tf32

        if RANK == 0:
            highlight_print('Setting')
            print_log('RNDSEED:', cfge.rnd_seed)
            print_log('VAL_DATASET:', cfgv.dataset.name)
            print_log('MODEL_G:', cfgm_g.name)
            print_log('MODEL_D:', cfgm_d.name)

        if cfge.debug and RANK==0:
            highlight_print('Debug')

        #################
        # Load eval set #
        #################

        if RANK == 0:
            highlight_print('My dataset')

        batch_gpu_eval = cfgv.batch_size_per_gpu
        num_workers_gpu_eval = cfgv.dataset_num_workers_per_gpu
        evalset = get_dataset()(cfgv.dataset)
        sampler = DistributedSampler(evalset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(
            evalset, 
            batch_size = batch_gpu_eval,
            sampler = sampler,
            num_workers = num_workers_gpu_eval,
            drop_last = False, 
            pin_memory = False,
            collate_fn = collate(), 
        )

        ######################
        # Construct networks #
        ######################

        if RANK == 0:
            highlight_print('My model')

        G = get_model()(cfgm_g).eval().requires_grad_(False).to(device)
        D = get_model()(cfgm_d).eval().requires_grad_(False).to(device)
        G_ema = copy.deepcopy(G).eval()

        # Resume from existing pickle.
        resume_pkl = getattr(cfgv, 'pretrained_pkl', None)
        if (resume_pkl is not None) and (RANK == 0):
            resume_pkl = osp.abspath(osp.join(cfg.eval.log_dir, '..', resume_pkl))
            G, D, G_ema = self.resume(resume_pkl = resume_pkl, models=[G, D, G_ema])

        # Load existing pth.
        pretrained_pth = cfgv.get('pretrained_pth', None)
        strict_sd = cfgv.get('strict_sd', True)
        if (pretrained_pth is not None) and (RANK == 0):
            gema_sd = torch.load(pretrained_pth, map_location='cpu')
            G_ema.load_state_dict(gema_sd, strict=strict_sd)
            print('Load from [{}] strict_sd [{}]'.format(pretrained_pth, strict_sd))

        if RANK == 0:
            highlight_print('Distributing across GPUs...')

        ddp_modules = self.get_ddp_modules(RANK, G, D, G_ema=G_ema)
        ddp_modules.pop('G_ema')
        ddp_modules['augment_pipe'] = None

        # Export sample images.
        output_sample_images = getattr(cfgv, 'output_sample_images', True)
        if (RANK == 0) and output_sample_images:
            highlight_print('Exporting sample images...')
            demof = self.draw_function(RANK, [8, 6], batch_gpu_eval, cfgv.log_dir)
            demof(z_dim=G.z_dim, generator=G_ema, evalset=evalset, 
                  filename='fakes.png', isinit=True)

        if RANK == 0:
            highlight_print('Evaluating metrics...')

        ################
        # My evaluator #
        ################

        from ..evaluator import get_evaluator
        from ..utils import torch_to_numpy
        import timeit

        evaluator = get_evaluator()(cfgv.evaluator)
        G_ema_foreval = copy.deepcopy(G_ema).eval().requires_grad_(False).to(device)

        def run_generator(x, z, c):
            m = x[:, 0:1]+0.5
            img = G_ema_foreval(x=x, z=z, c=c, noise_mode='random')
            img_combined = x[:, 1:4]*m + img*(1-m)
            img_combined = (img_combined * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
            return img_combined

        time_check = timeit.default_timer()

        for idx, batch in enumerate(evalloader): 
            real, mask, fn = batch

            real = real.to(device)
            mask = mask[:, None].to(device)
            real_erased = real * mask
            batch_size = real.size(0)

            x = torch.cat([mask-0.5, real_erased], dim=1)
            z = torch.randn([batch_size, G.z_dim], device=device)
            c = torch.randn([batch_size, G.c_dim], device=device)
            with torch.no_grad():
                fake = run_generator(x, z, c)

            # pred gt are normalized to 0-1 (for LPIPS, PSNR, SSIM)
            # fake real are 0-255 (for stylegan fid computation) and kept on cuda device for memory saving

            rv = {
                'pred' : torch_to_numpy(fake)/255,
                'gt'   : (torch_to_numpy(real)+1)/2,
                'fake' : fake.float(),
                'real' : real.float()*127.5 + 127.5,
                'fn'   : fn,
            }

            evaluator.add_batch(**rv)
            if idx % cfgv.log_display == cfgv.log_display-1:
                print_log('processed.. {}, Time:{:.2f}s'.format(
                    idx+1, timeit.default_timer() - time_check))
                time_check = timeit.default_timer()

        evaluator.set_sample_n(len(evalloader.dataset))
        eval_rv = evaluator.compute()
        if RANK == 0:
            evaluator.one_line_summary()
            evaluator.save(cfgv.log_dir)
        evaluator.clear_data()
        return {'eval_rv' : eval_rv}
