import psutil
import os
import glob
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import dnnlib
import legacy
from torch_utils import misc
from datasets import mask_generator_512
from metrics import metric_main
from networks import csmat

def plot_images(images, layout='square', layout_size=(), figsize=(16, 16), filename=None):
    h, w, c = images.shape[1:]
    
    if layout == 'square':
        hgrid_size = wgrid_size = math.ceil(np.sqrt(images.shape[0]))
    elif layout == 'rectangle':
        hgrid_size = layout_size[0]
        wgrid_size = layout_size[1]

    images = (images + 1) / 2. * 255.
    images = images.astype(np.uint8)
    images = (images.reshape(hgrid_size, wgrid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(hgrid_size*h, wgrid_size*w, c))
    
    plt.figure(figsize=figsize)
    if filename != None:
        plt.imsave(filename, images)
    plt.imshow(images)
    plt.show()

def plot_img_gird(images, layout='square', layout_size=(), figsize=(16, 16), filename=None):
    h, w, c = images.shape[1:]
    
    if layout == 'square':
        hgrid_size = wgrid_size = math.ceil(np.sqrt(images.shape[0]))
    elif layout == 'rectangle':
        hgrid_size = layout_size[0]
        wgrid_size = layout_size[1]

    images = (images + 1) / 2. * 255.
    images = images.astype(np.uint8)
    images = (images.reshape(hgrid_size, wgrid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(hgrid_size*h, wgrid_size*w, c))
    
    plt.figure(figsize=figsize)
    if filename != None:
        plt.imsave(filename, images)
    plt.imshow(images)
    plt.show()

def plot_img_paths(dpath=None, mpath=None, out1dir=None, outdir=None, sample=None, img_num=8):
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))
    mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
    if out1dir is not None:
        sample1_list = sorted(glob.glob(out1dir + '/*.png') + glob.glob(out1dir + '/*.jpg'))
    sample_list = sorted(glob.glob(outdir + '/*.png') + glob.glob(outdir + '/*.jpg'))

    if sample == 'random':
        idxs = np.random.randint(0, len(img_list), size=img_num)
    elif sample == 'tail':
        idxs = np.arange(-img_num, 0)
    else:
        idxs = np.arange(0, img_num)

    full_imgs = []

    for i in idxs:
        img = mpimg.imread(img_list[i])
        if img.max() > 1:
            img = img /255
        if img.shape[-1] == 4:
            img = img[..., :-1]
        full_imgs.append(img)

    for i in idxs:
        img = mpimg.imread(img_list[i])
        mask = mpimg.imread(mask_list[i])
        
        if img.max() > 1:
            img = img /255
        if img.shape[-1] == 4:
            img = img[..., :-1]

        if mask.max() > 1:
            mask = mask /255
        if mask.shape[-1] == 4:
            mask = mask[..., :-1]
        if len(mask.shape) == 2:
            mask = mask[..., None]

        full_imgs.append(img * mask)

    if out1dir is not None:
        for i in idxs:
            sample1 = mpimg.imread(sample1_list[i])
            if sample1.max() > 1:
                sample1 = sample1 /255
            if sample1.shape[-1] == 4:
                sample1 = sample1[..., :-1]
            full_imgs.append(sample1)

    for i in idxs:
        sample = mpimg.imread(sample_list[i])
        if sample.max() > 1:
            sample = sample /255
        if sample.shape[-1] == 4:
            sample = sample[..., :-1]
        full_imgs.append(sample)

    plot_img_gird(np.array(full_imgs), 'rectangle', (int(len(full_imgs) / img_num), img_num))

def print_model_tensors_stats(model, named_model, detail=False, num_gpus=0):
    ts, buff_ts = 0, 0
    nts, buff_nts = 0, 0
    ts_per_dvs, buff_ts_per_dvs = [0] * (num_gpus + 1), [0] * (num_gpus + 1)


    print('PARAMS-MODEL') if named_model == None or named_model == '' else print('PARAMS::' + named_model)

    for n, p in model.named_parameters():
        if detail:
            print(n)
            print('\t- Params sum:', p.numel())
            print('\t- params device:', p.get_device())
        
        ts_per_dvs[p.get_device()] += 1
        if p.requires_grad:
            ts += p.numel()
        else:
            nts += p.numel()
    
    if detail:
        print()
    
    print('Trainable params:', ts)
    print('Non-Trainable params:', nts)
    print('Params-tensors per device:')

    for i in range(len(ts_per_dvs)):
        if i == len(ts_per_dvs) - 1:
            print(f'- cpus\t{ts_per_dvs[i]}')
            break

        print(f'- cuda{i}\t{ts_per_dvs[i]}')


    print('\nBUFF-MODEL') if named_model == None or named_model == '' else print('BUFF::' + named_model)

    for buff_n, b in model.named_buffers():
        if detail:
            print(buff_n)
            print('\t- Buff sum:', b.numel())
            print('\t- Buff device:', b.get_device())
        
        buff_ts_per_dvs[b.get_device()] += 1
        if b.requires_grad:
            buff_ts += b.numel()
        else:
            buff_nts += b.numel()

    if detail:
        print()

    print('Trainable buff:', buff_ts)
    print('Non-Trainable buff:', buff_nts)
    print('Buff-tensors per device:')
    
    for i in range(len(buff_ts_per_dvs)):
        if i == len(buff_ts_per_dvs) - 1:
            print(f'- cpus\t{buff_ts_per_dvs[i]}')
            break

        print(f'- cuda{i}\t{buff_ts_per_dvs[i]}')

def print_modules_stats(model):
    tran, full_attn, cswin_attn, lepe, fuse, mlp = 0, 0, 0, 0, 0, 0
    for n, p in model.named_parameters():
        if '.tran' in n:
            tran += p.numel()
        if '.full_attn' in n:
            full_attn += p.numel()
        if '.blocks' in n and ('.qkv' in n or '.proj' in n) and 'full_attn' not in n:
            cswin_attn += p.numel()
        if  '.blocks' in n and '.attns' in n:
            lepe += p.numel()
        if '.fuse' in n:
            fuse += p.numel()
        if '.mlp' in n:
            mlp += p.numel()
    print(f'Params stat:\n- tran\t{tran}\n- full_attn\t{full_attn}\n- cswin_attn\t{cswin_attn}\n- lepe\t{lepe}\n- fuse\t{fuse}\n- mlp\t{mlp}')

def cal_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size

def create_training_masks(path, n=24183):
    for i in range(n):
        mask = mask_generator_512.RandomMask(512).transpose(1, 2, 0)
        plt.imsave(path + str(i) + '.png', np.concatenate((mask, mask, mask), axis=2))

def build_model(res=512, c=0, img_channels=3, batch=1, device=torch.device('cuda:0'), network_pkl=None):
    initG_start_time = time.time()
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            model = legacy.load_network_pkl(f)
            _G = model['G_ema'].to(device).requires_grad_(False) # type: ignore
            
        G = csmat.Generator(z_dim=res, c_dim=c, w_dim=res, img_resolution=res, img_channels=img_channels).to(device).requires_grad_(False)
        for p, _p in zip(G.parameters(), _G.parameters()):
            p.data = torch.nn.parameter.Parameter(_p.detach())
    else:
        G = csmat.Generator(z_dim=res, c_dim=c, w_dim=res, img_resolution=res, img_channels=img_channels).to(device)
    initG_time = time.time() - initG_start_time

    evalG_start_time = time.time()
    G.eval()
    evalG_time = time.time() - evalG_start_time

    D = csmat.Discriminator(c_dim=0, img_resolution=res, img_channels=img_channels).to(device)

    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z_dim = torch.randn(batch, res).to(device)
    c_dim = torch.randn(batch, c).to(device)
    batch = 1

    misc.print_module_summary(G, [img, mask, z_dim, c_dim])

    # print_model_tensors_stats(G, 'Generator', True, 1)
    # print()
    # print_model_tensors_stats(D, 'Discriminator', False, 1)
    # print()

    # print_modules_stats(G)

    print('G size:', cal_model_size(G) / 1024**2, 'MB')
    print('D size:', cal_model_size(D) / 1024**2, 'MB')

    genG_start_time = time.time()
    with torch.no_grad():
        img, img_stg1 = G(img, mask, z_dim, None, return_stg1=True)
    genG_time = time.time() - genG_start_time

    score, score_stg1 = D(img, mask, img_stg1, None)

    print('output of G:', img.shape, img_stg1.shape)
    print('output of D:', score.shape, score_stg1.shape)
    print()
    print('initializing G time: ', initG_time)
    print('evaluating G time: ', evalG_time)
    print('generating G time: ', genG_time)

    for i in range(torch.cuda.device_count()):
        print(i)
        print('gpu alloc', torch.cuda.memory_allocated(i) / 2**30)
        print('max gpu alloc', torch.cuda.max_memory_allocated(i) / 2**30)
        print('cpu alloc', psutil.Process(os.getpid()).memory_info().rss / 2**30)
    
def fid_evaluating(network_pkl, res=512, max_size=2993, use_labels=False, xflip=False,
                   data_val='/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img',
                   edge_path='/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_edge',
                   data_loader='datasets.dataset_512.ImageFolderMaskDataset',
                   metrics=['fid2993_full'], device=torch.device('cuda:0'), num_gpus=1, rank=0, random_seed=None):
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed * num_gpus + rank)
        torch.manual_seed(random_seed * num_gpus + rank)
        torch.cuda.manual_seed(random_seed * num_gpus + rank)

    val_set_kwargs = dnnlib.EasyDict(class_name=data_loader, path=data_val, edge_path=edge_path, use_labels=use_labels, max_size=max_size, xflip=xflip, resolution=res)

    print(f'Loading networks from: {network_pkl}')
    with dnnlib.util.open_url(network_pkl) as f:
        model = legacy.load_network_pkl(f)
        G = model['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        # D = model['D'].to(device)

    print('Evaluating metrics...')
    for metric in metrics:
        result_dict = metric_main.calc_metric(metric=metric, G=G, dataset_kwargs=val_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        if rank == 0:
            metric_main.report_metric(result_dict)
    del model # conserve memory