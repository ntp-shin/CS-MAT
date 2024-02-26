import contextlib
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.io import read_image

import dnnlib
import legacy
from torch_utils import misc
from datasets import mask_generator_512
from metrics import metric_main
from networks import mat

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

def trainable_except(model, except_name='yolov5'):
    for n, p in model.named_parameters():
        if except_name in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def my_check_ddp_consistency(named_modules, num_gpus=1):
    # [('G', G), ('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('G_ema', G_ema), ('D', D), ('augment_pipe', augment_pipe)]
    for name, module in named_modules:
        if module is not None:
            if num_gpus > 1:
                misc.check_ddp_consistency(module, ignore_regex=[r'.*\.w_avg', r'.*\.relative_position_index', r'.*\.avg_weight', r'.*\.attn_mask', r'.*\.resample_filter'])


def check_tensors_model_change(pre_named_params, pos_named_params, pre_named_buff, pos_named_buff, see_changed_modules=False):
    change = False

    for (n, pre_p), (_, pos_p) in zip(pre_named_params, pos_named_params):
        if (pre_p.detach() != pos_p.detach()).sum() and 'yolov5' not in n:
            change = True
            if see_changed_modules:
                print(n)
    
    for (buff_n, pre_b), (_, pos_b) in zip(pre_named_buff, pos_named_buff):
        if (pre_b.detach() != pos_b.detach()).sum() and 'yolov5' not in buff_n:
            change = True
            if see_changed_modules:
                print(buff_n)

    return change

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

def create_training_masks(path='/media/nnthao/MAT/training_masks/', n=30000):
    for i in range(n):
        mask = mask_generator_512.RandomMask(512).transpose(1, 2, 0)
        plt.imsave(path + str(i) + '.png', np.concatenate((mask, mask, mask), axis=2))


def build_model(res=512, c=0, img_channels=3, batch=2, device=torch.device('cuda:0')):
    initG_start_time = time.time()
    G = mat.Generator(z_dim=res, c_dim=c, w_dim=res, img_resolution=res, img_channels=img_channels).to(device)
    initG_time = time.time() - initG_start_time

    evalG_start_time = time.time()
    G.eval()
    evalG_time = time.time() - evalG_start_time

    D = mat.Discriminator(c_dim=0, img_resolution=res, img_channels=img_channels).to(device)

    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z_dim = torch.randn(batch, res).to(device)
    c_dim = torch.randn(batch, c).to(device)

    misc.print_module_summary(G, [img, mask, z_dim, c_dim])
    print()
    print_model_tensors_stats(G, 'Generator', False, 1)
    print()
    print_model_tensors_stats(D, 'Discriminator', False, 1)
    print()

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

def fid_evaluating(network_pkl, res=512, max_size=2993, use_labels=False, xflip=False,
                   data_val='/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img',
                   data_loader='datasets.dataset_512.ImageFolderMaskDataset',
                   metrics=['fid2993_full'], device=torch.device('cuda:0'), num_gpus=1, rank=0):
    val_set_kwargs = dnnlib.EasyDict(class_name=data_loader, path=data_val, use_labels=use_labels, max_size=max_size, xflip=xflip, resolution=res)

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