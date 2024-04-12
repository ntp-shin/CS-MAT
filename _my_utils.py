import psutil
import os
import contextlib
import glob
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision.io import read_image
import PIL.Image

import dnnlib
import legacy
from torch_utils import misc
from datasets import mask_generator_512
from metrics import metric_main
from networks import mat

def build_model(res=512, c=0, img_channels=3, batch=1, device=torch.device('cuda:0'), network_pkl=None):
    initG_start_time = time.time()
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            model = legacy.load_network_pkl(f)
            _G = model['G_ema'].to(device).requires_grad_(False) # type: ignore
            
        G = mat.Generator(z_dim=res, c_dim=c, w_dim=res, img_resolution=res, img_channels=img_channels).to(device).requires_grad_(False)
        for p, _p in zip(G.parameters(), _G.parameters()):
            p.data = torch.nn.parameter.Parameter(_p.detach())
    else:
        G = mat.Generator(z_dim=res, c_dim=c, w_dim=res, img_resolution=res, img_channels=img_channels).to(device)
    initG_time = time.time() - initG_start_time

    evalG_start_time = time.time()
    G.eval()
    evalG_time = time.time() - evalG_start_time

    D = mat.Discriminator(c_dim=0, img_resolution=res, img_channels=img_channels).to(device)

    img = torch.from_numpy(np.array(PIL.Image.open('/home/nnthao/lntuong/FDA/test_sets/CelebA-HQ/images/test2.png'))).permute(2, 0, 1)[None].to(device) / 127.5 - 1
    # mask = torch.from_numpy(np.array(PIL.Image.open('/media/nnthao/MAT/saved_model/model_feature_snapshots/mask_in.png')))[None][None].to(device) / 255
    mask = torch.from_numpy(mask_generator_512.RandomMask(res, hole_range=[.4, 6.])[None]).to(device)
    batch = 1

    # img = torch.randn(batch, 3, res, res).to(device)
    # mask = torch.randn(batch, 1, res, res).to(device)
    z_dim = torch.randn(batch, res).to(device)
    c_dim = torch.randn(batch, c).to(device)

    # misc.print_module_summary(G, [img, mask, z_dim, c_dim])
    # print()
    # print_model_tensors_stats(G, 'Generator', False, 1)
    # print()
    # print_model_tensors_stats(D, 'Discriminator', False, 1)
    # print()

    # print('G size:', cal_model_size(G) / 1024**2, 'MB')
    # print('D size:', cal_model_size(D) / 1024**2, 'MB')

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