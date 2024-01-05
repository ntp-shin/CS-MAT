import torch
from networks.mat import *
from torchvision.io import read_image
import time

from _my_utils import *

device = torch.device('cuda:0')
batch = 8
res = 512
img_channels = 3
z_dim = torch.empty([batch, res], device=device)
c_dim = torch.empty([batch, 0], device=device)

initG_start_time = time.time()
G = Generator(z_dim=res, c_dim=0, w_dim=res, img_resolution=res, img_channels=img_channels).to(device)
initG_time = time.time() - initG_start_time

D = Discriminator(c_dim=0, img_resolution=res, img_channels=img_channels).to(device)

# img = ((torch.cat((read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/images/test1.png')[None, ...], 
#        read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/images/test2.png')[None, ...]))) / 255).to(device)
# mask = ((torch.cat((read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/masks/mask1.png')[None, None, 0], 
#        read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/masks/mask2.png')[None, None, 0]))) / 255).to(device)

img = torch.randn(batch, 3, res, res).to(device)
mask = torch.randn(batch, 1, res, res).to(device)

misc.print_module_summary(G, [img, mask, z_dim, c_dim])
print()
print_model_tensors_stats(G, 'Generator', False, 1)
print()
print_model_tensors_stats(D, 'Discriminator', False, 1)
print()

z = torch.randn(batch, res).to(device)

evalG_start_time = time.time()
G.eval()
evalG_time = time.time() - evalG_start_time

# def count(block):
#     return sum(p.numel() for p in block.parameters()) / 10 ** 6
# print('Generator', count(G))
# print('discriminator', count(D))

genG_start_time = time.time()
with torch.no_grad():
    img, img_stg1 = G(img, mask, z, None, return_stg1=True)
genG_time = time.time() - genG_start_time

print('output of G:', img.shape, img_stg1.shape)
score, score_stg1 = D(img, mask, img_stg1, None)
print('output of D:', score.shape, score_stg1.shape)

print('initialized G time: ', initG_time)
print('evaluating G time: ', evalG_time)
print('generating G time: ', genG_time)