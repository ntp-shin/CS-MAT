import torch
from _my_utils import build_model

# build_model(device=torch.device('cpu'))
# build_model()
# build_model(device=torch.device('cpu'), network_pkl='/media/nnthao/MAT/saved_model/cmat/00009-CelebA-HQ-img-mirror-celeba512-mat_v1-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug-resumecustom/network-snapshot-000600.pkl')
# build_model(network_pkl='/media/nnthao/MAT/saved_model/cmat/00009-CelebA-HQ-img-mirror-celeba512-mat_v1-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug-resumecustom/network-snapshot-000600.pkl')

a = torch.rand(6)
print(a)
a.reshape(2, 3)[0][0] = 99
print(a)