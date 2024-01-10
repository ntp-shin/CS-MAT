import dnnlib
import legacy
import torch
from metrics import metric_main

# network_pkl = '/media/nnthao/MAT/saved_model/00011-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug-resumecustom/network-snapshot-000600.pkl'

# network_pkl = '/media/nnthao/MAT/saved_model/00103-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000600.pkl'
# network_pkl = '/media/nnthao/MAT/saved_model/00111-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug-resumecustom/network-snapshot-000600.pkl'

# network_pkl = '/media/nnthao/MAT/saved_model/00118-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000120.pkl'
network_pkl = '/media/nnthao/MAT/saved_model/00118-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000200.pkl'

# network_pkl = '/media/nnthao/MAT/old_model/00000-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000000.pkl'
# network_pkl = '/media/nnthao/MAT/old_model/00000-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000040.pkl'

device = torch.device('cuda:0')
metrics = ['fid2993_full']
rank = 0
num_gpus = 1
dataloader = 'datasets.dataset_512.ImageFolderMaskDataset'
data_val = '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img'
val_set_kwargs = dnnlib.EasyDict(class_name=dataloader, path=data_val, use_labels=False, max_size=2993, xflip=False, resolution=512)
print(f'Loading networks from: {network_pkl}')

with dnnlib.util.open_url(network_pkl) as f:
    model = legacy.load_network_pkl(f)
    G = model['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    D = model['D'].to(device)

print('Evaluating metrics...')
for metric in metrics:
    result_dict = metric_main.calc_metric(metric=metric, G=G,
        dataset_kwargs=val_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
    if rank == 0:
        metric_main.report_metric(result_dict)
del model # conserve memory