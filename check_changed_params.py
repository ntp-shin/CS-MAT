
import torch
import dnnlib
import legacy

inception_pkl = '/home/nnthao/.cache/dnnlib/gan-metrics/CelebA-HQ-val_img-inception-2015-12-05-871859055e49b26f93953f8cc77f698d.pkl'
# network_pkl = '/media/nnthao/MAT/old_model/00000-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000040.pkl'

# first_network_pkl = '/media/nnthao/MAT/saved_model/00118-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000120.pkl'
network_pkl = '/media/nnthao/MAT/saved_model/00118-CelebA-HQ-img-mirror-celeba512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg600-batch16-tc0.5-sm0.5-ema10-noaug/network-snapshot-000200.pkl'

device = torch.device('cuda:0')

print(f'Loading networks from: {network_pkl}')
with dnnlib.util.open_url(network_pkl) as f:
    model = legacy.load_network_pkl(f)
    G = model['G'].to(device).eval().requires_grad_(False) # type: ignore

yolo_path = '/media/nnthao/yolov5/runs/train/exp-mask/weights/best.pt'
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path, force_reload=True)

pG = []
for n, p in G.named_parameters():
    if 'yolov5' in n:
        pG.append(p)

print(len(pG), len(list(yolo.parameters())))
for (n, yp), p in zip(yolo.named_parameters(), pG):
    if (p != yp).sum():
        print(n, (p != yp).sum())

# print(f'Loading networks from: {first_network_pkl}')
# with dnnlib.util.open_url(first_network_pkl) as f:
#     model2 = legacy.load_network_pkl(f)
#     G2 = model2['G'].to(device).eval().requires_grad_(False) # type: ignore
        
# for (n, p), (_, p2) in zip(G.named_parameters(), G2.named_parameters()):
#     if 'yolov5' in n:
#         if (p != p2).sum():
#             print(n, (p != p2).sum())