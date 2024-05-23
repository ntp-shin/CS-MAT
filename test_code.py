from _my_utils import fid_evaluating, build_model
from evaluatoin import cal_fid_pids_uids, cal_lpips, cal_psnr_ssim_l1
import time

fpu_t = time.time()
print(cal_fid_pids_uids.calculate_metrics('/media/nnthao/MAT/Data/CelebA-HQ/cswintv4_4m8_sgen_s240_samples/', '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img'))
print('fpu_t', time.time() - fpu_t)

exit()

for random_seed in range(240, 241):
    print('random_seed', random_seed)

    fid_t = time.time()
    fid_evaluating('/media/nnthao/MAT/saved_model/cswint_v4/12-train640_4200-resuming11-24h39m/network-snapshot-000640_4200.pkl', random_seed=random_seed)
    print('fid_t', time.time() - fid_t)