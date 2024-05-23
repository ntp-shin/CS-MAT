from _my_utils import build_model
from evaluatoin import cal_fid_pids_uids, cal_lpips, cal_psnr_ssim_l1
import time

fpu_t = time.time()
print(cal_fid_pids_uids.calculate_metrics('/media/nnthao/MAT/Data/CelebA-HQ/cswintv62_5m12_sgen_s240_samples', '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img/'))
print('fpu_t', time.time() - fpu_t)

# lpips_t = time.time()
# print(cal_lpips.calculate_metrics('/media/nnthao/MAT/Data/CelebA-HQ/cswintv6_15r0_1200_samples', '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img/'))
# print('lpips_t', time.time() - lpips_t)
# psl_t = time.time()
# print(cal_psnr_ssim_l1.calculate_metrics('/media/nnthao/MAT/Data/CelebA-HQ/cswintv6_15r0_1200_samples', '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img/'))
# print('psl_t', time.time() - psl_t)