from evaluatoin import cal_fid_pids_uids, cal_lpips, cal_psnr_ssim_l1
import time

fpu_t = time.time()
print(cal_fid_pids_uids.calculate_metrics('/media/nnthao/CS-MAT/Data/CelebA-HQ/test_samples', '/media/nnthao/CS-MAT/Data/CelebA-HQ/CelebA-HQ-test_img'))
print('fpu_t', time.time() - fpu_t)
lpips_t = time.time()
print(cal_lpips.calculate_metrics('/media/nnthao/CS-MAT/Data/CelebA-HQ/test_samples', '/media/nnthao/CS-MAT/Data/CelebA-HQ/CelebA-HQ-test_img'))
print('lpips_t', time.time() - lpips_t)
psl_t = time.time()
print(cal_psnr_ssim_l1.calculate_metrics('/media/nnthao/CS-MAT/Data/CelebA-HQ/test_samples', '/media/nnthao/CS-MAT/Data/CelebA-HQ/CelebA-HQ-test_img'))
print('psl_t', time.time() - psl_t)