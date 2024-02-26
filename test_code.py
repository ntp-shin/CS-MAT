from _my_utils import fid_evaluating, build_model
from evaluatoin import cal_fid_pids_uids

# fid_evaluating('/media/nnthao/MAT/saved_model/good_cswint_model/11-train240_2400-best240_2400-resuming_best10-13h16m/network-snapshot-000240_2400.pkl')
print(cal_fid_pids_uids.calculate_metrics('/media/nnthao/MAT/Data/CelebA-HQ/cswintv6_samples', '/media/nnthao/MAT/Data/CelebA-HQ/CelebA-HQ-val_img/'))
# build_model()