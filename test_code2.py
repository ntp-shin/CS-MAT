from _my_utils import fid_evaluating
import time

random_seed = 0

for random_seed in range(0, 2):
    print('random_seed', random_seed)

    fid_t = time.time()
    fid_evaluating('/media/nnthao/MAT/saved_model/good_cswint/13-train1200_3800-best1160_3760-resuming_train12-58h30m/network-snapshot-001000_3600.pkl', random_seed=random_seed)
    print('fid_t', time.time() - fid_t)