from _my_utils import fid_evaluating
import time

random_seed = 0
for random_seed in range(0, 2):
    print('random_seed', random_seed)

    fid_t = time.time()
    fid_evaluating('/media/nnthao/CS-MAT/saved_model/csmat.pkl', random_seed=random_seed)
    print('fid_t', time.time() - fid_t)