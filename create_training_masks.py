from datasets import mask_generator_512
import matplotlib.pyplot as plt
import numpy as np

for i in range(30000):
    mask = mask_generator_512.RandomMask(512).transpose(1, 2, 0)
    plt.imsave('/media/nnthao/MAT/training_masks/' + str(i) + '.png', np.concatenate((mask, mask, mask), axis=2))