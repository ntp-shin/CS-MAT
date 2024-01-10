import dnnlib
import matplotlib.pyplot as plt

dataloader = 'datasets.dataset_512.ImageFolderMaskDataset'
data_val = '/home/nnthao/lntuong/FDA/test_sets/CelebA-HQ/images3'

val_set_kwargs = dnnlib.EasyDict(class_name=dataloader, path=data_val, use_labels=False, max_size=2993, xflip=False, resolution=512)
val_set = dnnlib.util.construct_class_by_name(**val_set_kwargs) # subclass of training.dataset.Dataset
print(val_set[0][0].flags)
# plt.imsave('det.png', val_set[0][0].transpose(1, 2, 0).copy(order='C'))