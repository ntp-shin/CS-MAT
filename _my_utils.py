
import math
import numpy as np
import matplotlib.pyplot as plt

from torch_utils import misc

def print_model_tensors_stats(model, named_model, detail=False, num_gpus=0):
    ts, buff_ts = 0, 0
    nts, buff_nts = 0, 0
    ts_per_dvs, buff_ts_per_dvs = [0] * (num_gpus + 1), [0] * (num_gpus + 1)


    print('PARAMS-MODEL') if named_model == None or named_model == '' else print('PARAMS::' + named_model)

    for n, p in model.named_parameters():
        if detail:
            print(n)
            print('\t- Params sum:', p.numel())
            print('\t- params device:', p.get_device())
        
        ts_per_dvs[p.get_device()] += 1
        if p.requires_grad:
            ts += p.numel()
        else:
            nts += p.numel()
    
    if detail:
        print()
    
    print('Trainable params:', ts)
    print('Non-Trainable params:', nts)
    print('Params-tensors per device:')

    for i in range(len(ts_per_dvs)):
        if i == len(ts_per_dvs) - 1:
            print(f'- cpus\t{ts_per_dvs[i]}')
            break

        print(f'- cuda{i}\t{ts_per_dvs[i]}')


    print('\nBUFF-MODEL') if named_model == None or named_model == '' else print('BUFF::' + named_model)

    for buff_n, b in model.named_buffers():
        if detail:
            print(buff_n)
            print('\t- Buff sum:', b.numel())
            print('\t- Buff device:', b.get_device())
        
        buff_ts_per_dvs[b.get_device()] += 1
        if b.requires_grad:
            buff_ts += b.numel()
        else:
            buff_nts += b.numel()

    if detail:
        print()

    print('Trainable buff:', buff_ts)
    print('Non-Trainable buff:', buff_nts)
    print('Buff-tensors per device:')
    
    for i in range(len(buff_ts_per_dvs)):
        if i == len(buff_ts_per_dvs) - 1:
            print(f'- cpus\t{buff_ts_per_dvs[i]}')
            break

        print(f'- cuda{i}\t{buff_ts_per_dvs[i]}')

def trainable_except(model, except_name='yolo_net'):
    for n, p in model.named_parameters():
        if except_name in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def my_check_ddp_consistency(named_modules, num_gpus=1):
    # [('G', G), ('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('G_ema', G_ema), ('D', D), ('augment_pipe', augment_pipe)]
    for name, module in named_modules:
        if module is not None:
            if num_gpus > 1:
                misc.check_ddp_consistency(module, ignore_regex=[r'.*\.w_avg', r'.*\.relative_position_index', r'.*\.avg_weight', r'.*\.attn_mask', r'.*\.resample_filter'])


def check_tensors_model_change(pre_named_params, pos_named_params, pre_named_buff, pos_named_buff, see_changed_modules=False):
    change = False

    for (n, pre_p), (_, pos_p) in zip(pre_named_params, pos_named_params):
        if (pre_p.detach() != pos_p.detach()).sum() and 'yolo_net' not in n:
            change = True
            if see_changed_modules:
                print(n)
    
    for (buff_n, pre_b), (_, pos_b) in zip(pre_named_buff, pos_named_buff):
        if (pre_b.detach() != pos_b.detach()).sum() and 'yolo_net' not in buff_n:
            change = True
            if see_changed_modules:
                print(buff_n)

    return change

def plot_images(images, layout='square', layout_size=(), figsize=(16, 16), filename=None):
    h, w, c = images.shape[1:]
    
    if layout == 'square':
        hgrid_size = wgrid_size = math.ceil(np.sqrt(images.shape[0]))
    elif layout == 'rectangle':
        hgrid_size = layout_size[0]
        wgrid_size = layout_size[1]

    images = (images + 1) / 2. * 255.
    images = images.astype(np.uint8)
    images = (images.reshape(hgrid_size, wgrid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(hgrid_size*h, wgrid_size*w, c))
    
    plt.figure(figsize=figsize)
    if filename != None:
        plt.imsave(filename, images)
    plt.imshow(images)
    plt.show()