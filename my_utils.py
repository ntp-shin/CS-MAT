def model_params_stats(model, named_model, see_named_params=False, num_gpus=0):
    ts = 0
    nts = 0
    ts_per_dvs = [0] * (num_gpus + 1)

    print('Unknown') if named_model == None or named_model == '' else print(named_model)

    for n, p in model.named_parameters():
        if see_named_params:
            print(n)
            print('\t- Total params:', p.numel())
            print('\t- Device:', p.get_device())
        
        ts_per_dvs[p.get_device()] += 1
        if p.requires_grad:
            ts += p.numel()
        else:
            nts += p.numel()

    if see_named_params:
        print()

    print('Trainable params:', ts)
    print('Non-Trainable params:', nts)
    print('Tensors per device:')

    for i in range(len(ts_per_dvs)):
        if i == len(ts_per_dvs) - 1:
            print(f'- cpus\t{ts_per_dvs[i]}')
            break

        print(f'- cuda:{i}\t{ts_per_dvs[i]}')

def trainable_except(model, except_name='yolov5'):
    for n, p in model.named_parameters():
        if except_name in n:
            p.requires_grad = False
        else:
            p.requires_grad = True