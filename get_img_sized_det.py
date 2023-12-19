import time

all_run_st = time.time()
import_st = time.time()

import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import Profile, check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

import_t = time.time() - import_st

weights = '/media/nnthao/yolov5/runs/train/exp-mask/weights/best.pt'    # model path or triton URL
data = '/home/nnthao/project/yolov5/data/datamask.yaml' # dataset.yaml path
device = ','.join(str(i) for i in list(range(torch.cuda.device_count())))   # cuda device, i.e. 0 or 0,1,2,3 or cpu
half=False  # use FP16 half-precision inference
dnn=False   # use OpenCV DNN for ONNX inference
imgsz = (512, 512)  # inference size (height, width)

# Load model
device = select_device(device)

load_model_st = time.time()
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
load_model_t = time.time() - load_model_st

stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size


augment=False   # augmented inference
visualize=False # visualize features
conf_thres=0.25 # confidence threshold
iou_thres=0.45  # NMS IOU threshold
classes=None    # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
max_det=18  # maximum detections per image
batch_size = 1  # batch_size
box_color = (255, 255, 255)

# Run inference
warmup_st = time.time()
model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgsz))  # warmup
warmup_t = time.time() - warmup_st

dt = (Profile(), Profile(), Profile())

from torchvision.io import read_image
images_in = ((torch.cat((read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/images/test1.png')[None, ...], 
       read_image('/home/nnthao/project/FDA/test_sets/CelebA-HQ/images/test2.png')[None, ...]))) / 255).to(device)

with dt[0]:
    images_in = images_in.half() if model.fp16 else images_in.float()  # uint8 to fp16/32
    if len(images_in.shape) == 3:
        images_in = images_in[None]  # expand for batch dim

# Inference
with dt[1]:
    infer_st = time.time()
    pred = model(images_in, augment=augment, visualize=visualize)
    infer_t = time.time() - infer_st

# NMS
with dt[2]:
    nms_st = time.time()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    nms_t = time.time() - nms_st

# Coordinate det (x-min, y-min, x-max, y-max) to image-sized det (batch, 1, res, res)
size_trans_st = time.time()
dets_in = None

for i, det in enumerate(pred):  # per image
    det_in = torch.zeros(512, 512, 3, dtype=torch.uint8).contiguous().cpu().detach().numpy()

    if len(det):
        # Rescale boxes from img_size to det_in size
        det[:, :4] = scale_boxes(images_in.shape[2:], det[:, :4], det_in.shape).round()

        # Fill fully in boxes with white
        for *xyxy, conf, cls in reversed(det):
            cv2.rectangle(det_in, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), box_color, -1)

    det_in = torch.from_numpy(det_in[..., 0].reshape(1, 1, 512, 512)).to(model.device)
    det_in = det_in.half() if model.fp16 else det_in.float()  # uint8 to fp16/32
    det_in /= 255  # 0 - 255 to 0.0 - 1.0

    if i:
        dets_in = torch.cat((dets_in, det_in))
    else:
        dets_in = det_in

size_trans_t = time.time() - size_trans_st
all_run_t = time.time() - all_run_st

print('import time: ', import_t)
print('load model time: ', load_model_t)
print('warmup time:', warmup_t)
print('infer time: ', infer_t)
print('nms time: ', nms_t)
print('image-sized tranform time: ', size_trans_t)
print('all running time: ', all_run_t)


# Testing
import matplotlib.image as mpimg
import numpy as np

for det_in in dets_in:
    mpimg.imsave('./imshow-ngu-vc.png', np.concatenate(((det_in.permute(1, 2, 0) * 255).floor().contiguous().cpu().detach().numpy(), 
                                                (det_in.permute(1, 2, 0) * 255).floor().contiguous().cpu().detach().numpy(), 
                                                (det_in.permute(1, 2, 0) * 255).floor().contiguous().cpu().detach().numpy()), axis=2).astype('uint8'))
    input()