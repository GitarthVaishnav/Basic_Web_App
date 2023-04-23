# import some common libraries
import numpy as np
from pathlib import Path
import os, json, cv2, random, glob, sys
import matplotlib.pyplot as plt
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm.auto import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Load model
img_size=640
agnostic_nms=False 
imgsz=(640, 640)
stride=32
auto=True
device=''
dnn=False
data=''
half=False
visualize = False
augment=False

data=ROOT / 'data/coco128.yaml'

#weights='/home/suryansh/Documents/Websites/Websites/mysite/yolo/yolov5/yolov5s.pt'
weights='/home/suryansh/Documents/Websites/Websites/mysite/yolo/yolov5/best.pt'
#path='/home/suryansh/Documents/Websites/Websites/mysite/yolo/yolov5/zidane.jpg'
path='/home/suryansh/Documents/Websites/Websites/mysite/yolo/yolov5/0_a_308.jpg'


device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
bs = 1 #batch size
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup



#Read Image
img0 = cv2.imread(path)  # BGR
#Padded Resize
img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
# Convert image to Yolo Format
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)







# Run inference

dt, seen = [0.0, 0.0, 0.0], 0   
t1 = time_sync()
im = torch.from_numpy(img).to(device)
im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim
t2 = time_sync()
dt[0] += t2 - t1


# Inference

pred = model(im, augment=augment, visualize=visualize)
t3 = time_sync()
dt[1] += t3 - t2

#Config
conf_thres=0.25  # confidence threshold
iou_thres=0.45
classes=None
max_det=1000
save_crop=False
line_thickness=3
view_img=False

# NMS
pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
print(pred)
dt[2] += time_sync() - t3

# Second-stage classifier (optional)
# pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

# Process predictions
for i, det in enumerate(pred):  # per image
    seen += 1
    im0 = img0.copy()
#    s += '%gx%g ' % im.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    imc = im0  # for save_crop
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
#            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()
    print(im0)
    cv2.imwrite("./debug.jpg",im0)
#    cv2.waitKey(10000)

