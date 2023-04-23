import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ExifTags, ImageOps
from django.conf import settings
from utils.general import (
    check_img_size, non_max_suppression, scale_coords
)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from django.conf import settings

def detector(model, image, img_size=640, conf_thres=0.20, iou_thres=0.40, max_det=1000, line_thickness=3, object_count={}):
    device = model.device
    
    # Prepare the image
    img = letterbox(image, img_size, stride=model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Convert image to torch tensor
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if model.fp16 else img_tensor.float()
    img_tensor /= 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]

    # Inference
    pred = model(img_tensor)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)

    # Process predictions
    im0 = image.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=line_thickness, example=str(model.names))

    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            # Count objects
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                # object_count[model.names[int(c)]] += int(n.item())

            # Draw boxes and labels
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    output_image = annotator.result()

    return output_image, object_count

def get_model_and_running_func(weights_path, data_path, img_size=640):
    device = select_device()
    
    # Load the model
    model = DetectMultiBackend(weights_path, device=device, dnn=False, data=data_path, fp16=False)
    
    # Check and set the image size
    img_size = check_img_size(img_size, s=model.stride)
    
    # Warm up the model
    model.warmup(imgsz=(1, 3, img_size, img_size))
    
    return model, detector


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self, function_run=None, model=None, threshold=0.20, iou_threshold=0.40):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        output_image, object_count = function_run(model, image, conf_thres = threshold, iou_thres=iou_threshold)
        ret, outputImagetoReturn = cv2.imencode('.jpg', output_image)
        return outputImagetoReturn.tobytes(), output_image
    
    def get_frame_without_detection(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        outputImage = outputs
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage)
        return outputImagetoReturn.tobytes(), outputImage


def generate_frames(camera, AI):
    try:
        while True:
            if AI:
                threshold = float(settings.CONF_THRESHOLD)
                print("model video threshold: " + str(threshold))
                model, function = get_model_and_running_func(weights_path=str(os.path.join(settings.WEIGHTS_DIR, 'yolov5s.pt')), data_path=str(os.path.join(settings.DATA_DIR, 'coco.yaml')))
                frame, img = camera.get_frame_with_detection(function, model, threshold, iou_threshold=float(settings.IOU_THRESHOLD))
            if not AI:
                frame, img = camera.get_frame_without_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
