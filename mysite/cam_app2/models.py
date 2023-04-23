from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.core.files.storage import default_storage

from pathlib import Path


import os, uuid, glob, cv2
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

str_uuid = uuid.uuid4()  # The UUID for image uploading

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



def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/*.*')), recursive=True)
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

# Create your models here.
class ImagePage(Page):
    """Image Page."""

    template = "cam_app2/image.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]

    

    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"]= []
        context["my_result_file_names"]=[]
        context["my_staticSet_names"]= []
        context["my_lines"]= []
        return context

    def serve(self, request):
        model, function_run = get_model_and_running_func(weights_path=str(os.path.join(settings.WEIGHTS_DIR, 'yolov5s.pt')), data_path=str(os.path.join(settings.DATA_DIR, 'coco.yaml')))
        emptyButtonFlag = False
        if request.POST.get('start')=="":
            context = self.reset_context(request)
            print(request.POST.get('start'))
            print("Start selected")
            fileroot = os.path.join(settings.MEDIA_ROOT, 'uploadedPics')
            res_f_root = os.path.join(settings.MEDIA_ROOT, 'Result')
            with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'r') as f:
                image_files = f.readlines()
            if len(image_files)>=0:
                for file in image_files:
                    ###
                    # For each file, read, pass to model, do something, save it #
                    ###
                    filename = file.split('/')[-1]
                    filepath = os.path.join(fileroot, filename)
                    img = cv2.imread(filepath.strip())
                    threshold = float(settings.CONF_THRESHOLD)
                    iou_threshold=float(settings.IOU_THRESHOLD)
                    output_image, object_count = function_run(model, img, conf_thres = threshold, iou_thres=iou_threshold)
                    fn = filename.split('.')[:-1][0]
                    r_filename = f'result_{fn}.jpeg'
                    print(r_filename)
                    cv2.imwrite(str(os.path.join(res_f_root, r_filename)), output_image)
                    r_media_filepath = Path(f"{settings.MEDIA_URL}Result/{r_filename}")
                    print(r_media_filepath)
                    with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                        f.write(str(r_media_filepath))
                        f.write("\n")
                    context["my_uploaded_file_names"].append(str(f'{str(file)}'))
                    context["my_result_file_names"].append(str(f'{str(r_media_filepath)}'))
            return render(request, "cam_app2/image.html", context)

        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
            context = self.reset_context(request)
            context["my_uploaded_file_names"] = []
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploadedPics/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploadedPics/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")

                context["my_uploaded_file_names"].append(str(f'{str(filename)}'))
            return render(request, "cam_app2/image.html", context)
        context = self.reset_context(request)
        reset()
        return render(request, "cam_app2/image.html", {'page': self})
