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
import numpy as np

from django.conf import settings

str_uuid = uuid.uuid4()  # The UUID for image uploading
import tensorflow as tf

# Load your custom TensorFlow model here
def load_tf_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image, target_size=(32, 32)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def tf_detector(model, image, class_names):
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)

    class_idx = np.argmax(predictions)
    class_name = class_names[class_idx]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 12
    font_thickness = 4
    text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
    text_x = image.shape[1] - text_size[0] - 100
    text_y = 300
    cv2.putText(image, class_name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    print(class_name)

    return image

def get_model_and_running_func(model_path):
    model = load_tf_model(model_path)
    return model, tf_detector

# Define your class names here
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']




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
        model, function_run = get_model_and_running_func(model_path=str(os.path.join(settings.WEIGHTS_DIR, 'my_model.h5')))
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
                    output_image = function_run(model, img, class_names)
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
