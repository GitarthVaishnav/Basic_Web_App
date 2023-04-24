import os
import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps
from django.conf import settings
import tensorflow as tf

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self, function_run=None, model=None):
        success, image = self.video.read()
        output_image = function_run(model, image)
        ret, outputImagetoReturn = cv2.imencode('.jpg', output_image)
        return outputImagetoReturn.tobytes(), output_image
    
    def get_frame_without_detection(self):
        success, image = self.video.read()
        ret, outputImagetoReturn = cv2.imencode('.jpg', image)
        return outputImagetoReturn.tobytes(), image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def load_tf_model():
    # Load your custom TensorFlow model here and return it
    model = tf.keras.models.load_model(str(os.path.join(settings.WEIGHTS_DIR, 'my_model.h5')))
    return model

def preprocess_image(image, target_size=(32, 32)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def tf_detector(model, image):
    # Implement your custom TensorFlow model inference here
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)

    # Get the class index with the highest probability
    class_idx = np.argmax(predictions)
    # print(class_idx)

    # Get the class name using the class index
    class_name = class_names[class_idx]
    # print(class_name)

    # Draw the class name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 4
    text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
    text_x = image.shape[1] - text_size[0] - 100  # 10 pixels from the right edge
    text_y = 300  # 30 pixels from the top edge
    cv2.putText(image, class_name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return image


def get_model_and_running_func():
    model = load_tf_model()
    return model, tf_detector

def generate_frames(camera, AI):
    try:
        while True:
            if AI:
                model, function = get_model_and_running_func()
                frame, img = camera.get_frame_with_detection(function, model)
            else:
                frame, img = camera.get_frame_without_detection()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
