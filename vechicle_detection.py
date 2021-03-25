import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import CONFIG
from CONFIG import FLAGS
import psutil
import time
import os
import multiprocessing
from collections import namedtuple

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class DetectVehicles:
    # function to define ROI regions and TFLite model interpreter 
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=CONFIG.yolo_weights)

    # Function to run TFLIte object detection
    def get_bbox(self, image_bytes):

        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = CONFIG.image_size
        cameraId = image_bytes["CameraId"]
        nparr = np.frombuffer(image_bytes["ImageBytes"], np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # print("Image shape: ",original_image.shape)
        image_h, image_w, _ = original_image.shape
        image_data = cv2.resize(original_image, (input_size[0], input_size[1]))
        image_data = image_data / 255.


        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)


        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], images_data)
        self.interpreter.invoke()
        pred = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size[0], input_size[1]]))

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=CONFIG.iou,
            score_threshold=CONFIG.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detections = utils.bbox_details(original_image, pred_bbox)

        return original_image, detections, classes.numpy()

    

