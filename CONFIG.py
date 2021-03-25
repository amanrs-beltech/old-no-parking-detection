import os
import tensorflow.compat.v1 as tf
current_path = os.path.dirname(os.path.realpath(__file__))

log_file = "violations.log"
framework = "tflite"
yolo_weights = './checkpoints/yolov3-416.tflite'
deepSORT_model = './model_data/mars-small128.pb'
image_size = (416,416)
model = "yolov3"
iou = 0.5
score = 0.33
image = "./data/girl.jpg"
output = "result.png"
color = {"green":(0,255,0),"red":(255,0,0)}
image_height, image_width = 576, 704
camera_rois = {
    '4554': {
        'y_min': int(0.10764*image_height),
        'x_min': int(0.7*image_width),
        'y_max': int(0.47*image_height),
        'x_max': int(1.0*image_width),
    },
    '554': {
        'y_min': int(0.502*image_height),
        'x_min': int(0.3935*image_width),
        'y_max': int(0.7344*image_height),
        'x_max': int(0.3935*image_width)
    },
    '5554_BRTS': {
        'y_min': int(0.1875*image_height),
        'x_min': int(0.298*image_width),
        'y_max': int(0.66*image_height),
        'x_max': int(0.54*image_width)
    },
    '5554_Non_BRTS': {
        'y_min': int(0.1844*image_height),
        'x_min': int(0.5809*image_width),
        'y_max': int(0.6806*image_height),
        'x_max': int(0.983*image_width)
    },
    '2553_Non_BRTS': {
        'y_min': int(0.0454*image_height),
        'x_min': int(0.3*image_width),
        'y_max': int(0.86*image_height),
        'x_max': int(0.83*image_width),
    },
    '2553_BRTS': {
        'y_min': int(0.122*image_height),
        'x_min': int(0.0*image_width),
        'y_max': int(0.833*image_height),
        'x_max': int(0.28*image_width)
    }
}

tf.app.flags.DEFINE_string('framework', framework, '(tf, tflite, trt')
tf.app.flags.DEFINE_string('weights', yolo_weights,
                    'path to weights file')
tf.app.flags.DEFINE_integer('size', image_size[0], 'resize images to')
tf.app.flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
tf.app.flags.DEFINE_string('model', model, 'yolov3 or yolov4')
tf.app.flags.DEFINE_string('image', image, 'path to input image')
tf.app.flags.DEFINE_string('output', output, 'path to output image')
tf.app.flags.DEFINE_float('iou', iou, 'iou threshold')
tf.app.flags.DEFINE_float('score', score, 'score threshold')
FLAGS = tf.app.flags.FLAGS