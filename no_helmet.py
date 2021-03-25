# imports 
from PIL import Image
import cv2
import numpy as np
from math import *
from matplotlib import path
from tensorflow.python.framework.ops import prepend_name_scope
from tensorflow.python.keras.backend import one_hot

from vechicle_detection import DetectVehicles
from vechicle_tracking import TrackVehicles
from vehicle_violation import TrafficViolation
import core.utils as utils
import CONFIG
from core.config import cfg
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer

from collections import deque
import json
import time
import math
import datetime
import argparse


def create_model():
    network = Sequential()
    network.add(Conv2D(filters=32, kernel_size= (3,3), input_shape = (90, 90, 1), activation='relu'))
    network.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu'))
    network.add(Dense(2, activation='softmax'))
    network.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return network

def show_roi(road_name):
    polygons = get_roi(road_name)
    pts = np.array(polygons, np.int32)
    pts = pts.reshape((-1,1,2))
    return pts, polygons

def get_roi(road_name):
    with open('config.json', 'r+') as f:
        data = json.load(f)
        polygons = data[road_name]['polygon']
        return polygons

# get distance between two centroids
def great_circle_distance(coordinates1, coordinates2):
  latitude1, longitude1 = coordinates1
  print(latitude1)
  print(longitude1)
  latitude2, longitude2 = coordinates2
  print(latitude2)
  print(longitude2)
  d = pi / 180  # factor to convert degrees to radians
  return acos(int(sin(longitude1*d) * sin(longitude2*d) +
              cos(longitude1*d) * cos(longitude2*d) *
              cos((latitude1 - latitude2) * d))) / d

# check if two centroids are in the range
def in_range(coordinates1, coordinates2, range1):
    k = great_circle_distance(coordinates1, coordinates2)
    print(k)
    return int(k) <= range1

# main function
if __name__ == '__main__': 
    # initialise objects
    detect_vehicles = DetectVehicles()
    vehicle_violations = TrafficViolation()
    # model = create_model()
    helmet_list = np.array([["Has helmet"], ["No Helmet"]])
    network = load_model('model_data/shallownet.hdf5')
    one_hot = LabelBinarizer()



    YOLO_Classes = utils.read_class_names(cfg.YOLO.CLASSES)
    track_only = ["car","truck","motorbike","bus","bicycle"]
    track_vehicles = TrackVehicles()
    
    speed = [None] * 1000
    
    # get the live feed from the server
    parser = argparse.ArgumentParser(description='parse the image locations')
    parser.add_argument('-video_path',
                    metavar='s',
                    type=str,
                    default='rtsp://admin:admin@123@112.133.197.90:2554/cam/realmonitor?channel=1&subtype=1',
                    help='the rtsp video source')

    parser.add_argument('-camera_id',
                    metavar='s',
                    type=str,
                    default='5554',
                    help='the rtsp video source')

    # parse the args
    args = parser.parse_args()
    video_path = args.video_path
    # video_path = "np_test.mp4"
    
    vehicle_location1 = dict()
    vehicle_location2 = dict()
    camera_id = args.camera_id
    
    # Capture the video frames
    # vid = cv2.VideoCapture(video_path)
    vid = cv2.VideoCapture('Test2.mp4')
    # vid = cv2.VideoCapture('front_vid2.mp4')


    # assign video dimensions
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(width)
    assert (width==int(vid.get(3))), "Not matching - width"
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(height)
    assert (height==int(vid.get(4))), "Not matching - height"
    # result = cv2.VideoWriter('no_parking.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (width,height)) 
    out = cv2.VideoWriter('helmet_detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
    frame_skip = 0
    # checking frames on video
    frame_counter = 0
    np_count = 0
    vehicle_in_no_parking = list()
    start_time = datetime.datetime.now()
    v_np_tracker = dict()
    
    while vid.isOpened():
        frame_skip += 1
        # skip to every 10th frame
        if frame_skip % 10 != 0:
            frame_counter += 1
            continue
        # get the frame
        ret , original_frame = vid.read()
        if not ret:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
        # change the frame to bytes
        img_buffer = cv2.imencode('.jpg', original_frame)[1].tobytes()
        image_bytes = {}
        image_bytes["ImageBytes"] = img_buffer
        image_bytes["ContentType"]= 'image/jpg'
        image_bytes["CameraId"]= camera_id
        
        # load signal_status
        with open("signal_status.json","r") as json_file:
            signal_json = json.load(json_file)  
        signal_status = signal_json['color']
        
        # detect the bounding box and 
        original_frame, bboxes, classes = detect_vehicles.get_bbox(image_bytes)
        updated_bboxes = vehicle_violations.bbox_nms_format(bboxes)
        bboxes = bboxes if np.asarray(bboxes).size == 0 else vehicle_violations.non_max_suppression_fast(updated_bboxes)
        # print(len(bboxes))
        # print(len(classes[0]))
        # track the vehicle using bounding boxes
        detections = track_vehicles.format_detection_bbox(original_frame, bboxes)

        track_vehicles.update_tracker(detections)
        tracked_bboxes = []
        for track in track_vehicles.tracked_vehicles():
            # print("track_ids")
            # print(track.track_id)
            # collect the bounding box, trackid and class index

            # print(track.get_class()) 
            if track.get_class() == "motorbike" or track.get_class() == "bicycle":
                index = track_vehicles.key_list[track_vehicles.val_list.index(track.get_class())]
                tracked_bboxes.append(track.to_tlbr().tolist() + [track.track_id, index])
                [x, y, x_bar, y_bar] = track.to_tlbr()
                y_min = int(y)-(1.5 * int((y_bar-y)/2))
                y_min = y_min if y_min >= 0 else y
                y_min = y_min if y_min <= height else y
                y_max = int(y_bar)-(1.5*int((y_bar-y)/2))
                y_max = y_max if y_max >= 0 else y_bar
                y_max = y_max if y_max <= height else y_bar
                centroid = ((x+x_bar)/2, (y+y_bar)/2)
                print("centroid")
                print(centroid)
                c1 = (int(x), int(y_min))
                c2 = (int(x_bar), int(y_bar))
                print(c1)
                print(c2)
                crop_img = original_frame[int(y_min): int(y_bar), int(x): int(x_bar)]
                # crop_img = original_frame[int(x): int(x_bar), int(y): int(y_bar)]
                print(crop_img)
                if len(crop_img) != 0:
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    crop_img = cv2.resize(crop_img, (90, 90))
                    crop_img = crop_img.reshape(1, 90, 90,  1)
                    predicted_target = network.predict(crop_img)
                    if int(predicted_target[0][0]) == 0:
                        cv2.putText(original_frame, "No Helmet " + str(track.track_id), (int(x_bar), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                        # cv2.rectangle(original_frame, c1, c2, (255, 0, 0), 2) 
                    print(int(predicted_target[0][0]))
                cv2.rectangle(original_frame, c1, c2, (0, 0, 0), 2)      
        
        # draw the track box and show the image
        original_frame = utils.draw_track_bbox(original_frame, bboxes, tracking=True)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) 
        
        track_vehicles.clear_memory()
        cv2.imshow("Frame",original_frame)
        out.write(original_frame) 
        cv2.waitKey(1)