# imports 
from PIL import Image
import cv2
import numpy as np
from math import *
from matplotlib import path

from vechicle_detection import DetectVehicles
from vechicle_tracking import TrackVehicles
import core.utils as utils
import CONFIG
from core.config import cfg

from collections import deque
import json
import time
import math
import datetime
import argparse



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
    vid = cv2.VideoCapture('np_test.mp4')

    # assign video dimensions
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(width)
    assert (width==int(vid.get(3))), "Not matching - width"
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(height)
    assert (height==int(vid.get(4))), "Not matching - height"
    # result = cv2.VideoWriter('no_parking.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (width,height)) 
    out = cv2.VideoWriter('no_parking_2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
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
        # track the vehicle using bounding boxes
        detections = track_vehicles.format_detection_bbox(original_frame, bboxes)
        track_vehicles.update_tracker(detections)
        pts, polygons = show_roi("hubli_cam_1")
        # pts, polygons = show_roi("wework")
        polygon_xy1 = (int(polygons[0][0]), int(polygons[0][1]))
        polygon_xy2 = (int(polygons[1][0]), int(polygons[1][1]))
        polygon_xy1_bar = (int(polygons[2][0]), int(polygons[2][1]))
        polygon_xy2_bar = (int(polygons[3][0]), int(polygons[3][1]))
        p = path.Path([polygon_xy1, polygon_xy2, polygon_xy1_bar, polygon_xy2_bar])
        cv2.polylines(original_frame, [pts], True,(255,255,255), 2)
        
        for track in track_vehicles.tracked_vehicles():
            print("track_ids")
            print(track.track_id)
            if track.track_id in v_np_tracker:
                v_np_tracker[track.track_id]['current_time'] = datetime.datetime.now()
            else:
                v_np_tracker[track.track_id] = dict()
                v_np_tracker[track.track_id]['start_time'] = datetime.datetime.now()
                v_np_tracker[track.track_id]["counter"] = 0
            if (frame_counter) % 20 == 0:
                [x1, y1, x1_bar, y1_bar] = track.to_tlbr()
                x, y = int((x1+x1_bar)/2), int((y1+y1_bar)/2)
                vehicle_location1[track.track_id] = [x,y]
            else:
                [x2, y2, x2_bar, y2_bar] = track.to_tlbr()
                x, y = int((x2+x2_bar)/2), int((y2+y2_bar)/2)
                vehicle_location2[track.track_id] = [x,y]
            if track.track_id in vehicle_location1 and track.track_id in vehicle_location2:
                [xx1, yy1] = vehicle_location1[track.track_id]
                [xx2, yy2] = vehicle_location2[track.track_id]
                
                # in_range([x1, y1], [x1, y1], 4)
                # if [x1, y1, x1_bar, y1_bar] == [x2, y2, x2_bar, y2_bar]:
                if in_range([xx1, yy1], [xx2, yy2], 20):
                    if p.contains_points([(xx2, yy2)])[0]:
                        print("inside no parking zone")
                        continue
                    else:
                        if track.track_id in v_np_tracker:
                            v_np_tracker[track.track_id]["counter"] += 1
                        if v_np_tracker[track.track_id]["counter"] > 10:
                            vehicle_in_no_parking.append(track.track_id)
                # else: 
                #     if track.track_id in vehicle_in_no_parking:
                #         vehicle_in_no_parking.remove(track.track_id)
            if track.track_id in v_np_tracker:
                if "current_time" in  v_np_tracker[track.track_id]:
                    tot_time = v_np_tracker[track.track_id]['current_time'] - v_np_tracker[track.track_id]['start_time']
                    tot_time = tot_time.seconds
                    # print(type(tot_time))
                    if (track.track_id in vehicle_in_no_parking) and  tot_time>=30:            
                        cv2.putText(original_frame, "NP " + str(track.track_id), (int(xx2), int(yy2)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                
                
        # draw the track box and show the image
        original_frame = utils.draw_track_bbox(original_frame, bboxes, tracking=True)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) 
        
        track_vehicles.clear_memory()
        cv2.imshow("Frame",original_frame)
        out.write(original_frame) 
        cv2.waitKey(1)