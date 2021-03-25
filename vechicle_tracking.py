import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from PIL import Image
import cv2
import numpy as np

import CONFIG
from CONFIG import FLAGS
import core.utils as utils
from core.config import cfg

import psutil
import time
import os
import multiprocessing
from collections import deque
import math 



from collections import deque
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from deep_sort import generate_detections as gdet


class TrackVehicles:
    '''
    
    '''
    def __init__(self):
      self.YOLO_Classes = utils.read_class_names(cfg.YOLO.CLASSES)
      self.key_list = list(self.YOLO_Classes.keys()) 
      self.val_list = list(self.YOLO_Classes.values())
      self.deepSORT_model = CONFIG.deepSORT_model
      self.encoder = gdet.create_box_encoder(self.deepSORT_model, batch_size=1)
      self.max_cosine_distance = 0.7
      self.nn_budget = None
      self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
      self.tracker = Tracker(self.metric)
      self.track_only = ["car","truck","motorbike","bus","bicycle", "person"]
      self.memory = {}
      self.already_counted = deque(maxlen=50)
      
    def format_detection_bbox(self, image, bboxes):
      boxes, scores, names = [], [], []
      for bbox in bboxes:
        if len(self.track_only) !=0 and self.YOLO_Classes[int(bbox[5])] in self.track_only or len(self.track_only) == 0:
          boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
          scores.append(bbox[4])
          names.append(self.YOLO_Classes[int(bbox[5])])
      # Obtain all the detections for the given frame.
      boxes = np.array(boxes) 
      names = np.array(names)
      scores = np.array(scores)
      features = np.array(self.encoder(image, boxes))
      detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
      return detections
    
    def update_tracker(self, detections):
      self.tracker.predict()
      self.tracker.update(detections)
       
    def tracked_vehicles(self):
      for track in self.tracker.tracks:
          yield track
          
    def clear_memory(self):
      if len(self.memory) > 100:
        del self.memory[list(self.memory)[0]]
        # self.check_violation(original_frame, roi_line, midpoint, previous_midpoint, track_id, signal_status)
      
      
      


