from PIL import Image
import cv2
import numpy as np

import core.utils as utils
import datetime
import CONFIG 

from collections import deque
import json
import time
import math
import logging
import sys

class TrafficViolation:
  def __init__(self):
    self.color = CONFIG.color
    self.logging_format = "%(camera_id)s : %(class_name)s-%(track_id)s %(message)s %(time)s"
    logging.basicConfig(handlers=[logging.FileHandler(CONFIG.log_file),logging.StreamHandler(sys.stdout)], format=self.logging_format, level=logging.INFO)
    self.logger = logging.getLogger("violations_logger")   
  
  def non_max_suppression_fast(self, boxes, iou_threshold=0.33):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
      return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)
      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])
      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)
      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]
      # delete all indexes from the index list that have
      idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > iou_threshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

  def ccw(self, A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

  def intersect(self, A, B, C, D):
    return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

  def vector_angle(self, midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))

  def check_violation(self, original_frame, track_vehicles, roi_line, midpoint, previous_midpoint, track_id, signal_status):
    intersect_info = []
    origin_midpoint = (midpoint[0], original_frame.shape[0] - midpoint[1])
    origin_previous_midpoint = (previous_midpoint[0], original_frame.shape[0] - previous_midpoint[1])
    if signal_status == "red" and self.intersect(midpoint, previous_midpoint, roi_line[0], roi_line[1]) and track_id not in track_vehicles.already_counted:
      track_vehicles.already_counted.append(track_id)  # Set already counted for ID to true.
      angle = self.vector_angle(origin_midpoint, origin_previous_midpoint)
      intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
      intersect_info.extend([track_id, intersection_time])
      if abs(angle) > 0:
        return intersect_info

  def bbox_nms_format(self, bboxes):
    updates_bboxes = []
    x1, y1, x2, y2, score, classes = [],[],[],[],[],[]
    for box in bboxes:
      x1.append(float(box[0]))
      y1.append(float(box[1]))
      x2.append(float(box[2]))
      y2.append(float(box[3]))
      score.append(box[4])
      classes.append(box[5]) 
    updates_bboxes.extend([x1, y1, x2, y2, score, classes]) 
    updates_bboxes = np.transpose(updates_bboxes)
    return updates_bboxes
  
  def raise_alart(self, violation_info):
    self.logger.info('Red Light traffic violation at: ', extra=violation_info)
    
  def track_violations(self, original_frame, camera_id, track_vehicles, roi_line, signal_status):
    tracked_bboxes = []
    for track in track_vehicles.tracked_vehicles(): 
      if not track.is_confirmed() or track.time_since_update > 5:
        continue 
      bbox = track.to_tlbr() # Get the corrected/predicted bounding box
      class_name = track.get_class() #Get the class name of particular object
      tracking_id = track.track_id # Get the ID for the particular track
      index = track_vehicles.key_list[track_vehicles.val_list.index(class_name)] # Get predicted object index by object name
      tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
      midpoint = track.tlbr_midpoint(bbox)
      origin_midpoint = (midpoint[0], original_frame.shape[0] - midpoint[1])  # get midpoint respective to botton-left
      if track.track_id not in track_vehicles.memory:
        track_vehicles.memory[track.track_id] = deque(maxlen=2)
      track_vehicles.memory[track.track_id].append(midpoint)
      previous_midpoint = track_vehicles.memory[track.track_id][0]
      origin_previous_midpoint = (previous_midpoint[0], original_frame.shape[0] - previous_midpoint[1])
      violation_vehicles = self.check_violation(original_frame, track_vehicles, roi_line, midpoint, previous_midpoint, tracking_id, signal_status)
      if violation_vehicles is not None:
        violation_info = {"camera_id":camera_id, "class_name":class_name, "track_id":violation_vehicles[0], "time":violation_vehicles[1].strftime("%d-%m-%Y %H:%M:%S")}
        self.raise_alart(violation_info)
    return tracked_bboxes
    
    

