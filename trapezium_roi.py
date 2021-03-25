# imports
from os import remove
import cv2
import numpy as np
import json 
import sys

from numpy.lib.polynomial import poly

# default to hubli cam_1
img = ""


if len(sys.argv) < 2:
    raise Exception("No road specified.")

road_name = sys.argv[1]

if len(sys.argv) < 3:
    set_val = 0
else:
    set_val = sys.argv[2]


# class for selecting a point
class CoordinateStore:
    def __init__(self):
        self.points = []
        self.click_counter = 0

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),3,(0,0,0),-1)
            self.click_counter += 1
            self.points.append((x,y))

# Write cropped polygons to file for later use/loading
def save_roi(road_name, polygons):
    with open('config.json', 'r+') as f:
        data = json.load(f)
        data[road_name]['polygon'] = polygons

        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    print('Saved polygons to settings.json!')

def get_roi(road_name):
    with open('config.json', 'r+') as f:
        data = json.load(f)
        polygons = data[road_name]['polygon']
        return polygons

def set_roi(road_name, road_img):
    global img
    coordinateStore1 = CoordinateStore()
    img = road_img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',coordinateStore1.select_point)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if coordinateStore1.click_counter >=4:
            break
    cv2.destroyAllWindows()

    polygons = list()
    print("Selected Coordinates:")
    for i in coordinateStore1.points:
        polygons.append(list(i))

    polygons[3], polygons[2] = polygons[2], polygons[3]

    save_roi(road_name, polygons)
    pts = np.array(polygons, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(road_img, [pts], True,(255,255,0), 3)
    cv2.imshow('Window', road_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_roi(road_name, road_img):
    polygons = get_roi(road_name)
    pts = np.array(polygons, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(road_img, [pts], True,(255,255,0), 3)
    cv2.imshow('Window', road_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(road_name, set_val):
    road_name = road_name
    road_img = cv2.imread(road_name + ".png")
    if set_val == 0:
        show_roi(road_name, road_img)
    else:
        set_roi(road_name, road_img)

if __name__ == '__main__':
    # road_name = "hubli_cam_1"
    main(road_name, set_val)