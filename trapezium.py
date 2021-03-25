#%%
import numpy as np
import cv2
import json
from math import *

from numpy.lib.function_base import append
#%%


my_img = "hubli_cam_1.png"
my_img = cv2.imread(my_img)
pts = np.array([[157,250],[277,259],[256,371],[156,371]], np.int32)
pts = pts.reshape((-1,1,2))
pts
#%%
# parameters : image, pts, npts, ncontours, isClosed, color, thickness
cv2.polylines(my_img,[pts],True,(0,0,0), 5)
cv2.imshow('Window', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%


class CoordinateStore:
    def __init__(self):
        self.points = []
        self.click_counter = 0

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),3,(255,0,0),-1)
            self.click_counter += 1
            self.points.append((x,y))


#instantiate class
coordinateStore1 = CoordinateStore()


# Create a black image, a window and bind the function to window
img = my_img
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
#%%
polygon_x = int((polygons[0][0] + polygons[1][0])/2)
polygon_y = int((polygons[0][1] + polygons[1][1])/2)
polygon_x_bar = int((polygons[2][0] + polygons[3][0])/2)
polygon_y_bar = int((polygons[2][1] + polygons[3][1])/2)
[polygon_x, polygon_y, polygon_x_bar, polygon_y_bar]
#%%
my_img = "hubli_cam_1.png"
my_img = cv2.imread(my_img)
pts = np.array(polygons, np.int32)
pts = pts.reshape((-1,1,2))
pts
#%%

cv2.polylines(my_img, [pts], True,(255,0,0), 4)
cv2.imshow('Window', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%


def save_roi(road_name, polygons):
	global ref_rects

	with open('config.json', 'r+') as f:
		data = json.load(f)
		data[road_name]['polygon'] = polygons

		f.seek(0)
		json.dump(data, f, indent=4)
		f.truncate()

	print('Saved ref_rects to settings.json!')
#%%
road_name = "hubli_cam_1"


save_roi(road_name, polygons)

# %%
x1 = 5
x2 = 10
y1 = 10
y2 = 10
x, y = int((x1+x2)/2), int((y1+y2)/2)

# %%

def great_circle_distance(coordinates1, coordinates2):
  latitude1, longitude1 = coordinates1
  latitude2, longitude2 = coordinates2
  d = pi / 180  # factor to convert degrees to radians
  return acos(sin(longitude1*d) * sin(longitude2*d) +
              cos(longitude1*d) * cos(longitude2*d) *
              cos((latitude1 - latitude2) * d)) / d

def in_range(coordinates1, coordinates2, range1):
    k = great_circle_distance(coordinates1, coordinates2)
    print(k)
    return int(k) <= range1

#%%
in_range((290,436), (290,436), 3)
#%%
d = pi/ 180
a = sin(436*d)
b = sin(436*d)
c = cos(436*d)
d = cos(436*d)
e = cos((290-290) * d)
# [a,b,c,d, e]
k = (a*b) + ((c*d) * e)
k
acos(int(k))

#%%
xx = 4
yy = 5
y1 = 4
x1 = 3
y2 = 7
x2 = 6
#%%

polygon_xy1 = (int(polygons[0][0]), int(polygons[0][1]))
polygon_xy2 = (int(polygons[1][0]), int(polygons[1][1]))
polygon_xy1_bar = (int(polygons[2][0]), int(polygons[2][1]))
polygon_xy2_bar = (int(polygons[3][0]), int(polygons[3][1]))
#%%
from matplotlib import path
polygon_xy1 = (265, 368)
polygon_xy2 = (333, 366)
polygon_xy1_bar = (287, 239)
polygon_xy2_bar = (256, 277)
#%%
p = path.Path([polygon_xy1, polygon_xy2, polygon_xy1_bar, polygon_xy2_bar])
p.contains_points([(400, 300)])[0]
# %%
