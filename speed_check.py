# import the files
import cv2
import dlib
import time
import threading
import math

# initialize the model and the video
carCascade = cv2.CascadeClassifier('myhaar.xml')
# video = cv2.VideoCapture('rtsp://admin:admin@123@112.133.197.90:2554/cam/realmonitor?channel=1&subtype=1')
video = cv2.VideoCapture('speed_test.mp4')


# set width and height of the video
WIDTH = 704
HEIGHT = 576

# function to estimate the speed
def estimateSpeed(location1, location2):
    # we get the values of location 1 and 2
    # we calculate the difference in pixels in a frame through distance formula
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    # we calculate distance per frame in meters by dividing pixels by pixel per meter
    ppm = 8.8
    # ppm = 14.28
    d_meters = d_pixels / ppm
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 18
    # we determine the speed by distance for a frame with frame per second and to convert it to kmph we multiply 3.6
    speed = d_meters * fps * 3.6
    return speed
    
# function to track multiple objects
def trackMultipleObjects():
    # initialize the rectange color and set the frameCounter, fps and currentCarId to 0
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    
    # initialize dicts for carTracker, carNumbers, carLocation1 and carLocation2
    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    # initialize the speed to be multiplied by 1000
    speed = [None] * 1000
    
    # Write output to video file 
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))

    # while the loop doesn't break
    while True:
        # intialize the start_time and start reading the video frames (break if frame not image)
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break
        
        # resize the image to width and height
        image = cv2.resize(image, (WIDTH, HEIGHT))
        # copy the image to resultImage
        resultImage = image.copy()
        
        # increment frameCounter by 1
        frameCounter = frameCounter + 1
        
        # initialize car delete list
        carIDtoDelete = []

        # for each id in car tracker image we get the tracking quality if that id has quality less then 7 we add it to be deleted
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
                
        # delete the car ids from carIDtoDelete
        for carID in carIDtoDelete:
            print ('Removing carID ' + str(carID) + ' from list of trackers.')
            print ('Removing carID ' + str(carID) + ' previous location.')
            print ('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
        
        # we process each 10th frame
        if not (frameCounter % 10):
            # convert the car image to gray scale for it to be detected by the carCascade detector
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get the detected x,y,w,h values from the carCascade model
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            # for each detected cars do the following
            # change x,y,w,h values to int
            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                # find the x_bar and y_bar
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                matchCarID = None
            
                # for each carID in carTracker get tracker position (if carTracker is empty create new matchCarID)
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                
                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID
                
                # create a new tracker for the car
                if matchCarID is None:
                    print ('Creating new tracker ' + str(currentCarID))
                    # initialize the correlation tracker from dlib and start tracking for the current car coordinates in the image
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    
                    # initialize the carTracker dict with current cars tracker and initialize location 1
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    # increment carID
                    currentCarID = currentCarID + 1
        
        #cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

        # for carID in car tracker get the tracker positions for the rest of the 9 frames
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
                    
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            
            # we plot the rectangle
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            
            # and initialize the car location 2 for each cars
            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]
        
        # we calculate the end time
        end_time = time.time()
        
        # if start time and end time are not same we initialize the frame per second we are getting
        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)
        
        #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # we iterate through each car location value
        for i in carLocation1.keys():	
            # Then we iterate through each car location
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                # initialize car location1 with carlocation2 values
                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                carLocation1[i] = [x2, y2, w2, h2]

                # if the values of two locations don't match and the vehicle is between 275 to 285 calculate the speed
                # print 'new previous location: ' + str(carLocation1[i])
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0):
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    # if speed is initialized for the car and its greater then 180 pixels show the speed on the image
                    #if y1 > 275 and y1 < 285:
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, ), 2)
                    
                    #print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                    #else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        #print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
        # show the result frame
        cv2.imshow('result', resultImage)
        # Write the frame into the file 'output.avi'
        #out.write(resultImage)

        # stop the video
        if cv2.waitKey(33) == 27:
            break
    
    cv2.destroyAllWindows()

# the program  starts executing here
if __name__ == '__main__':
    trackMultipleObjects()
