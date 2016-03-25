# optical disc detection

import cv2
import numpy as np
import copy

retina = cv2.imread(r"E:\acads\internships\Syntel\sample\16_left.jpeg",1)
retina_blue , retina_green , retina_red = cv2.split(retina)

screen_res = 1280, 720
scale_width = screen_res[1] / retina_blue.shape[1]
scale_height = screen_res[0] / retina_blue.shape[0]
scale = min(scale_width, scale_height)
window_width = int(retina_blue.shape[1] * scale)
window_height = int(retina_blue.shape[0] * scale)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', window_width, window_height)

#cv2.namedWindow('true', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('true', window_width, window_height)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(155,155))
#erosion = cv2.erode(retina_green,kernel,iterations = 2)
tophat = cv2.morphologyEx(retina_green, cv2.MORPH_TOPHAT, kernel)
tophat = cv2.medianBlur(tophat,25)
#tophat = cv2.medianBlur(tophat,5)


#cv2.imshow('true',retina_green)
#cv2.imshow('erosion',tophat)

'''
cv2.namedWindow('red', cv2.WINDOW_NORMAL)
cv2.resizeWindow('red', window_width, window_height)

cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
cv2.resizeWindow('blue', window_width, window_height)
'''

#preprocessing of the images
retina_g = cv2.equalizeHist(retina_green)
retina_b = cv2.equalizeHist(retina_blue)
retina_r = cv2.equalizeHist(retina_red)

#detection of optical disc
#finding the brightest spots in red and green channels 
ret, green = cv2.threshold(retina_g,retina_g.max()-5,255,cv2.THRESH_BINARY)
ret, red = cv2.threshold(retina_r,retina_r.max()-5,255,cv2.THRESH_BINARY)

# median filtering to remove the unwanted bright spots
red = cv2.medianBlur(red,5)
green = cv2.medianBlur(green,5)

# usually the bright spot is yellow coloured so it should be bright in red and green channels
# so taking bitwise and to remove the unwanted spots from both the channels
output = cv2.bitwise_and(red,green)

# median filtering to remove the residual (unwanted) bright spots; 
output = cv2.medianBlur(output,65)
cont = cv2.bitwise_not(output)
#copying image befor applying findcontour as findcontour modifies the image 
images, contours,hierarchy = cv2.findContours(output, 1, 2)

# locating the extreme points in the image to mask out the optical disc
leftmost = tuple(retina_blue.shape);
rightmost = tuple([0,0]);
topmost = tuple(retina_blue.shape);
bottommost = tuple([0,0]);
for item in contours:
        ##print "item :", item
        area = cv2.contourArea(item)
        new_left = tuple(item[item[:,:,0].argmin()][0])
        new_right = tuple(item[item[:,:,0].argmax()][0])
        new_top = tuple(item[item[:,:,1].argmin()][0])
        new_bottom = tuple(item[item[:,:,1].argmax()][0])

        if new_left[0] <= leftmost[0] :
                leftmost = new_left
        if new_right[0] >= rightmost[0] :
                rightmost = new_right
        if new_top[1] <= topmost[1] :
                topmost = new_top
        if new_bottom[1] >= bottommost[1] :
                bottommost = new_bottom
# mask to remove optical disc
# images after optical disc removal
#tophat[topmost[1]-200:bottommost[1]+200 , (leftmost[0]-200):(rightmost[0]+200)] = black_spot;
centre = map(sum, zip(topmost,bottommost,leftmost,rightmost))
centre[0] = int(centre[0]/4);
centre[1] = int(centre[1]/4);
centre = np.array(centre)

# the largest point from the centre
dist_top = np.linalg.norm(centre-np.array(topmost))
dist_bottom = np.linalg.norm(centre-np.array(bottommost))
dist_left = np.linalg.norm(centre-np.array(leftmost))
dist_right = np.linalg.norm(centre-np.array(rightmost))
distance = np.array([dist_top,dist_bottom,dist_left,dist_right])
radius = np.max(distance)
centre = tuple(centre);
tophat = cv2.circle(tophat,centre, int(radius)+100, (0,0,0), -1)
retina_red =   cv2.circle(retina_red,centre, int(radius)+100, (127,127,127), -1)
ret, output_green = cv2.threshold(tophat,tophat.max()-30,255,cv2.THRESH_BINARY);
#ret, output_red = cv2.threshold(retina_r,retina_r.max()-60,255,cv2.THRESH_BINARY);
#output_gr   = cv2.bitwise_and(output_green,output_red);
#cont2 = cv2.bitwise_not(output_gr);
#images, contours,hierarchy = cv2.findContours(output_gr, 1, 2)

'''
for item in contours:
                if cv2.contourArea(item)>100:
                        print cv2.contourArea(item);
'''


cv2.imshow("output",tophat)

cv2.waitKey(0)
cv2.destroyAllWindows()
