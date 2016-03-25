'''
optical disc detection using intensity thresholding
'''

import cv2
import numpy as np
import copy
import math

retina = cv2.imread(r"C:\Users\Skotur\Desktop\sample\13_right.jpeg",1)
cv2.normalize(retina,retina,0,255,cv2.NORM_MINMAX)
retina = cv2.resize(retina, (0,0), fx=0.5, fy=0.5) 
retina_blue , retina_green , retina_red = cv2.split(retina)
print retina_blue.shape
screen_res = 1280, 720
scale_width = screen_res[1] / retina_blue.shape[1]
scale_height = screen_res[0] / retina_blue.shape[0]
scale = min(scale_width, scale_height)
window_width = int(retina_blue.shape[1] * scale)
window_height = int(retina_blue.shape[0] * scale)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', window_width, window_height)

cv2.namedWindow('output_mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output_mask', window_width, window_height)

cv2.imshow('output_mask',retina);

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
tophat = cv2.morphologyEx(retina_green, cv2.MORPH_TOPHAT, kernel)

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
output = cv2.medianBlur(output,33)
cont = cv2.bitwise_not(output)
cv2.imshow('output_mask',output)

#copying image befor applying findcontour as findcontour modifies the image 
contours,hierarchy = cv2.findContours(output, 1, 2)

# locating the extreme points in the image to mask out the optical disc
leftmost = tuple(retina_blue.shape);
rightmost = tuple([0,0]);
topmost = tuple(retina_blue.shape);
bottommost = tuple([0,0]);
for item in contours:
        new_left = tuple(item[item[:,:,0].argmin()][0])
        new_right = tuple(item[item[:,:,0].argmax()][0])
        new_top = tuple(item[item[:,:,1].argmin()][0])
        new_bottom = tuple(item[item[:,:,1].argmax()][0])
        print new_bottom
        if new_left[0] <= leftmost[0] :
                leftmost = new_left
        if new_right[0] >= rightmost[0] :
                rightmost = new_right
        if new_top[1] <= topmost[1] :
                topmost = new_top
        if new_bottom[1] >= bottommost[1] :
                bottommost = new_bottom
#mask to remove optical disc
#images after optical disc removal
centre = map(sum, zip(topmost,bottommost,leftmost,rightmost))
centre[0] = int(centre[0]/4);
centre[1] = int(centre[1]/4);
centre = np.array(centre)

# the farthest point from the centre
dist_top = np.linalg.norm(centre-np.array(topmost))
dist_bottom = np.linalg.norm(centre-np.array(bottommost))
dist_left = np.linalg.norm(centre-np.array(leftmost))
dist_right = np.linalg.norm(centre-np.array(rightmost))
distance = np.array([dist_top,dist_bottom,dist_left,dist_right])
radius = np.max(distance)
centre = tuple(centre);
print radius
print centre
#copying image as cv2.circle modifies the image
top1 = copy.copy(retina_g);
top2 = copy.copy(tophat);

cv2.circle(retina_g,centre, int(radius)+100, (0,0,0), -1);
cv2.circle(retina_green,centre, int(radius)+100, (0,0,0), -1);


# exudates detection
# thresholding to get the region of interest
ret, output_green = cv2.threshold(tophat,tophat.max()-80,255,cv2.THRESH_BINARY);
exudate = copy.copy(output_green);
contours,hierarchy = cv2.findContours(output_green, 1, 2)
mask = np.ones(retina_green.shape[:2], dtype="uint8") * 255

# removing false exudates
for items in contours:
        area = cv2.contourArea(items)
        perimeter = cv2.arcLength(items,True)
        if perimeter != 0:
                 R= 4*np.pi*area/math.pow(perimeter,2);
                 x,y,w,h = cv2.boundingRect(items)
                 aspect_ratio = float(w)/h
                 mu = cv2.moments(items)
                 bigSqrt = math.sqrt( ( mu['m20'] - mu['m02'] ) *  ( mu['m20'] - mu['m02'] )  + 4 * mu['m11'] * mu['m11']);
                 if (( mu['m20'] + mu['m02'] - bigSqrt ) !=0 ):
                                 e = ( mu['m20'] + mu['m02'] + bigSqrt ) / ( mu['m20'] + mu['m02'] - bigSqrt );
                 else:
                                 e=1;
                 if (R<0.5) or (aspect_ratio>1.8) or (e<0.7) or (area<20):
                         cv2.drawContours(mask, [items], -1, 0, -1)
exudate = cv2.bitwise_and(exudate,exudate, mask=mask);
exudate = cv2.medianBlur(exudate,5)
cv2.imshow('output',output_green);

#cv2.imshow('output',output_green)
cv2.waitKey(0)
cv2.destroyAllWindows()
