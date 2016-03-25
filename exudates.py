# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 20:43:45 2015

@author: Saiprasad
"""


import cv2
import numpy as np
import copy
import math

retina = cv2.imread(r"E:\train\train\16_left.jpeg",1);
retina_blue , retina_green , retina_red = cv2.split(retina)
screen_res = 1280, 720
scale_width = screen_res[1] / retina_blue.shape[1]
scale_height = screen_res[0] / retina_blue.shape[0]
scale = min(scale_width, scale_height)
window_width = int(retina_blue.shape[1] * scale)
window_height = int(retina_blue.shape[0] * scale)

cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output1', window_width, window_height)
'''
cv2.namedWindow('output2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output2', window_width, window_height)
'''
kernel = np.ones((50,50),np.uint8);
retina_g = cv2.equalizeHist(retina_green)
a =  retina_green.max();
new = cv2.morphologyEx(retina_g, cv2.MORPH_CLOSE, kernel);
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#preprocessing of the images
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
kernel = np.ones((25,25),np.uint8)
output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel) 
cont = cv2.bitwise_not(output)
#copying image before applying findcontour as findcontour modifies the image
image,contours,hierarchy = cv2.findContours(output, 1, 2)
leftmost = retina_blue.shape;
rightmost = [0,0];
topmost = retina_blue.shape;
bottommost = [0,0];
for item in contours:
        area = cv2.contourArea(item)
        perimeter = cv2.arcLength(item,True);
        if(perimeter !=0):
            R= 4*np.pi*area/math.pow(perimeter,2);
        if (R>0.3) & (area>10000):
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
centre = [0,0];
centre[1] = int((topmost[1]-10 + bottommost[1]-10)/2)
centre[0] = int((leftmost[0]-10 + rightmost[0]-10)/2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
radius = math.sqrt(math.pow(topmost[1]-10-centre[1],2) + math.pow(leftmost[0]-10-centre[0],2));
radius = int(radius);
#if (radius<250):
#   radius = 250;
#elif (radius>400):
#    radius = 400;
centre = tuple(centre);
#retina_g = clahe.apply(retina_green)
cv2.circle(retina_green,centre, int(radius)+20, (0,0,0), -1);
b = retina_green.max();
print a,b
cv2.imshow('output1',retina_green);
cv2.waitKey(0)
cv2.destroyAllWindows()

