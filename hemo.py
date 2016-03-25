# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:13:26 2015

@author: Saiprasad
"""

# optical disc detection

from __future__ import division
import cv2
import numpy as np
import copy
import math


retina = cv2.imread(r"E:\train_sample\13_right.jpeg",1);
retina_blue , retina_green , retina_red = cv2.split(retina)
screen_res = 1366, 768
scale_width = screen_res[1] / retina_blue.shape[1]
scale_height = screen_res[0] / retina_blue.shape[0]
scale = min(scale_width, scale_height)
window_width = int(retina_blue.shape[1] * scale)
window_height = int(retina_blue.shape[0] * scale)
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', window_width, window_height)

#cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('output1', window_width, window_height)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#preprocessing of the images
retina_green[retina_green<20] = 255;
retina_green = cv2.bitwise_not(retina_green);
retina_green = clahe.apply(retina_green);
retina_green = cv2.equalizeHist(retina_green)
ret, retina_green = cv2.threshold(retina_green,retina_green.max()-10,255,cv2.THRESH_BINARY);
exudate = copy.copy(retina_green);
#cv2.imshow('output',dilation);
image,contours,hierarchy = cv2.findContours(retina_green, 1, 2);
mask = np.ones(retina_green.shape[:2], dtype="uint8") * 255
# removing false exudates
for items in contours:
        #print retina_g[items]
        area = cv2.contourArea(items)
        perimeter = cv2.arcLength(items,True)
        if perimeter != 0:
                 R= 4*np.pi*area/math.pow(perimeter,2);
                 if (R<0.2) or  (area<200)  :
                         cv2.drawContours(mask, [items], -1, 0, -1)
exudate = cv2.bitwise_and(exudate,exudate, mask=mask);
exudate = cv2.medianBlur(exudate,5);
cv2.imshow('output',exudate);
cv2.waitKey(0)
cv2.destroyAllWindows()
