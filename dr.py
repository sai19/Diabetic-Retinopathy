# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:59:29 2015

@author: Saiprasad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 11:53:38 2015

@author: SKotur
"""
from __future__ import division
import cv2
import numpy as np
import copy
import math
import os
import csv
'''
This function gets the radius of the eye in image
'''
def GetRadius(image):
         retina = copy.copy(image);
         retina_blue , retina_green , retina_red = cv2.split(retina)
         # scaling the histogram linearly to normalize all the images
         retina_green = cv2.bitwise_not(retina_green);
         ret, retina_green = cv2.threshold(retina_green,retina_green.max()-1,255,cv2.THRESH_BINARY);
         retina_green = cv2.bitwise_not(retina_green)
         retina_green = cv2.medianBlur(retina_green,55)
         img,contours,hierarchy = cv2.findContours(retina_green, 1, 2)
         area = 0;
         for item in contours:
              if cv2.contourArea(item) > area :
                  area = cv2.contourArea(item)
                  cnt = item          
         leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
         rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
         topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
         bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
         topmost = list(topmost);
         bottommost = list(bottommost);
         leftmost = list(leftmost);
         rightmost = list(rightmost);
         centre = [0,0];
         centre[1] = (rightmost[0] + leftmost[0] + topmost[0] + bottommost[0])/4 ;
         centre[0] = (rightmost[1] + leftmost[1] + topmost[1] + bottommost[1])/4 ;         
         radius_x = math.pow(rightmost[0]-leftmost[0],2) + math.pow(rightmost[1]-leftmost[1],2);
         radius_x = math.sqrt(radius_x)/2;
         radius_y = math.pow(topmost[0]-bottommost[0],2) + math.pow(topmost[1]-bottommost[1],2);
         radius_y = math.sqrt(radius_y)/2;
         radius =( radius_x + radius_y )/2;
         return(radius);
         
         

'''
this function normalizes all the images to (3000,3000)
and a linear intensity scaling is applied over the images in order to normalize
'''
def Normalize(image,parameter):
         retina = copy.copy(image);
         retina_blue , retina_green , retina_red = cv2.split(retina)
         # scaling the histogram linearly to normalize all the images
         if (parameter == 0):         
             cv2.normalize(retina,retina,0,255,cv2.NORM_MINMAX)
         elif(parameter == 1):
                retina = cv2.cvtColor(retina,cv2.COLOR_BGR2YCR_CB)
                channels = cv2.split(retina);
                cv2.equalizeHist(channels[0],channels[0]);
                cv2.merge(channels,retina);
                retina = cv2.cvtColor(retina,cv2.COLOR_YCR_CB2BGR);
         retina_green = cv2.bitwise_not(retina_green);
         ret, retina_green = cv2.threshold(retina_green,retina_green.max()-1,255,cv2.THRESH_BINARY);
         retina_green = cv2.bitwise_not(retina_green)
         retina_green = cv2.medianBlur(retina_green,55)
         img,contours,hierarchy = cv2.findContours(retina_green, 1, 2)
         area = 0;
         for item in contours:
              if cv2.contourArea(item) > area :
                  area = cv2.contourArea(item)
                  cnt = item          
         leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
         rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
         topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
         bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
         top = list(topmost);
         bottom = list(bottommost);
         leftmost = list(leftmost);
         rightmost = list(rightmost);
         leftmost[0] = leftmost[0];
         rightmost[0] = rightmost[0];
         if topmost[1] > 20 :
            top[1] = top[1] - 20;    
         if bottommost[1] > retina_blue.shape[0] :
            bottom[1] = bottom[1] + 20;
         image = retina[top[1] : bottom[1],leftmost[0] : rightmost[0]];
         resized_image = cv2.resize(image, (3000, 3000));
         return (resized_image);
'''
Exudate detection using binary thresholding
this function takes color retinal image as input and then outputs the exudate features
'''
def Exudates(image):
                        retina = copy.copy(image);
                        retina_blue , retina_green , retina_red = cv2.split(retina);
                        #preprocessing of the images
                        #applying adaptive histogram equalization
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        #preprocessing of the images
                        retina_green = clahe.apply(retina_green);
                        test_green = copy.copy(retina_green)                        
                        a = retina_green.max();                        
                        test_green = copy.copy(retina_green)                        
                        retina_g = cv2.equalizeHist(retina_green)
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
                        output = cv2.medianBlur(output,7)
                        kernel = np.ones((5,5),np.uint8)
                        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel) 
                        #copying image before applying findcontour as findcontour modifies the image
                        img,contours,hierarchy = cv2.findContours(output, 1, 2)
                        leftmost = retina_blue.shape;
                        rightmost = [0,0];
                        topmost = retina_blue.shape;
                        bottommost = [0,0];
                        for item in contours:
                            area = cv2.contourArea(item)
                            perimeter = cv2.arcLength(item,True);
                            if(perimeter !=0):
                                R= 4*np.pi*area/math.pow(perimeter,2);
                                if (R>0.3):
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
                        centre[1] = int((topmost[1] + bottommost[1])/2)
                        centre[0] = int((leftmost[0] + rightmost[0])/2)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
                        radius = math.sqrt(math.pow(topmost[1]-2-centre[1],2) + math.pow(leftmost[0]-2-centre[0],2));
                        radius = int(radius);
                        if (radius<25):
                           radius = 25;
                        elif (radius>40):
                            radius = 40;
                        centre = tuple(centre);
                        cv2.circle(retina_green,centre, int(radius)+5, (0,0,0), -1);
                        kernel = np.ones((10,10),np.uint8)
                        tophat = cv2.morphologyEx(retina_green, cv2.MORPH_TOPHAT, kernel)
                        tophat = cv2.equalizeHist(tophat)
                        ret, tophat = cv2.threshold(tophat,tophat.max()-10,255,cv2.THRESH_BINARY)
                        tophat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel)
                        exudate = copy.copy(tophat);
                        img,contours,hierarchy = cv2.findContours(tophat, 1, 2)
                        mask = np.ones(retina_green.shape, dtype="uint8") * 255
                        # removing false exudates
                        area = [];
                        distance = 0;                        
                        number = 0;
                        total_area = 0;
                        for items in contours:
                            temp_area = cv2.contourArea(items)
                            perimeter = cv2.arcLength(items,True)
                            #print temp_area
                            if perimeter != 0:
                                     R= 4*np.pi*temp_area/math.pow(perimeter,2);
                            else :
                                     R=0;
                            if (R<0.3) or  (temp_area>3000) :
                                             cv2.drawContours(mask, [items], -1, 0, -1)
                            else:
                                         number = number + 1;
                                         M = cv2.moments(items);
                                         if(M['m00'] != 0):
                                             cx = int(M['m10']/M['m00'])
                                             cy = int(M['m01']/M['m00'])
                                             distance = distance + math.sqrt(math.pow(centre[1]-cy,2) + math.pow(centre[0]-cx,2));
                                         area.append(temp_area);
                                         total_area = total_area + temp_area;                                         
                        exudate = cv2.bitwise_and(exudate,exudate, mask=mask);
                        exudate = cv2.medianBlur(exudate,5);
                        test = cv2.bitwise_and(test_green,exudate)
                        b1 = test.max();
                        test[(test>(b1-10))] = 0;
                        b2 = test.max();
                        test[(test>(b2-10))] = 0;
                        b3 = test.max();
                        #tophat = cv2.adaptiveThreshold(tophat,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,301,0)                        
                        if(number!=0):
                            distance = distance/number;
                            output = [total_area,np.asarray(area).max(),distance,number,b1/a,b2/a,b3/a];
                        else :
                            output = [total_area,0,0,number,b1/a,b2/a,b3/a];
                        return(output); 
   
'''
blood_vessels detection using blackhat operation and then binary thresholding
this function takes coloured retinal image and output the blood vessel features
'''                   
def blood_vessels(image):
                        retina = copy.copy(image);
                        retina_blue , retina_green , retina_red = cv2.split(retina);
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        retina_green = clahe.apply(retina_green)                
                        kernel = np.ones((10,10),np.uint8)
                        blackhat = cv2.morphologyEx(retina_green, cv2.MORPH_BLACKHAT, kernel)
                        #blackhat = cv2.GaussianBlur(blackhat,(15,15),0)
                        blackhat = cv2.medianBlur(blackhat,5)
                        blackhat = cv2.equalizeHist(blackhat)
                        ret, output_green = cv2.threshold(blackhat,blackhat.max()-10,255,cv2.THRESH_BINARY);
                        blood_vessel = cv2.medianBlur(output_green,5)                        
                        vessels = copy.copy(blood_vessel)
                        mask1 = np.ones(retina_green.shape, dtype="uint8") * 255
                        img,contours,hierarchy = cv2.findContours(blood_vessel, 1, 2)
                        for items in contours:
                            area = cv2.contourArea(items)
                            if (area < 10):
                                cv2.drawContours(mask1, [items], -1, 0, -1)
                        vessel = cv2.bitwise_and(vessels,vessels, mask=mask1);
                        blood_clot = copy.copy(vessel)
                        mask2 = np.ones(retina_green.shape, dtype="uint8") * 255
                        img,contours,hierarchy = cv2.findContours(vessel, 1, 2)
                        for items in contours:
                            area = cv2.contourArea(items)
                            perimeter = cv2.arcLength(items,True)
                            if perimeter !=0:
                                R= 4*np.pi*area/math.pow(perimeter,2);
                                if R>0.5:
                                    cv2.drawContours(mask2, [items], -1, 0, -1)
                        blood_vessel = cv2.bitwise_and(blood_clot,blood_clot, mask=mask2);
                        kernel = np.ones((5,5),np.uint8)
                        blood_vessel = cv2.morphologyEx(blood_vessel, cv2.MORPH_CLOSE, kernel)
                       
                        hemo = cv2.bitwise_not(mask2);
                        img,contours,hierarchy = cv2.findContours(blood_vessel, 1, 2)
                        thickness = [];
                        number_vessel = len(contours);
                        total_area = 0;
                        for items in contours:
                            area = cv2.contourArea(items)
                            total_area = total_area + area;
                            perimeter = cv2.arcLength(items,True)
                            if(perimeter !=0) :
                                thickness.append(area/perimeter);
                        if(len(thickness)!=0):
                            max_vessel = np.asarray(thickness).max();
                        else:
                            max_vessel = 0;
                        mean = np.average(np.asarray(thickness));
                        return (mean,max_vessel,number_vessel,total_area);
                        
def Hemorrhages(image):
    retina_green = copy.copy(image);
    retina_green[(retina_green<20)] = 255;
    retina_green = cv2.bitwise_not(retina_green);
