# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:34:09 2015

@author: Saiprasad
"""
# optical disc detection
from __future__ import division
import cv2
import numpy as np
import os
import csv
import DR
import math

rootDir = r'E:\train_sample'
RESULT = [];
count = 0;
for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                        a = dirName + '\\' + fname
                        count =count + 1;                        
                        retina = cv2.imread(a,1);
                        image_name = os.path.splitext(os.path.basename(a))[0];
                        with open(r"E:\acads\internships\Syntel\Diabetic Retinopathy\trainLabels.csv", 'r') as file:
                            reader = csv.reader(file)
                            DR_Stage = [line[1] for line in reader if line[0] == image_name]
                            DR_Stage = np.asarray(DR_Stage); 
                        mean,median,maximum = DR.disturbance(retina);
                        RESULT.append([fname ,mean,median,maximum,int(DR_Stage[0])]);

print 'writing to csv file'
with open(os.path.join(r'E:\acads\internships\Syntel\Diabetic Retinopathy', 'output2.csv'), 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["image", "mean","median","maximum","DR stage"], delimiter = ',')
    writer.writeheader()
    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerows(RESULT)               
                       
