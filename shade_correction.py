# shade correction of the images


import cv2
import numpy as np
import copy
import os

retina = cv2.imread(r'C:\Users\Skotur\Desktop\Diabetic Retinopathy\normalized_images\10_right.jpeg',1)
retina_blue , retina_green , retina_red = cv2.split(retina)
kernel = np.ones((15,15),np.float32)/225
retina_mean = cv2.filter2D(retina,-1,kernel)

screen_res = 1280, 720
scale_width = screen_res[1] / retina_blue.shape[1]
scale_height = screen_res[0] / retina_blue.shape[0]
scale = min(scale_width, scale_height)
window_width = int(retina_blue.shape[1] * scale)
window_height = int(retina_blue.shape[0] * scale)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', window_width, window_height)

Z = np.float32(retina)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((retina.shape))

#background = cv2.absdiff(retina_mean,retina)
#output = cv2.adaptiveThreshold(retina_green,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,255,2)
#output,th3 = cv2.threshold(retina_green,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#create NumPy arrays from the boundaries
'''
lower = np.array([0,150,0], dtype = "uint8")
upper = np.array([200,255,180], dtype = "uint8")
mask = cv2.inRange(retina, lower, upper)
output = cv2.bitwise_and(retina, retina, mask = mask)
'''
cv2.imshow("output", res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
