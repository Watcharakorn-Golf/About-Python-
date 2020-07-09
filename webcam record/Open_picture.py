# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 19:13:06 2018

@author: Golferate
"""
import time
import cv2
keep = list(range(3))
#while True:

for i in keep:
    #read img
    file_img = 'miori.png'
    img = cv2.imread(file_img)
  
    cv2.imwrite('copy.jpg',img)
    num = cv2.flip(img,i)
    cv2.imshow('miori.png',num)
    
    time.sleep(2)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
