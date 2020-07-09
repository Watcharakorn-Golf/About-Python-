# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:32:01 2018

@author: Golferate
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:28:07 2018

@author: Golferate
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    # Take each frame
    _, frame = cap.read()
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    edge = cv2.Canny(frame,100,100)
    
    cv2.imshow('org',laplacian)
    cv2.imshow('edge',edge)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
