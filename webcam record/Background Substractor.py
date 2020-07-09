# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 23:03:49 2018

@author: Golferate
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('000.WEBM')
thresh_pic = 'Picture Threshold.png'
#fgbg = cv2.createBackgroundSubtractorMOG2()

ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray , (5,5), 0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray , (5,5), 0)
    diff = cv2.absdiff(first_gray, gray)
    ret, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
  #  fgmask2 = fgbg.apply(frame)
    #fgmask = fgbg.apply(gray)
    
    
    cv2.imshow('original', first_frame)
   # cv2.imshow('fg',fgmask)
    cv2.imshow('difference',diff)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()

cv2.destroyAllWindows()
    
    
    
    
    
