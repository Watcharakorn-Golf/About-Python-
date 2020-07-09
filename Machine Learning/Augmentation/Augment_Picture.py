# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:28:28 2019

@author: Capmu
"""

from PIL import Image
import numpy as np
import os
import h5py
from progress.bar import Bar

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2

#--------------------------------------------------------------------------------------------------------------------------
#   name of folder that store the data set "and" name of each image in that folder (ต้องตั้งชื่อโฟลเดอร์กับชื่อรูปให้เป็น type เดียวกัน)
#--------------------------------------------------------------------------------------------------------------------------

test_set_CAT = "sunrise"
#--------------------------------------------------------------------------------------------------------------------------
#   Path that keep a Data set #เริ่มจากที่อยู่ของไฟล์นี้
#--------------------------------------------------------------------------------------------------------------------------
def capmu_get_from(data_type):
    if data_type == test_set_CAT:
        get_from = "dataset/sunrise/" + str(test_set_CAT)
    return get_from
#--------------------------------------------------------------------------------------------------------------------------
#   Function that use for save the Data
#--------------------------------------------------------------------------------------------------------------------------
def capmu_save_as(save_type, data_type):
    if save_type == 1:
        save_as = "Augmented_image/1Horizontal_shifted/" + str(data_type) + "/hz_shifted_"
    elif save_type == 2:
        save_as = "Augmented_image/2Vertical_shifted/" + str(data_type) + "/vc_shifted_"
    elif save_type == 3:
        save_as = "Augmented_image/3Horizontal_fliped/" + str(data_type) + "/hz_fliped_"
    elif save_type == 4:
        save_as = "Augmented_image/4Random_rotationed/" + str(data_type) + "/RR_"
    elif save_type == 5:
        save_as = "Augmented_image/5Brighted/" + str(data_type) + "/Brighted_"
    elif save_type == 6:
        save_as = "Augmented_image/6Zoomed/" + str(data_type) + "/Zed_"
    return save_as
#----------------------------------------------------------------------------------------------------------------------------------
#               horizontal shift image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_horizontal_shift_x9(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 1
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,100])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9): # ถ้า i = 9 ก็จะได้รูปใหม่ออกมาเพิ่ม 9 รูป
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
    # show the figure
    pyplot.show()
    
#----------------------------------------------------------------------------------------------------------------------------------
#               vertical shift image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_vertical_shift_x9(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 2
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=0.5)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
    # show the figure
    pyplot.show()

#----------------------------------------------------------------------------------------------------------------------------------
#               horizontal flip image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_horizontal_flip_x1(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 3
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(horizontal_flip=True)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)

        # generate samples and plot
        for i in range(1): # ใช้แค่ อันเดียวพอเพราะ flip ได้แค่ครั้งเดียว
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
    # show the figure
    pyplot.show()

#----------------------------------------------------------------------------------------------------------------------------------
#               random rotation image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_rotation_x9(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 4
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
        # show the figure
        pyplot.show()

#----------------------------------------------------------------------------------------------------------------------------------
#               brighting image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_brighting_x9(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 5
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
        # show the figure
        pyplot.show()         
            
#----------------------------------------------------------------------------------------------------------------------------------
#               zoom image augmentation
#----------------------------------------------------------------------------------------------------------------------------------
def capmu_zoom_x9(start_at, end_at, data_type):
    
    get_from = capmu_get_from(data_type)
    
    s = 6
    j = 0 # ใช้ในการตั้งชื่อรูปไม่ให้ซ้ำ

    for i in range((end_at - start_at)+1): #ทำ n รอบ แต่ใช้ i เป็น n-1
        # load the image
        image_path_andtype = str(get_from) + ' (' + str(start_at) + ').jpg'
        img = load_img(image_path_andtype)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # ทำให้สีหายเพี้ยน
            cv2.imwrite(str(capmu_save_as(s, data_type)) + str((i+1)+j) + ' .jpg', image) # Save รูปที่ทำตามจำนวน i ไว้ที่ directory เดียวกันกับที่ไฟล์นี้อยู่
        j = j+9
        start_at = start_at + 1
        # show the figure
        pyplot.show()
            
#----------------------------------------------------------------------------------------------------------------------------------
#           Using
what_doyou_want = 'rainy'
#----------------------------------------------------------------------------------------------------------------------------------
capmu_horizontal_shift_x9(1, 50, test_set_CAT)
capmu_vertical_shift_x9(1, 50, test_set_CAT)
capmu_horizontal_flip_x1(1, 50, test_set_CAT)
capmu_rotation_x9(1, 50, test_set_CAT)
capmu_brighting_x9(1, 50, test_set_CAT)
capmu_zoom_x9(1, 50, test_set_CAT)
