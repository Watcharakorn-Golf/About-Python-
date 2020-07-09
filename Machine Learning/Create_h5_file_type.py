"""
README

Prepare data
    - Seperate your image data. One class for one folder.

How to use this code
    - Copy this file and paste in the same directory of your main program.

    - save_dir, it is file name or directory that you want to save file.
        save_dir must have file types 
        such as .h5
                .hdf5

    - base_path , you have to input directory of your image data and end up with /

    - class_name is your folder's names that in base_path directory.

    The Function does not return any parameters.
    
    The output of function is .h5 or .hdf5 files.
"""

from PIL import Image
import numpy as np
import cv2
import os
import h5py
from progress.bar import Bar


def create_h5_data(save_dir, base_path, class_name, file_for, img_row=64, img_col=64):
    """
    Implements to convert image to numpy array.
    Arguments :
        - save_dir : the directory and file name + .hdf5 or .h5

        - base path and class name : must end with /
            Ex. base_path = 'Homework/Dataset/shapes/'
                class_name = ['circle/', 'square/', 'star/', 'triangle/']

        - class name is the name of folder that are in base path directory.

        - file_for : type of file that has 2 types , train and test

        - img_row : the image row. By default is 64

        - img_col : the image column. By default is 64
    """
    # define variable
    test_img = []
    test_labels = []
    train_img = []
    train_labels = []
    textf = open("Readme_"+file_for+".txt","w")
    # for train dataset
    if file_for == "train":
        for i in range(len(class_name)):
            for r, d, files in os.walk(base_path + class_name[i]):
                with Bar(class_name[i] +'Processing', max=len(files)) as bar:        # create progress bar
                    for num in range(len(files)):
                        # collect image to list and label is depends on index of class_name
                        image_ori = cv2.imread(base_path + class_name[i] + '/' + files[num])
                        image = cv2.resize(image_ori, (img_row,img_col))
                        train_img.append(image)
                        train_labels.append(i)
                        bar.next()

        # write data in .hdf5 or .h5 form
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('train_img', data=train_img)
            f.create_dataset('train_labels', data=train_labels)

        print("train dataset has ",len(train_img))
        textf.write("train dataset of " + save_dir + " has "+ str(len(train_img)) + '/n')

        for c in range(len(class_name)):
            textf.write("label " + str(c)+" is " + class_name[c] + '/n')
            print("label ",c," is ",class_name[c])
        textf.close() 
    # for test dataset
    
    elif file_for == "test":
        for i in range(len(class_name)):
            for r, d, files in os.walk(base_path + class_name[i]):
                with Bar(class_name[i] +'Processing', max=len(files)) as bar:        # create progress bar
                    for num in range(round(0.5*len(files)+1),len(files)):
                        # collect image to list and label is depends on index of class_name
                        image_ori = cv2.imread(base_path + class_name[i] + '/' + files[num])
                        image = cv2.resize(image_ori, (img_row,img_col))
                        test_img.append(image)
                        test_labels.append(i)
                        bar.next()

        # write data in .hdf5 or .h5 form
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('test_img', data=test_img)
            f.create_dataset('test_labels', data=test_labels)
            

        print("test dataset has ",len(test_img))
        textf.write("test dataset of " + save_dir + " has "+ str(len(test_img)) + '/n')

        for c in range(len(class_name)):
            textf.write("label " + str(c)+" is " + class_name[c] + '/n')
            print("label ",c," is ",class_name[c])
        textf.close() 
    # the other file_for input
    else:
        return print("create_h5_file does not have " + file_for + ". It has /'train/' and /'test/'")
    
#set parameters for create datasets    
        
save_dirc ="C:/Users/USER/Downloads/Documents/ML/Proj ML/finalproj/weather.h5"
b_path = "C:/Users/USER/Downloads/Documents/ML/Proj ML/finalproj/dataset/img/"
c_name = ["cloudy/","rainy/","sunny/","sunrise/"]
file_f = "test"
create_h5_data(save_dir= save_dirc, base_path= b_path, class_name= c_name, file_for= file_f, img_row=64, img_col=64)
