# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:10:49 2020

@author: Golferate
"""
import csv

data = open("datatest2.txt","r")
txt = data.read()
spt = txt.split(" ")
#create_file = open("Occupancy room.csv","w+")    
dataset = list()

print(spt[-1])
for i in spt[1:]:
    spt2 = i.split(",")
    spt3 = spt2[1:-1]
    spt4 = spt3[-1].split("\n")
    spt5 = spt3[:-1]
    spt6 = spt5 + spt4 
    finish = spt6[:-1]
    dataset.append(finish)
    #print(finish)
#myfile = open("iot_dataset.csv", 'a+', newline='')
#    
#with myfile:
#    wr = csv.writer(myfile)
#    wr.writerows(dataset)


        
