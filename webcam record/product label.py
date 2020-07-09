# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:49:29 2018

@author: Golferate
"""
prdt_keep = []
price_keep =[]

while True: 
    prdt= input("product:")
    if prdt == 'stop':
            break
    else:
               price= input("price:")
               prdt_keep.append(str(prdt))
               price_keep.append(int(price))
for i in range(0,len(price_keep)):
  print("Food:",prdt_keep[i],"Price:",price_keep[i])             


    


       
    
