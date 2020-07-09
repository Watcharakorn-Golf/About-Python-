# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:32:08 2018

@author: Golferate
"""

a = int(input("first"))
b = int(input("second"))

even_list = []
odd_list  = []
li = range(a,b+1)
for i in range(len(li)):
        if i%2 ==0:
            even_list.append(i+2)
            
        else:
             odd_list.append(i)
             
print(even_list)
print(odd_list)
SamE= sum(even_list)
SamO= sum(odd_list)
print(SamE)
print(SamO)
