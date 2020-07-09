# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:16:28 2018

@author: Golferate
"""

from ubidots import ApiClient

api = ApiClient(token='A1E-CbLzMyw7Yvp4nWCoEeDkvh6fNdg7x0')

# 1st - human detection / Bangkok
human1 = api.get_variable('5b98da2ec03f97595adda426')
human1 = human1.save_value({'value':7, 'context':{'lat': 13.736717, 'lng': 100.523186}}) 