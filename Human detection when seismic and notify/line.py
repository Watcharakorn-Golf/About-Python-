# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:35:38 2018

@author: Golferate
"""

import requests

def main():

    url = 'https://notify-api.line.me/api/notify'
    token = 'ph40Bf6gkRCuH0obCHMOhkZluLDLFhkBAlCjEQj9nY3'     # token key
    headers = {'Authorization': 'Bearer ' + token}

    message =  'hello'
    payload = {"message" :  message}
    files = {"imageFile": open("cat.jpg", "rb")} 

    r = requests.post(url, headers = headers, params = payload,  files = files)

if _name_ == '_main_':
    main()