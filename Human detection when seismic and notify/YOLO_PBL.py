import cv2
from darkflow.net.build import TFNet
import numpy as np
import math
import time
from ubidots import ApiClient
import requests

list = []
avg_num = 10

option = {
    'model': 'cfg/tiny-yolo.cfg',
    'load': 'bin/tiny-yolo.weights',
    'threshold': 0.25,
    'gpu': 0.6
}

tfnet = TFNet(option)
colors = [tuple(255 * np.random.rand(3)) for i in range(50)]
   
api = ApiClient(token='A1E-CbLzMyw7Yvp4nWCoEeDkvh6fNdg7x0')
human1 = api.get_variable('5bd7ec6ec03f973282f5d565')
switch = api.get_variable('5bd7a976c03f97712a439948')
eq_sensor = api.get_variable('5bda60cbc03f9751dc6ff7a8')
power = api.get_variable('5bda9246c03f9704e2198b49')

url = 'https://notify-api.line.me/api/notify'
token = '7ZOYtfjPvDhyVbnrDCOSPQ4EQRTAFSFG1DOTD3YcF6h'
headers = {'Authorization': 'Bearer ' + token}

print("ready for it")

while True:
    enable = eq_sensor.get_values(1)
    ook = power.get_values(1)
    
    if ook[0]['value']:
        power.save_value({'value': 0})
        break
    time.sleep(2)
    
    if enable[0]['value']:
        print("OKAY START !!!")
        capture = cv2.VideoCapture(0)
        while (capture.isOpened()):
            
            count = 0
            ret, frame = capture.read()
            enable = eq_sensor.get_values(1)
            sw_ctrl = switch.get_values(1)
    
            if ret:
                results = tfnet.return_predict(frame)
                
                for color, result in zip(colors, results):
                    label = result['label']
            
                    if label == 'person':
                        tl = (result['topleft']['x'], result['topleft']['y'])
                        br = (result['bottomright']['x'], result['bottomright']['y'])
                        frame = cv2.rectangle(frame, tl, br, color, 5)
                        count = count + 1
                        frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, ( 20,  20,  20), 7)
                        frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (207, 227, 253), 2)
                        #(ตัวอักษร,ตำแหน่ง,front,ขนาดตัวอักษร,สี,ความหนา,)
            
                if sw_ctrl[0]['value']:
                    cv2.imwrite("Capture.jpg",frame)
                    print ("Captured")
            
                    message = 'Captured'
                    payload = {"message" :  message}
                    files = {"imageFile": open("Capture.jpg", "rb")} 
                    r = requests.post(url, headers = headers, params = payload,  files = files)
                
                    switch.save_value({'value': 0})
                       
            
                # cv2.imshow('frame', frame)
            
            #time.sleep(1)
            
            if len(list) < avg_num:
                list.append(count)
                print("crop")
            elif len(list) == avg_num:
                avg = sum(list) / avg_num
                avg_people = math.ceil(avg)
                print("found{}".format(avg_people))
                human1.save_value({'value':avg_people, 'context':{'lat': 13.736717, 'lng': 100.523186}})
                list.clear()
                       
            if enable[0]['value']:
                pass
            else:
                break
        
        human1.save_value({'value' :0})
        t = int(time.time()) * 1000
        human1.remove_values(0, t)
        print ("Reset data")
               
        capture.release()
    
cv2.destroyAllWindows()

