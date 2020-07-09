from ubidots import ApiClient
import time
import cv2
import requests

api = ApiClient(token='A1E-CbLzMyw7Yvp4nWCoEeDkvh6fNdg7x0')

human1 = api.get_variable('5bd7ec6ec03f973282f5d565')
human1.save_value({'value':5, 'context':{'lat': 13.736717, 'lng': 100.523186}})
switch = api.get_variable('5bd7a976c03f97712a439948')

while True:
    # 1st - human detection / Bangkok
    
    sw_ctrl = switch.get_values(1)
    time.sleep(1)
    
    if sw_ctrl[0]['value']:
        print ("on")
        switch.save_value({'value': 0})
        
        human1.save_value({'value' :0})
        t = int(time.time()) * 1000
        human1.remove_values(0, t)
        
    else:
        print ("off")

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
url = 'https://notify-api.line.me/api/notify'
token = 'ph40Bf6gkRCuH0obCHMOhkZluLDLFhkBAlCjEQj9nY3'     # token key
headers = {'Authorization': 'Bearer ' + token}

message =  'hello'
payload = {"message" :  message}
files = {"imageFile": open("cat.jpg", "rb")} 

r = requests.post(url, headers = headers, params = payload,  files = files)