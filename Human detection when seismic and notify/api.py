from ubidots import ApiClient
import time

api = ApiClient(token='A1E-CbLzMyw7Yvp4nWCoEeDkvh6fNdg7x0')

# 1st - human detection / Bangkok
human1 = api.get_variable('5b98da2ec03f97595adda426')
human1 = human1.save_value({'value':4, 'context':{'lat': 13.736717, 'lng': 100.523186}})
time.sleep(1)

# 2nd - human detection / Nonthaburi
human2 = api.get_variable('5b98d435c03f97522959e350')
human2 = human2.save_value({'value':14, 'context':{'lat': 13.859108, 'lng': 100.521652}})
time.sleep(1)

# 3rd - human detection / Nakhon Ratchasima
human3 = api.get_variable('5b98d467c03f9752adffcf63')
human3 = human3.save_value({'value':2, 'context':{'lat': 14.979900, 'lng': 102.097771}})
time.sleep(1)

# 4th - human detection / Nakhon Pathom
human4 = api.get_variable('5b98d473c03f975290b64f6f')
human4 = human4.save_value({'value':1, 'context':{'lat': 13.814029, 'lng': 100.037292}})
time.sleep(1)

# 5th - human detection / Chon Buri
human5 = api.get_variable('5bd69f96c03f974c48a47f43')
human5 = human5.save_value({'value':15, 'context':{'lat': 13.361143, 'lng': 100.984673}})
time.sleep(1)