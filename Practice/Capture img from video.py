import cv2
from time import sleep
cap = cv2.VideoCapture(0)
num = []
while True:
    ret, img = cap.read()    
    cv2.imshow('show',img)
    if cv2.waitKey(20) & 0xff == ord('g'):
        cop = cv2.imwrite("copy_file{}.jpg".format(len(num)),img)
        num.append(cop)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
#(len(upper_body))
