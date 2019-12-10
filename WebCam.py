# import cv2

# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)

# if vc.isOpened(): # try to get the first frame
    # rval, frame = vc.read()
# else:
    # rval = False

# while rval:
    # cv2.imshow("preview", frame)
    # rval, frame = vc.read()
    # key = cv2.waitKey(20)
    # if key == 27: # exit on ESC
        # break
# cv2.destroyWindow("preview")



# sudo modprobe bcm2835-v4l2

import cv2
# import os 
# os.system('sudo modprobe bcm2835-v4l2') 
cap = cv2.VideoCapture(0) 
cap.set(3, 480) #sirka videa 

cap.set(4, 320) #vyska videa 
cap.read() #adjustace svetla 
while(True): 
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()