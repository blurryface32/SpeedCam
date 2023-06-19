import cv2
import numpy as np
end = 0


# reading video
cap = cv2.VideoCapture("Resources/traffic4.mp4")

#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)

#KERNALS
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

#loop for frames to run one after other
while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height,width,_ = frame.shape

    #Extract ROI
    roi = frame[50:540,200:960]

    #Masking
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)

    #finding contours
    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    #sorting the vehicles from other objects 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)

            #drawing a box around vehicles in the frame
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])
    

    #DISPLAY
    cv2.imshow("Mask",mask2)
    # cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(30)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()