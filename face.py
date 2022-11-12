import numpy as np
import cv2

profile_face_cascade = cv2.CascadeClassifier('FaceCascades/haarcascade_profileface.xml')
front_face_cascade = cv2.CascadeClassifier('FaceCascades/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
img_flopsie = cv2.imread('images/flopsie.png', cv2.IMREAD_COLOR)
while(True):
    #Capature frame-by-frame lmao
    old = None
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            img_flopsie = cv2.resize(img_flopsie, (w, h), interpolation=cv2.INTER_AREA)
            frame[y:y+h, x:x+w, :] = img_flopsie
    else:
        faces = front_face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            img_flopsie = cv2.resize(img_flopsie, (w, h), interpolation=cv2.INTER_AREA)
            frame[y:y+h, x:x+w, :] = img_flopsie
    #Display the resulting frame xd
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()