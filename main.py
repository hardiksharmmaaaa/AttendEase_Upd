import cv2
import pickle
import numpy as np
import os
import random

# Try multiple camera indices
for cam_index in range(4):
    video = cv2.VideoCapture(cam_index)
    ret, frame = video.read()
    if ret and frame is not None:
        print(f"Using camera index {cam_index}")
        break
    video.release()
else:
    print("No accessible camera found! Please check your camera and permissions.")
    exit(1)

facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    if not ret or frame is None:
        print("Camera not accessible! Please check your camera and permissions.")
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Semi-transparent info panel at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 60), (30,30,60), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Watermark/logo
    cv2.putText(frame, 'AttendEase', (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255,180), 3, cv2.LINE_AA)
    # Progress bar for face data collection
    bar_x, bar_y, bar_w, bar_h = 50, 15, 300, 20
    progress = int((len(faces_data)/100)*bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100,100,100), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+progress, bar_y+bar_h), (0,255,180), -1)
    cv2.putText(frame, f'Collecting faces: {len(faces_data)}/100', (bar_x+10, bar_y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        # Animated neon border
        neon_color = tuple([random.randint(180,255) for _ in range(3)])
        for j in range(6,0,-2):
            cv2.rectangle(frame, (x-j,y-j), (x+w+j, y+h+j), neon_color, 2)
        cv2.putText(frame, str(len(faces_data)), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)