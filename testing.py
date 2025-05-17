from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import random
import pyttsx3

def speak(str1):
    engine = pyttsx3.init()
    engine.say(str1)
    engine.runAndWait()

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

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground=cv2.imread("Background.jpeg")

COL_NAMES = ['NAME', 'TIME']

# Ensure Attendance directory exists
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

while True:
    ret,frame=video.read()
    if not ret or frame is None:
        print("Camera not accessible! Please check your camera and permissions.")
        break
    # Blur the background for depth effect
    blurred_bg = cv2.GaussianBlur(imgBackground, (51, 51), 0)
    overlay = blurred_bg.copy()
    # Semi-transparent HUD at the top
    cv2.rectangle(overlay, (0,0), (overlay.shape[1], 80), (30,30,60), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, blurred_bg, 1 - alpha, 0, blurred_bg)
    # Watermark/logo
    cv2.putText(blurred_bg, 'AttendEase', (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255,180), 4, cv2.LINE_AA)
    # Attendance stats (dummy for now)
    cv2.putText(blurred_bg, f'Faces in DB: {len(LABELS)}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        # Animated neon border
        neon_color = tuple([random.randint(180,255) for _ in range(3)])
        for i in range(6,0,-2):
            cv2.rectangle(frame, (x-i,y-i), (x+w+i, y+h+i), neon_color, 2)
        # Smooth fade-in for name
        for alpha_txt in np.linspace(0.2, 1, 5):
            cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        attendance=[str(output[0]), str(timestamp)]
    # Place the processed frame into the blurred background
    frame_resized = cv2.resize(frame, (640, 480))
    blurred_bg[162:162 + 480, 55:55 + 640] = frame_resized
    cv2.imshow("Frame",blurred_bg)
    k=cv2.waitKey(1)
    if k==ord('m'):
        speak("Attendance Taken.. Thankyou")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k==ord('m'):
        break
video.release()
cv2.destroyAllWindows()
