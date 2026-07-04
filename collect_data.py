import cv2
import urllib
import numpy as np
import os

classifier = cv2.CascadeClassifier(r"D:\Face Detection Project\Face Detection Project\haarcascade_frontalface_default.xml")

# url = "http://10.205.147.40:8080/shot.jpg"

# Initialize the laptop camera (0 is usually the default built-in webcam)
cap = cv2.VideoCapture(1)

data = []

while len(data) < 100:
    
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    # image_from_url = urllib.request.urlopen(url)
    # frame = np.array(bytearray(image_from_url.read()),np.uint8)
    # frame = cv2.imdecode(frame,-1)
    
    face_points = classifier.detectMultiScale(frame,1.3,5)
    
    if len(face_points)>0:
        for x,y,w,h in face_points:
            face_frame = frame[y:y+h+1,x:x+w+1]
            cv2.imshow("Only face",face_frame)
            if len(data) < 100:
                data.append(face_frame)
                print(f"{len(data)} / 100")
                break # Only process one face per frame for the dataset

            # if len(data)<=100:
            #     print(len(data)+1,"/100")
            #     data.append(face_frame)
            #     break


    cv2.putText(frame, str(len(data)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255))
    cv2.imshow("frame",frame)

    if cv2.waitKey(30) == ord("q"):
        break

# Release the camera hardware and close all windows
cap.release()
cv2.destroyAllWindows()
        
# Save the data if 100 images were collected
if len(data) == 100:
    name = input("Enter Face holder name: ")
    
    # Create 'images' folder if it doesn't exist to prevent errors
    if not os.path.exists("images"):
        os.makedirs("images")
        
    for i in range(100):
        cv2.imwrite("images/" + name + "_" + str(i) + ".png", data[i])
    print("Done! All images saved.")
else:
    print(f"Need more data. Only collected {len(data)} images.")