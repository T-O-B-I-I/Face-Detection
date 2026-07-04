import cv2
import numpy as np
from keras.models import load_model
# from deepface import DeepFace  # NEW: Import the emotion detection library

# Load your cascade and your custom name model
classifier = cv2.CascadeClassifier(r"D:\Face Detection Project\Face Detection Project\haarcascade_frontalface_default.xml")
model = load_model("final_model.h5")
emotion_model = load_model("Emotion_detection_model_new.h5")

cap = cv2.VideoCapture(0)

def get_pred_label(pred):
    # Ensure this matches the exact output from your training script!
    labels = ["ankush", "priyansu", "suvam"] 
    return labels[pred]

def get_emotion_label(pred):
    # Standard FER2013 labels. 
    # IMPORTANT: Ensure this matches the exact order of classes you used when training!
    labels = ['angry', 'happy', 'neutral', 'sad', 'suprised']
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

def preprocess_for_emotion(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # IMPORTANT: Change (48, 48) if your emotion model was trained on a different size!
    img = cv2.resize(img, (48, 48)) 
    img = img.reshape(1, 48, 48, 1)
    img = img / 255.0
    return img

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
      
    for x, y, w, h in faces:
        # Crop the face out of the frame
        face = frame[y:y+h, x:x+w]
        
        # 1. Get the NAME using your custom model
        processed_face = preprocess(face)
        prediction = model.predict(processed_face, verbose=0) 
        name = get_pred_label(np.argmax(prediction))
        
        # 2. Get the EMOTION
        # Wrap in try/except in case the face crop is too small/glitchy
        try:
            processed_face_emo = preprocess_for_emotion(face)
            prediction_emo = emotion_model.predict(processed_face_emo, verbose=0)
            emotion = get_emotion_label(np.argmax(prediction_emo))
        except Exception as e:
            emotion = "Error"
            
        # 3. Combine them into your desired text format
        display_text = f"{name} is {emotion}"
        
        # Draw the rectangle and the text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame, display_text, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("capture", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()