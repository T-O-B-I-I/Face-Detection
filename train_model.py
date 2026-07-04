import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data you collected
data_dir = os.path.join(os.getcwd(), 'data')

print("Loading data...")
with open(os.path.join(data_dir, "images.p"), 'rb') as f:
    X = pickle.load(f)
    
with open(os.path.join(data_dir, "labels.p"), 'rb') as f:
    y = pickle.load(f)

print(X.shape)

print(y.shape)    

# 2. Preprocess the Data
# Reshape X to include the channel dimension (1 for grayscale) and normalize to 0-1
X = X.reshape(-1, 100, 100, 1) / 255.0

# Convert text labels (e.g., "ankush", "suvam") into numbers, then into categories
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
set(y)
encoder.inverse_transform([0])
p = len(set(y))
y_categorical = to_categorical(y_encoded)

# Save the exact order of the labels so you know how to write them in recognize.py
print("\n--- CRITICAL: COPY THESE LABELS ---")
print("Your label order for recognize.py should be:")
print(list(encoder.classes_))
print("-----------------------------------\n")

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 3. Build the Neural Network
print("Building the model...")
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Feed-Forward
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents the model from memorizing the data (overfitting)
    
    # Output Layer (Number of neurons equals the number of people)
    Dense(len(encoder.classes_), activation='softmax')
])

# 4. Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()
# 5. Train the Model
print("Training started...")
# Epochs = how many times it loops through the data. 
# You can increase epochs to 20 or 30 if accuracy is low.
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# 6. Save the Model
model.save("final_model.h5")
print("\nSuccess! 'final_model.h5' has been created in your project folder.")