import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64
model = load_model(r'C:\Users\baasi\task4\gesture_model.h5')

train_data_dir = r'C:\Users\baasi\task4\archive (3)\leapGestRecog'

train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

test_img_path = r'C:\Users\baasi\task4\archive (3)\leapGestRecog\01\01_palm\frame_01_01_0001.png'
test_img = cv2.imread(test_img_path)
test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
test_img = np.expand_dims(test_img, axis=0)
test_img = test_img / 255.0

prediction = model.predict(test_img)

predicted_class_index = np.argmax(prediction, axis=1)[0]

class_labels = {v: k for k, v in train_data.class_indices.items()}
predicted_class = class_labels[predicted_class_index]

print(f'Predicted Gesture Class: {predicted_class}')
print(train_data.class_indices)
print(f'Predicted Gesture Class: {predicted_class} ({train_data.class_indices[predicted_class]})')
