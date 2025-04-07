#!/bin/python3

import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# Load the ResNet50 model
model = ResNet152(weights='imagenet')

# Load and preprocess the image
img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to classify the image
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=8)[0])

