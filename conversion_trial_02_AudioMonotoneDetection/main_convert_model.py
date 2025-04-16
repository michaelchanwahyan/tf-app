#!/usr/loca/bin/python3
import os
from datetime import datetime as datetime
import tensorflow as tf


# c.f.: https://ai.google.dev/edge/litert/models/convert_tf


# Convert the model
model_serial = 'model_20250416085423'
saved_model_dir = f'../trial_02_AudioMonotoneDetection/{model_serial}/model'
print(os.path.exists(saved_model_dir))
converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir
    ) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
tfmodel_tobesaved = f'{model_serial}.tflite'
with open(tfmodel_tobesaved, 'wb') as f:
  f.write(tflite_model)


# for model optimization to INT16 or INT8, 
# c.f.: https://ai.google.dev/edge/litert/models/model_optimization

