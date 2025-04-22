#!/usr/loca/bin/python3
import os
import sys
from datetime import datetime as datetime
import tensorflow as tf


# c.f.: https://ai.google.dev/edge/litert/models/convert_tf

# for model optimization to INT16 or INT8, 
# c.f.: https://ai.google.dev/edge/litert/models/model_optimization
# c.f.: https://b10515007.medium.com/tensorflow-lite-model-quantization-2b538eb5bc04


if __name__ == "__main__":
    _ = os.system('clear')
    if len(sys.argv) == 1:
        print('no trained model pathfile specified ...')
        print('exit ...')
        exit()

    if len(sys.argv) >= 2:
        model_pathfile = sys.argv[1]
        if not os.path.exists(model_pathfile):
            print(f'model file {model_pathfile} is not found !')
            print('exit ...')
            exit()

    # Convert the model
    saved_model_dir = model_pathfile #f'../trial_02_AudioMonotoneDetection/{model_serial}/model'
    print(os.path.exists(saved_model_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(
            saved_model_dir
        ) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    tmp = saved_model_dir.split('/')
    if len(tmp) > 1:
        model_serial = tmp[-2]
    else:
        model_serial = tmp[-1]
    tfmodel_tobesaved = f'{model_serial}.tflite'
    with open(tfmodel_tobesaved, 'wb') as f:
      f.write(tflite_model)

