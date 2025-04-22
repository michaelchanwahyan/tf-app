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
        print(
'''
NAME
     main_convert_model.py - convert tensorflow ("tf") PC-trained model to ARM-based
                             embedded format, with optional quantization control

SYNOPSIS
     main_convert_model.py    [tf model path-file-name]    [optional argument]

DESCRIPTION
     PC-trained tf model may have 4 ways of model conversion.
         A) post-training float16 quantization
                >    no data is needed,
                >    size reduction up to 50%
         B) post-training dynamic range quantization
                >    no data is needed,
                >    size reduction up to 75%
         C) post-training integer quantization
                >    unlabelled representative sample is needed,
                >    size reduction up to 75%
         D) quantization-aware training
                >    labelled training data is needed,
                >    size reduction up to 75%

     to provide post-training quantization type (i.e. type A)-C)) info,
     specify "FLOAT16" or "DYNAMIC" or "INT8" as the 2nd input argument
'''
        )
        exit()

    if len(sys.argv) >= 2:
        model_pathfile = sys.argv[1]
        if not os.path.exists(model_pathfile):
            print(f'model file {model_pathfile} is not found !')
            print('exit ...')
            exit()

    if len(sys.argv) >= 3:
        quant_opt = sys.argv[2]
        if \
                not quant_opt.lower() == 'float16' \
                and \
                not quant_opt.lower() == 'dynamic' \
                and \
                not quant_opt.lower() == 'int8' :
            print(f'specified quantization option {quant_opt} is not supported !')
            print('exit ...')
            exit()


    # Convert the model
    saved_model_dir = model_pathfile
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

