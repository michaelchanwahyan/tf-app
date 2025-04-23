#!/usr/loca/bin/python3
import os
import sys
import pickle as pkl
from datetime import datetime as datetime
import numpy as np
import random
import tensorflow as tf


# c.f.: https://ai.google.dev/edge/litert/models/convert_tf

# for model optimization to INT16 or INT8, 
# c.f.: https://ai.google.dev/edge/litert/models/model_optimization
# c.f.: https://b10515007.medium.com/tensorflow-lite-model-quantization-2b538eb5bc04


# -----------------------------------------------------------------------------
# GLOBAL VARIABLES
# -----------------------------------------------------------------------------
__NOQNT__       = 'noquant'
__FLT16__       = 'float16'
__DYRNG__       = 'dynamic'
__INT8__        = 'int8'
__FULLINT8__    = 'fullint8'

data_train  = './representative_data_train.pkl'
data_num    = 1
sample_rate = 0.01

representative_data_exception = []

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS
# -----------------------------------------------------------------------------
quant_opt = __NOQNT__
representative_data_pkl_pathfile = '' # for INT8 quantization, the representative dataset


#def func_representative_data_gen(data_train, data_num, sample_rate):
def func_representative_data_gen():
    # -------------------------------------------------------------------------
    # Description:
    #   remark: the following global variables are assumed to be set
    #   before the entry into this function.
    #
    #   data_train:
    #     the variable is a list of training data item
    #     where the values are used in PC training process
    #
    #   data_num:
    #     the number of data itmem in data_train
    #     it should equal to data_train.shape[0]
    #
    #   sample_rate:
    #     the amount of data_train used to serve as the representation
    #     recommended value is 0.01 , so that 1% of data_train
    #     are randomly drawn for the fine tuning.
    #     no labelling is required.
    #
    # -------------------------------------------------------------------------
    if data_num < 0:
        print('representative_data_gen() :: invalid data_num ...')
        print('exit ...')
        exit()
    if sample_rate >= 1 or sample_rate <= 0:
        print('representative_data_gen() :: invalid sample_rate ...')
        print('exit ...')
        exit()
    if sample_rate > 0.5:
        sampleNum = np.floor(sample_rate * data_num)
    else:
        sampleNum = np.ceil(sample_rate * data_num)
    sampleNum = int(sampleNum)
    # randomly sample sampleNum from data_train
    randSampleIdx = random.sample(range(data_num), sampleNum)
    for idx in randSampleIdx:
        # Model has only one input so each data point has one element.
        data_curr = data_train[idx]
        if isinstance(data_curr, str):
            representative_data_exception.append(data_curr)
            continue
        data_curr = tf.data.Dataset.from_tensor_slice(data_curr).batch(1).take(1)
        data_curr = tf.cast(data_curr, tf.float32)
        data_curr = tf.expand_dims(data_curr, 0)
        yield [data_curr]


def func_obtain_representative_data():
    if len(sys.argv) >= 4:
        representative_data_pkl_pathfile = sys.argv[3]
        if not os.path.exists(representative_data_pkl_pathfile):
            print(f'representative data {representative_data_pkl_pathfile} is not found !')
            print('exit ...')
            exit()
        with open(representative_data_pkl_pathfile, 'rb') as fp:
            data_train = pkl.load(fp)
        if type(data_train) != np.ndarray:
            print('representative data is not numpy.ndarray type...')
            print('exit ...')
            exit()
        # recall:    data_train is a list of training data item
        print(data_train[0])
        print(data_train[10])
        data_num = data_train.shape[0]
        # recall:    data_num is the number of data itmem in data_train 
        #            it should equal to data_train.shape[0]
        sample_rate  = 0.01
    else:
        print('representative data is not provided ...')
        print('exit ...')
        exit()
    return


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

     Optional argument:
     quant_opt [ FLOAT16 / DYNAMIC / INT8 / FULLINT8 / NOQNT ]:
         To provide post-training quantization type info (A-C), this serves as
         the 2nd input argument. For type C), whether using integer quantization
         or full integer quantization, it is addressed by 'INT8' or 'FULLINT8'

         If no options are specified for quant_opt,
         default value is set to 'NOQNT'

         Default: NOQNT

     representative_data_pkl_pathfile:
         If quant_opt is specified as INT8, unlabelled representative data is
         required for fine tuning.

         Specify the relative path or absolute path for function yielding.
         For e.g., "../../trial_xx_yyyy/data/training_data.pkl"
'''
        )
        exit()
    # END OF    if len(sys.argv) == 1:

    if len(sys.argv) >= 2:
        model_pathfile = sys.argv[1]
        if not os.path.exists(model_pathfile):
            print(f'model file {model_pathfile} is not found !')
            print('exit ...')
            exit()
    # END OF    if len(sys.argv) >= 2:

    if len(sys.argv) >= 3:
        quant_opt = sys.argv[2]
        quant_opt = quant_opt.lower()
        if \
                not quant_opt == __FLT16__ \
                and \
                not quant_opt == __DYRNG__ \
                and \
                not quant_opt == __INT8__ \
                and \
                not quant_opt == __FULLINT8__ :
            print(f'specified quantization option {quant_opt} is not supported !')
            print('exit ...')
            exit()
    # END OF    if len(sys.argv) >= 3:


    # -------------------------------------------------------------------------
    # Specify input model
    # -------------------------------------------------------------------------
    saved_model_dir = model_pathfile
    print(f'file {saved_model_dir} exist status check: {os.path.exists(saved_model_dir)}')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)


    # -------------------------------------------------------------------------
    # Quantization options control
    # -------------------------------------------------------------------------
    if quant_opt != __NOQNT__:
        # for case DYRNG, FLT16, INT8, FULLINT8
        # they all require optimizations set to DEFAULT
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # -----------------------------
        # case of    float16
        if quant_opt == __FLT16__:
            converter.target_spec.supported_types = [tf.float16]
        # -----------------------------
        # case of    integer 8
        elif quant_opt == __INT8__:
            func_obtain_representative_data()
            # -------------------------------------------------------------
            # provide the function argument for yielding
            # representative (unlabelled) training data
            converter.representative_dataset = func_representative_data_gen
        # -----------------------------
        # case of    full integer 8
        elif quant_opt == __FULLINT8__:
            func_obtain_representative_data()
            # -------------------------------------------------------------
            # provide the function argument for yielding
            # representative (unlabelled) training data
            converter.representative_dataset = func_representative_data_gen
            # ensure if any ops cannot be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # set input and output tensors to uint8
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:
            print('quant_opt error ...')
            print('exit ...')
            exit()
    else:
        print('quant_opt is not specified')
        print('tflite conversion takes "no quantization"')
    # END OF   if quant_opt != __NOQNT__:


    # -------------------------------------------------------------------------
    # Convert the model
    # -------------------------------------------------------------------------
    tflite_model = converter.convert()


    # -------------------------------------------------------------------------
    # Observe tflite converted model if quantization is specified
    # -------------------------------------------------------------------------
    if quant_opt != __NOQNT__:
        if quant_opt == __INT8__:
            interpreter = tf.lite.Interpreter(model_content = tflite_model)
            input_type = interpreter.get_input_details()[0]['dtype']
            output_type = interpreter.get_output_details()[0]['dtype']
            print(f'INT8 quantized conversion input data type: {input_type}')
            print(f'INT8 quantized conversion output data type: {output_type}')
            print('the data type should align with data type for PC training')
    # END OF    if quant_opt == __NOQNT__:


    # -------------------------------------------------------------------------
    # Save the model
    # -------------------------------------------------------------------------
    tmp = saved_model_dir.split('/')
    if len(tmp) > 1:
        model_serial = tmp[-2]
    else:
        model_serial = tmp[-1]
    tfmodel_tobesaved = f'{model_serial}_{quant_opt}.tflite'
    with open(tfmodel_tobesaved, 'wb') as f:
      f.write(tflite_model)

    print(representative_data_exception)
