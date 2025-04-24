#!/bin/bash

MODEL_DIR=../trial_02_AudioMonotoneDetection
MODEL_NAME=
REPRESENTATIVE_PKL=representative_data.pkl

python3    ../common/main_convert_model.py    $MODEL_DIR/$MODEL_NAME
python3    ../common/main_convert_model.py    $MODEL_DIR/$MODEL_NAME    DYNAMIC
python3    ../common/main_convert_model.py    $MODEL_DIR/$MODEL_NAME    FLOAT16
python3    ../common/main_convert_model.py    $MODEL_DIR/$MODEL_NAME    INT8        $MODEL_DIR/$REPRESENTATIVE_PKL
python3    ../common/main_convert_model.py    $MODEL_DIR/$MODEL_NAME    FULLINT8    $MODEL_DIR/$REPRESENTATIVE_PKL
