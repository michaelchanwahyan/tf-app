#!/usr/loca/bin/python3
import os
import sys
from datetime import datetime as datetime
import numpy as np
import tensorflow as tf

import keras


if __name__ == "__main__":
    _ = os.system('clear')


    if len(sys.argv) == 1:
        print('Warning: no input model specified !')
        print('Warning: Exit !')
        exit()
    if len(sys.argv) == 2:
        print(f'Info: import model: {sys.argv[1]}')
        model_path = sys.argv[1]


    # ----------------------------------------
    # LOAD TRAINED MODEL PARAMS
    # ----------------------------------------
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']

    print('output structure quick view:', infer.structured_outputs)
    output_layer_name = list(infer.structured_outputs.keys())[0]
    print('output layer name:', output_layer_name)


    # ----------------------------------------
    # MODEL INFERENSING
    # ----------------------------------------
    res = infer(
        tf.constant(
                [
                    [0.2, 0.3]
                ]
        )
    )[output_layer_name]
    print(res)


    # ----------------------------------------
    # compute results landscape
    # ----------------------------------------
    coord_test = []
    for i_ in np.linspace(-0.5, 0.5, 51):
        for j_ in np.linspace(-0.5, 0.5, 51):
            #print(i_, j_)
            coord_test.append(
                [float(i_), float(j_)]
            )
    res = infer(
        tf.constant(
            coord_test
        )
    )[output_layer_name]


