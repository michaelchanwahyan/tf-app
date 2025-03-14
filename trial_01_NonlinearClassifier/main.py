#!/usr/loca/bin/python3
import os
from datetime import datetime as datetime
import numpy as np
import tensorflow as tf
import keras
from keras import layers


def get_uncompiled_model():
    # Define Sequential model with 2 layers
    #
    #  INPUT LAYER          LAYER 1           LAYER 2
    #                  ------ [] ------------
    #                 /                      \
    #           ------                        []  ->
    #          /                             /
    #    ->  [] ------------     /-----------
    #                        \  /
    #                         []
    #                        /  \
    #    ->  [] ------------     \-----------
    #          \                             \
    #           ------                        []  ->
    #                 \                      /
    #                  ------ [] ------------
    #

    model = keras.Sequential(
        [
            layers.Dense(3, input_shape=(2,), activation="sigmoid", name="layer1"),
            layers.Dense(2, input_shape=(3,), activation="sigmoid", name="layer2"),
        ]
    )
    # Call model on a test input
    #x = tf.ones((1, 2))
    #y = model(x)

    #print(model.weights)
    print(model.summary())
    return model


def get_compiled_model(model):
    model.compile(
        loss         = keras.losses.CategoricalCrossentropy(),
        optimizer    = keras.optimizers.Adam(learning_rate=1e-3),
        metrics      = ["accuracy"]
    )
    return model


def gen_training_data_xy(
        sample_num,
        verbose = False
    ):
    #        REGION          REGION
    #        SYMBOL          SYMBOL
    #      [ -1 , +1 ]  ^  [ +1 , -1 ]
    #                   |
    #                   |
    #                   |
    #           --------+------->
    #                   |
    #        REGION     |    REGION
    #        SYMBOL     |    SYMBOL
    #      [ +1 , -1 ]     [ -1 , +1 ]
    x1         = np.random.rand(SAMPLE_NUM) # generate (0.5, ~N) distribution samples within [0, +1]
    x2         = np.random.rand(SAMPLE_NUM) # generate (0.5, ~N) distribution samples within [0, +1]
    x_train    = np.array(
        [ [_1, _2] for _1, _2 in zip(x1, x2) ]
    )
    if verbose:
        print(x_train)
    y_train    = np.array(
        [
            [+1,  0] if x_train_curr[0] >= 0.5 and x_train_curr[1] >= 0.5 else [ 0, +1] for x_train_curr in x_train
        ]
    )
    if verbose:
        print(y_train)
    return x_train, y_train


def gen_testing_data_xy(
        sample_num,
        verbose = False
    ):
    x_test, y_test = gen_training_data_xy(sample_num, verbose)
    return x_test, y_test


if __name__ == "__main__":
    _ = os.system('clear')


    # ----------------------------------------
    # DEFINE MODEL STRUCTURE
    # ----------------------------------------
    model = get_uncompiled_model()


    # ----------------------------------------
    # COMPILE MODEL
    # ----------------------------------------
    model = get_compiled_model(model)




    # ----------------------------------------
    # TRAINING DATA GENERATION
    # ----------------------------------------
    SAMPLE_NUM = 12800
    x_train, y_train = gen_training_data_xy(SAMPLE_NUM)


    # ----------------------------------------
    # DEFINE MODEL TRAINING PARAMS
    # ----------------------------------------
    BATCH_SIZE    = 128
    EPOCHS        = 40


    # ----------------------------------------
    # MODEL FITTING
    # ----------------------------------------
    model.fit(
        x_train,
        y_train,
        batch_size       = BATCH_SIZE,
        epochs           = EPOCHS,
        validation_split = 0.1
    )


    # ----------------------------------------
    # MODEL EVALUATION
    # ----------------------------------------
    x_test, y_test = gen_testing_data_xy(1280)
    results = model.evaluate(x_test, y_test, )
    print("test loss, test acc:", results)


    x_predict = np.array([[0.3, 0.3],
                          [0.7, 0.7]])
    prediction = model.predict(x_predict)
    print(x_predict, prediction)

    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    model.save('./model_' + timestamp_str + '.keras')

