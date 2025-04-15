#!/usr/loca/bin/python3
import os
from datetime import datetime as datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
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
            layers.Input(shape=(2,)),
            layers.Dense(3, activation="tanh", name="layer1"),
            layers.Dense(2, activation="tanh", name="layer2"),
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
        loss         = 'mse',
        optimizer    = 'adam',
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
    x1         = np.random.rand(SAMPLE_NUM) - 0.5 # generate (0.5, ~N) distribution samples within [0, +1]
    x2         = np.random.rand(SAMPLE_NUM) - 0.5 # generate (0.5, ~N) distribution samples within [0, +1]
    x_train    = np.array(
        [ [_1, _2] for _1, _2 in zip(x1, x2) ]
    )
    if verbose:
        print(x_train)
    y_train    = np.array(
        [
            [+1, -1] if x_train_curr[0]*x_train_curr[1] >= 0.0 else [-1, +1] for x_train_curr in x_train
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


def get_tsb_ckp_cbk():
    # Load Tensorboard callback
    tensorboard = TensorBoard(
      log_dir=os.path.join(os.getcwd(), "logs"),
      histogram_freq=1,
      write_images=True
    )

    # Save a model checkpoint after every epoch
    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), "model_checkpoint"),
        save_freq="epoch"
    )

    # Add callbacks to list
    callbacks = [
      tensorboard,
      checkpoint
    ]
    return callbacks


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
    SAMPLE_NUM = 128000
    x_train, y_train = gen_training_data_xy(SAMPLE_NUM)
    print('x train mean:',x_train.mean(axis=0))
    print('y train mean:',y_train.mean(axis=0))
    print('x train std:',x_train.std(axis=0))
    print('y train std:',y_train.std(axis=0))


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
        callbacks        = get_tsb_ckp_cbk(),
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


    x_predict = np.array([[-0.5,-0.5],
                          [ 0.5,-0.5],
                          [-0.5, 0.5],
                          [ 0.5, 0.5]])
    prediction = model.predict(x_predict)
    print(x_predict, prediction)

    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    #model.save('./model_' + timestamp_str + '.keras')
    tf.saved_model.save(model, './model_' + timestamp_str + '/model')

