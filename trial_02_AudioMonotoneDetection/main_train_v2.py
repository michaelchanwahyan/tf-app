#!/usr/loca/bin/python3
import os
import pickle as pkl
from datetime import datetime as datetime
import numpy as np
from sklearn.utils import shuffle
import random
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import keras
from keras import layers


# -------------------------------------------------
# Problem settings
#
# the setting is that there is a microphone sampling audio data at 16000Hz.
#
# the data format is in a frame-by-frame manner, with each frame containing
# 512 samples.  each sample is an int_16 type, with positive or negative
# or zero value.
#
# the task is to detect whether there is a monotone 880Hz sinusoidal
# audio signal in the measured audio
#
# -------------------------------------------------
frameSize = 512
sampleFreq = 16000


# -------------------------------------------------
# Introduction
# -------------------------------------------------
# Detecting monotone sinusoidal signals in audio data is a common task
# in various applications, such as signal processing and audio analysis.
# Here, we focus on detecting a 880Hz sinusoidal signal from data sampled
# at 16000Hz using neural networks.


# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
# To ensure accurate detection, we must preprocess the audio data.
# Given that each frame contains 512 samples of int_16 type data,
# follow these essential preprocessing steps:

# 1. Normalization:
# Transform your int_16 samples into floating-point values between -1 and 1.
# This step is crucial for neural network training.
def normalization_layer(inputFrame):
    normalizedFrame = inputFrame / 32768.0
    return normalizedFrame

# 2. Framing:
# As each frame contains 512 samples, it spans approximately 64ms of
# audio data (512 samples / 16000Hz). Ensure your data is
# well-segmented into frames for consistent processing.


# -------------------------------------------------
# Feature Extraction
# -------------------------------------------------
# Proper feature extraction is necessary for the neural network
# to effectively detect the 880Hz signal

# 1. FFT:
# Apply FFT to convert time-domain audio data into frequency-domain representations
def fft_layer(inputFrame):
    fftFrame = tf.signal.fft(tf.cast(inputFrame, tf.complex64))
    return tf.abs(fftFrame)

# 2. Power Spectral Density (PSD):
# Compute the PSD for better frequency resolution and noise reduction
def psd_layer(inputFrame):
    psdFrame = tf.square(inputFrame) / frameSize
    return psdFrame


# -------------------------------------------------
# Neural Network Setup
# -------------------------------------------------
# Designing the neural network involves selecting an appropriate
# architecture, loss function, and training parameters

# 1. Architecture:
# Use a simple feedforward neural network for feature extraction and
# classification
def get_uncompiled_model():
    input_shape = (frameSize,)
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
#            layers.Lambda(normalization_layer),
#            layers.Lambda(fft_layer),
#            layers.Lambda(psd_layer),
            layers.Dense(128, activation='relu', input_shape=(frameSize,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ]
    )

    #print(model.weights)
    print(model.summary())
    return model

# 2. Loss Function:
# Binary cross-entropy is suitable for this binary classification problem
def get_compiled_model(model):
    model.compile(
        loss         = 'mse',
        optimizer    = 'adam',
        metrics      = ["accuracy"]
    )
    return model


def gen_training_data(sampleNum):
    x_train = []
    y_train = []

    # -------------------------------------------------
    # for background data
    # -------------------------------------------------

    # obtain the background data
    with open('background.txt' ,'r') as fp:
        lines = [ _.strip() for _ in fp.readlines() ]

    # randomly sample sampleNum / 2 entries from background data
    randSampleIdx_bkgnd = random.sample(range(len(lines)), int(sampleNum/2))

    for randIdx in randSampleIdx_bkgnd:
        x_train.append([ int(_) for _ in lines[randIdx].split(',') ])
        y_train.append([ 0 ])

    # -------------------------------------------------
    # for signal data
    # -------------------------------------------------

    # obtain the signal data
    with open('signals_monotone880Hz.txt', 'r') as fp:
        lines = [ _.strip() for _ in fp.readlines() ]

    # randomly sample sampleNum / 2 entries from signal data
    randSampleIdx_signal = random.sample(range(len(lines)), int(sampleNum/2))

    for randIdx in randSampleIdx_signal:
        x_train.append([ int(_) for _ in lines[randIdx].split(',') ])
        y_train.append([ 1 ])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train


def gen_testing_data(sampleNum):
    x_test, y_test = gen_training_data(sampleNum)
    return x_test, y_test


def save_pickle_unlabelled_representative_data(data_train, output_filename):
    if os.path.exists(output_filename):
        print(f'remove existing {output_filename} ...')
    with open(output_filename, 'wb') as fp:
        pkl.dump(data_train, fp)
    return


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
    SAMPLE_NUM = 1600
    x_train, y_train = gen_training_data(SAMPLE_NUM)
    x_train_normalized = x_train / 32768.0
    x_train_fftpsd = np.square(np.abs(np.fft.fft(x_train_normalized, axis=1))) / frameSize


    # ----------------------------------------
    # ADD DUMP REPRESENTATIVE DATA
    # ----------------------------------------
    output_data_train_filename = './representative_data.pkl'
    save_pickle_unlabelled_representative_data(x_train_fftpsd, output_data_train_filename)


    # ----------------------------------------
    # DEFINE MODEL TRAINING PARAMS
    # ----------------------------------------
    BATCH_SIZE    = 16
    EPOCHS        = 100


    # ----------------------------------------
    # MODEL FITTING
    # ----------------------------------------
    model.fit(
#        x_train,
        x_train_fftpsd,
        y_train,
        callbacks        = get_tsb_ckp_cbk(),
        batch_size       = BATCH_SIZE,
        epochs           = EPOCHS,
        validation_split = 0.2
    )


    # ----------------------------------------
    # MODEL EVALUATION
    # ----------------------------------------
    x_test, y_test = gen_testing_data(500)
    x_test_normalized = x_test / 32768.0
    x_test_fftpsd = np.square(np.abs(np.fft.fft(x_test_normalized, axis=1))) / frameSize
    results = model.evaluate(x_test_fftpsd, y_test, )
    print("test loss, test acc:", results)

    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    #model.save('./model_' + timestamp_str + '.keras')
    tf.saved_model.save(model, './model_' + timestamp_str + '/model')

