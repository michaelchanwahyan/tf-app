#!/usr/loca/bin/python3
import os
import pickle as pkl
from datetime import datetime as datetime
import numpy as np
from sklearn.utils import shuffle
from scipy.io import wavfile
import random
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import keras
from keras import layers



def get_raw_audio_data():
    # each 30 second duration
    sample_rate, surfaceA_audio = wavfile.read('./data/lowpassed_5000Hz/psockit_surface_circling_audio.wav')
    surfaceA_audio = surfaceA_audio[:(30 * sample_rate)]
    _          , surfaceB_audio = wavfile.read('./data/lowpassed_5000Hz/table_cover_plastic_surface_circling_audio.wav')
    surfaceB_audio = surfaceB_audio[:(30 * sample_rate)]
    _          , airnoise_audio = wavfile.read('./data/lowpassed_5000Hz/background_audio.wav')
    airnoise_audio = airnoise_audio[:(30 * sample_rate)]
    return sample_rate, surfaceA_audio, surfaceB_audio, airnoise_audio


def get_stft_from_audio(inAudio, stft_fftNum=128, stft_progression=800):
    # -------------------------------------------------------------------------
    #
    # STFT Spectrum  _
    #            ^  | |
    #            |  | |
    #  stft fft# |  | |
    #     128    |  | |
    #            |  | |
    #            |  |_|_________________________
    #            v  |_|_________________________|
    #                               _
    #                              | |
    #                              | |
    #                              | |      ==>
    #                <----         | |
    #                 progression  | |
    #                        ----> |_|_________________________
    #                              |_|_________________________|
    #
    #
    # -------------------------------------------------------------------------

    L = len(inAudio)
    if L < stft_fftNum:
        print(f'error: L ({L}) cannot be lower than stft_fftNum ({stft_fftNum})')
        print('exit ...')
        exit()

    # ------------------------------------------------------
    # pilot check STFT Spectrum column number
    # ------------------------------------------------------
    stft_col_cnt = 0
    sIdx_start = 0 # sample index (start)
    sIdx_end = sIdx_start + stft_progression
    while sIdx_end < len(inAudio):
        stft_col_cnt += 1
        sIdx_start = sIdx_end
        sIdx_end += stft_progression
    print(f'inAudio STFT Spectrum contains {stft_col_cnt} column given progression={stft_progression}')

    # take stft_col_cnt for STFT Spectrum data size init
    X = np.zeros((stft_fftNum, stft_col_cnt)) # vertical dimension of X reuses fftNum because we need real and imag part

    # segment one second segment from inAudio
    stft_col_idx = 0
    sIdx_start = 0 # sample index (start)
    sIdx_end = sIdx_start + stft_progression
    while sIdx_end < len(inAudio):
        # obtain audio segment for current FFT
        x_curr = inAudio[ sIdx_start : sIdx_start + stft_fftNum ]

        X_curr = np.fft.fft(x_curr)
        X_curr = X_curr[:int(stft_fftNum/2)]
        X[:int(stft_fftNum/2), stft_col_idx] = np.real(X_curr)
        X[int(stft_fftNum/2):, stft_col_idx] = np.imag(X_curr)

        stft_col_idx += 1
        sIdx_start = sIdx_end
        sIdx_end += stft_progression
    # END OF while sIdx_end < len(inAudio):

    return X


def gen_training_data(
    sampleNum_per_class,
    surfaceA_STFT,
    surfaceB_STFT,
    airnoise_STFT
):
    [M, N] = airnoise_STFT.shape
    print(f'MxN : {M} x {N}')

    x_train = []
    y_train = []

    # randomly sample sampleNum_per_class entries
    randSampleIdx = random.sample(range(N-M), sampleNum_per_class)

    for randIdx in randSampleIdx:
        # airnoise data
        x_train.append(airnoise_STFT[:, randIdx : randIdx + M])
        y_train.append([1,0,0])
        # surfaceA data
        x_train.append(surfaceA_STFT[:, randIdx : randIdx + M])
        y_train.append([0,1,0])
        # surfaceB data
        x_train.append(surfaceB_STFT[:, randIdx : randIdx + M])
        y_train.append([0,0,1])

    x_train = np.array(x_train) / 65536.0
    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train


def get_uncompiled_model():
    input_shape = (128, 128, 1)
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D( 32, (3,3), activation='relu', padding='same'),
            layers.Conv2D( 32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2), strides=(2,2)),
            layers.Conv2D( 64, (3,3), activation='relu', padding='same'),
            layers.Conv2D( 64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2), strides=(2,2)),
            layers.Flatten(),
            layers.Dense( 32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(  3, activation='relu'),
        ]
    )
    print(model.summary())
    return model


def get_compiled_model(model):
    model.compile(
        loss         = 'mse',
        optimizer    = 'adam',
        metrics      = ["accuracy"]
    )
    return model


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


if __name__ == '__main__':

    # read audio dataset
    sample_rate, surfaceA_audio, surfaceB_audio, airnoise_audio = get_raw_audio_data()

    # obtain STFT (real-imag) from audio
    surfaceA_STFT = get_stft_from_audio(surfaceA_audio)
    surfaceB_STFT = get_stft_from_audio(surfaceB_audio)
    airnoise_STFT = get_stft_from_audio(airnoise_audio)

    # training data creation
    sampleNum_per_class = 1000
    x_train, y_train = \
        gen_training_data(
            sampleNum_per_class,
            surfaceA_STFT,
            surfaceB_STFT,
            airnoise_STFT,
        )

    model = get_uncompiled_model()
    model = get_compiled_model(model)

    BATCH_SIZE    = 1
    EPOCHS        = 100

    model.fit(
        x_train,
        y_train,
        callbacks        = get_tsb_ckp_cbk(),
        batch_size       = BATCH_SIZE,
        epochs           = EPOCHS,
        validation_split = 0.1
    )

