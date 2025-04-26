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

import PIL


def get_uncompiled_model():
    input_shape = (10,9600,3)
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D((1,2), strides=(2,2)),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((1,2), strides=(2,2)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((1,2), strides=(2,2)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((1,2), strides=(2,2)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((1,2), strides=(2,2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(3, activation='softmax')
        ]
    )
    print(model.summary())
    return model


def get_compiled_model(model):
    model.compile(
        loss         = 'mse',
        optimizer    = 'adam',
        metrics      = ['accuracy']
    )
    return model


# Function to load and preprocess an image
def load_image(path):
    #raw_img = tf.io.read_file(path)
    raw_img = np.asarray(PIL.Image.open(path))
    #img = tf.image.decode_jpeg(raw_img, channels=1) # assume gray scale images
    #img = tf.cast(img, tf.float32) / 255.0 # normalize to [0,1]
    return raw_img


def gen_training_data():

    # Define the class labels and their corresponding folder name
    train_root = './data/training_data/'
    class_labels = {
        '0': os.path.join(train_root, '000/'),
        '1': os.path.join(train_root, '001/'),
        '2': os.path.join(train_root, '002/')
    }

    # Get a list of all jpg files in each class folder
    file_paths = []
    y_train = []

    for label, folder in class_labels.items():

        # Get all jpg files in the folder
        folder_files = tf.io.gfile.listdir(folder)
        jpg_files = [ f for f in folder_files if f.endswith('.jpg') ]

        # Add the full file paths and corresponding labels
        full_paths = [ os.path.join(folder, f) for f in jpg_files ]
        file_paths.extend(full_paths)

        # Add the corresponding class labels
        y_train.extend([int(label)] * len(jpg_files))

    # load all images
    x_train = []
    for file in file_paths:
        img = load_image(file)
        x_train.append(img)

    return np.array(x_train), np.array(y_train)


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
    x_train, y_train = gen_training_data()


    # ----------------------------------------
    # ADD DUMP REPRESENTATIVE DATA
    # ----------------------------------------
    output_data_train_filename = './representative_data.pkl'
    save_pickle_unlabelled_representative_data(x_train, output_data_train_filename)


    # ----------------------------------------
    # DEFINE MODEL TRAINING PARAMS
    # ----------------------------------------
    BATCH_SIZE    = 1
    EPOCHS        = 1


    # ----------------------------------------
    # MODEL FITTING
    # ----------------------------------------
    model.fit(
        x_train,
        y_train,
        callbacks        = get_tsb_ckp_cbk(),
        batch_size       = BATCH_SIZE,
        epochs           = EPOCHS
    )


