#!/usr/loca/bin/python3
import os
import tensorflow as tf
import keras
from keras import layers

_ = os.system('clear')

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
        layers.Dense(3, input_dim=2, activation="relu", name="layer1"),
        layers.Dense(2, input_dim=3, name="layer2"),
    ]
)
# Call model on a test input
#x = tf.ones((1, 2))
#y = model(x)

#print(model.weights)
print(model.summary())

