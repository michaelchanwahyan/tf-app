#!/usr/loca/bin/python3
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers

_ = os.system('clear')

# ----------------------------------------
# DEFINE MODEL STRUCTURE
# ----------------------------------------

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


# ----------------------------------------
# TRAINING DATA GENERATION
# ----------------------------------------
SAMPLE_NUM = 12800
x1         = np.random.rand(SAMPLE_NUM) # generate (0.5, ~N) distribution samples within [0, +1]
x2         = np.random.rand(SAMPLE_NUM) # generate (0.5, ~N) distribution samples within [0, +1]
x_train    = np.array(
    [ [_1, _2] for _1, _2 in zip(x1, x2) ]
)
print(x_train)
y_train    = np.array(
    [
        [+1,  0] if x_train_curr[0] >= 0.5 and x_train_curr[1] >= 0.5 else [ 0, +1] for x_train_curr in x_train
    ]
)
print(y_train)

# ----------------------------------------
# DEFINE MODEL TRAINING PARAMS
# ----------------------------------------
model.compile(
    loss         = keras.losses.CategoricalCrossentropy(),
    optimizer    = keras.optimizers.Adam(learning_rate=1e-3),
    metrics      = ["accuracy"]
)

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

