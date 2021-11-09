import numpy as np

from keras.layers import Dense, Flatten, Convolution1D, Dropout, concatenate
from keras import losses, Input, Model

activation1 = 'relu'
activation2 = 'softmax'
dropout = 0.5

def creat_Mul_model():
    # Mono_Mer
    towerA_input_seq = Input(shape=(81, 4))
    towerA_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerA_input_seq)
    towerA_1 = Dropout(0.5)(towerA_1)

    towerA_2 = Convolution1D(filters=50, kernel_size=1, activation=activation1)(towerA_1)
    towerA_2 = Dropout(0.5)(towerA_2)

    towerA_3 = Convolution1D(filters=24, kernel_size=1, activation=activation1)(towerA_2)
    towerA_3 = Dropout(0.5)(towerA_3)

    towerA = Flatten()(towerA_3)

    # Tri_Mer
    towerB_input_seq = Input(shape=(79, 64))
    towerB_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerB_input_seq)
    towerB_1 = Dropout(0.5)(towerB_1)

    towerB_2 = Convolution1D(filters=40, kernel_size=1, activation=activation1)(towerB_1)
    towerB_2 = Dropout(0.5)(towerB_2)

    towerB_3 = Convolution1D(filters=24, kernel_size=1, activation=activation1)(towerB_2)
    towerB_3 = Dropout(0.5)(towerB_3)
    towerB = Flatten()(towerB_3)

    # Di_nucleotide Structural Prop
    towerC_input_seq = Input(shape=(80, 90))
    towerC_1 = Convolution1D(filters=256, kernel_size=1, activation=activation1)(towerC_input_seq)
    towerC_1 = Dropout(0.5)(towerC_1)

    towerC_2 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerC_1)
    towerC_2 = Dropout(0.5)(towerC_2)

    towerC_3 = Convolution1D(filters=64, kernel_size=1, activation=activation1)(towerC_2)
    towerC_3 = Dropout(0.5)(towerC_3)

    towerC_4 = Convolution1D(filters=32, kernel_size=1, activation=activation1)(towerC_3)
    towerC_4 = Dropout(0.5)(towerC_4)
    towerC = Flatten()(towerC_4)

    # Tri_nucleotide Structural Prop
    towerD_input_seq = Input(shape=(79, 12))
    towerD_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerD_input_seq)
    towerD_1 = Dropout(0.5)(towerD_1)

    towerD_2 = Convolution1D(filters=64, kernel_size=1, activation=activation1)(towerD_1)
    towerD_2 = Dropout(0.5)(towerD_2)

    towerD_3 = Convolution1D(filters=32, kernel_size=1, activation=activation1)(towerD_2)
    towerD_3 = Dropout(0.5)(towerD_3)
    towerD = Flatten()(towerD_3)

    # Concatenate
    concatenate1 = concatenate([towerA, towerB, towerC, towerD], axis=-1)
    concatenate1 = Dropout(0.5)(concatenate1)

    # Dense
    Dense1 = Dense(units=128, activation=activation1)(concatenate1)
    Dense1 = Dropout(0.5)(Dense1)

    Dense2 = Dense(units=64, activation=activation1)(Dense1)

    output = Dense(units=2, activation=activation2)(Dense2)

    model = Model(inputs=[towerA_input_seq, towerB_input_seq, towerC_input_seq, towerD_input_seq], outputs=[output])
    return model

def visualizing_model():
    # Mono_Mer
    towerA_input_seq = Input(shape=(81, 4))
    towerA_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerA_input_seq)
    towerA_1 = Dropout(0.5)(towerA_1)

    towerA_2 = Convolution1D(filters=50, kernel_size=1, activation=activation1)(towerA_1)
    towerA_2 = Dropout(0.5)(towerA_2)

    towerA_3 = Convolution1D(filters=24, kernel_size=1, activation=activation1)(towerA_2)
    towerA_3 = Dropout(0.5)(towerA_3)

    towerA = Flatten()(towerA_3)

    # Tri_Mer
    towerB_input_seq = Input(shape=(79, 64))
    towerB_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerB_input_seq)
    towerB_1 = Dropout(0.5)(towerB_1)

    towerB_2 = Convolution1D(filters=40, kernel_size=1, activation=activation1)(towerB_1)
    towerB_2 = Dropout(0.5)(towerB_2)

    towerB_3 = Convolution1D(filters=24, kernel_size=1, activation=activation1)(towerB_2)
    towerB_3 = Dropout(0.5)(towerB_3)

    towerB = Flatten()(towerB_3)

    # Di_nucleotide Structural Prop
    towerC_input_seq = Input(shape=(80, 90))
    towerC_1 = Convolution1D(filters=256, kernel_size=1, activation=activation1)(towerC_input_seq)
    towerC_1 = Dropout(0.5)(towerC_1)

    towerC_2 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerC_1)
    towerC_2 = Dropout(0.5)(towerC_2)

    towerC_3 = Convolution1D(filters=64, kernel_size=1, activation=activation1)(towerC_2)
    towerC_3 = Dropout(0.5)(towerC_3)

    towerC_4 = Convolution1D(filters=32, kernel_size=1, activation=activation1)(towerC_3)
    towerC_4 = Dropout(0.5)(towerC_4)

    towerC = Flatten()(towerC_4)

    # Tri_nucleotide Structural Prop
    towerD_input_seq = Input(shape=(79, 12))
    towerD_1 = Convolution1D(filters=128, kernel_size=1, activation=activation1)(towerD_input_seq)
    towerD_1 = Dropout(0.5)(towerD_1)

    towerD_2 = Convolution1D(filters=64, kernel_size=1, activation=activation1)(towerD_1)
    towerD_2 = Dropout(0.5)(towerD_2)

    towerD_3 = Convolution1D(filters=32, kernel_size=1, activation=activation1)(towerD_2)
    towerD_3 = Dropout(0.5)(towerD_3)

    towerD = Flatten()(towerD_3)

    # Concatenate
    concatenate1 = concatenate([towerA, towerB, towerC, towerD], axis=-1)
    concatenate1 = Dropout(0.5)(concatenate1)

    # Dense
    Dense1 = Dense(units=128, activation=activation1)(concatenate1)
    Dense1 = Dropout(0.5)(Dense1)
    Dense2 = Dense(units=64, activation=activation1)(Dense1)

    output = Dense(units=2, activation=activation2)(Dense2)

    model = Model(inputs=[towerA_input_seq, towerB_input_seq, towerC_input_seq, towerD_input_seq], outputs=[towerA, towerB, towerC, towerD])
    return model

def create_trained_model(model1):
    model = creat_Mul_model()
    model.load_weights(model1)
    return model


def create_truncated_model(trained_model):
    model = visualizing_model()
    from keras.utils.conv_utils import convert_kernel
    for i, layer in enumerate(model.layers):
        w = trained_model.layers[i].get_weights()
        if len(w) == 0:
            continue
        else:
            layer.set_weights(trained_model.layers[i].get_weights())
    model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model
