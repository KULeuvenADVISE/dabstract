import keras as kr
from keras.layers import Input, Dense, Lambda

def model(input_shape):
    model = kr.models.Sequential()
    model.add(Dense(1, activation='sigmoid',input_dim=input_shape[0]))
    return model