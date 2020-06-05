import tensorflow.keras as kr
from tensorflow.keras.layers import Input, Dense, Lambda

def model(input_shape):
    model = kr.models.Sequential()
    model.add(Dense(64, activation='relu', input_dim=(input_shape[0])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model