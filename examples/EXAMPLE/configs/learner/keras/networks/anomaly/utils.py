import sys
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

def base_model_FC_autoencoder(length_input, num_layers=2,encode_neurons=16,cost='MSE'):
    input = Input(shape=(length_input,))
    layer_stages = [None] * (num_layers*2+2)
    layer_stages[0] = Dense(encode_neurons*2**(num_layers), activation='relu')(input)
    for i in range(num_layers-1):
        layer_stages[i+1] = Dense(encode_neurons*2**(num_layers-(i+1)), activation='relu')(layer_stages[i])
    layer_stages[num_layers] = Dense(encode_neurons, activation='linear')(layer_stages[num_layers-1])
    for i in range(num_layers):
        layer_stages[num_layers+i+1] = Dense(encode_neurons*2**(i+1), activation='relu')(layer_stages[num_layers+i])
    layer_stages[num_layers*2+1] = Dense(length_input, activation='linear')(layer_stages[num_layers*2])
    output = autoencoder_reconstruction_error(input,layer_stages[-1],cost=cost)
    return Model(inputs=input, outputs=output)

def autoencoder_reconstruction_error(input, output, cost='MSE'):
    if cost=='MSE':
        return Lambda(mean_squared_error,name='MSE',output_shape=(1,))([input, output])
    elif cost=='MAE':
        return Lambda(mean_absolute_error, name='MAE',output_shape=(1,))([input, output])
    else:
        print('autoencoder_reconstruction_error option does not exists in models_ad_ae')
        sys.exit()

def mean_squared_error(input):
    return K.mean(K.square(input[1] - input[0]), axis=-1)

def mean_absolute_error(input):
    return K.mean(K.abs(input[1] - input[0]), axis=-1)