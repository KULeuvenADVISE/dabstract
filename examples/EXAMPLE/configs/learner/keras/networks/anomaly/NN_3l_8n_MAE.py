from .utils import base_model_FC_autoencoder

def model(input_shape):
    model = base_model_FC_autoencoder(input_shape[0], num_layers=3, encode_neurons=8, cost='MAE')
    return model