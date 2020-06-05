import tensorflow.keras.backend as K

def example_mae(ref, pred):
    custom_loss = K.mean(K.abs(ref-pred))
    return custom_loss