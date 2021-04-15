import numpy as np
import os
from scipy.io.wavfile import write as wav_write
from dabstract.dataprocessor.processing_chain import ProcessingChain

def test_WavDatareader():
    """Test WavDatareader"""
    from dabstract.dataprocessor.processors import WavDatareader

    # create a temporary wav file
    samplerate = 44100;
    fs = 100
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np.sin(2. * np.pi * fs * t)
    wav_write("tmp.wav", samplerate, data.astype(np.int16))

def test_Normalizer():
    """Test Normalizer"""
    from dabstract.dataprocessor.processors import Normalizer

    ## test with 2D data and norm reference is axis -1
    # data init
    fit_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=float)
    example = fit_data[0]
    # MinMax
    nr = Normalizer(type='minmax')
    nr.fit(fit_data)
    output = nr(example)
    assert all(output == np.array([0,0,0,0,0], dtype=float))
    # std
    nr = Normalizer(type='standard')
    nr.fit(fit_data)
    output = nr(example)
    np.testing.assert_allclose(output, np.array([-1.22474487, -1.22474487, -1.22474487, -1.22474487, -1.22474487], dtype=float))

    ## test with 3D data and norm reference is axis -1
    # data init
    fit_data2 = np.array([[[0,1,5],[0,1,2]],
                          [[0,2,2],[0,2,2]],
                          [[3,3,2],[0,3,2]]], dtype=float)
    example = fit_data2[2]
    # MinMax
    nr = Normalizer(type='minmax')
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[1., 1., 0.],[0., 1., 0.]]))
    # std
    nr = Normalizer(type='standard')
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[ 2.23606798,  1.22474487, -0.4472136 ],[-0.4472136 ,  1.22474487, -0.4472136 ]]))

    ## test with 3D data and norm reference is axis 0
    # data init
    fit_data2 = np.array([[[0,1,5],[0,1,2]],
                          [[0,2,2],[0,2,2]],
                          [[3,3,2],[0,3,2]]], dtype=float)
    example = fit_data2[2]
    # MinMax
    nr = Normalizer(type='minmax', axis = 0)
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[0.6, 0.6, 0.4],[0., 1.,2/3]]))
    # std
    nr = Normalizer(type='standard', axis = 0)
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[0.67082039,0.67082039,0.],[-1.26491106,1.58113883,0.63245553]]))

    ## test with 3D data and norm reference is all axis
    # data init
    fit_data2 = np.array([[[0,1,5],[0,1,2]],
                          [[0,2,2],[0,2,2]],
                          [[3,3,2],[0,3,2]]], dtype=float)
    example = fit_data2[2]
    # MinMax
    nr = Normalizer(type='minmax', axis = 'all')
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[1., 1., 0.],[0., 1., 0.]]))
    # std
    nr = Normalizer(type='standard', axis = 'all')
    nr.fit(fit_data2)
    output = nr(example)
    np.testing.assert_allclose(output,np.array([[1.41421356,1.22474487,-0.70710678],[0.,1.22474487,0.]]))

if __name__ == "__main__":
    test_WavDatareader()
    test_Normalizer()