import numpy as np
import sklearn.preprocessing as pp
from soundfile import read as read_wav
import sys
import os
import scipy
import scipy.signal as signal

from dabstract.utils import listnp_combine, flatten_nested_lst
from dabstract.dataprocessor import processor

class WavDatareader(processor):
    def __init__(self, format=None, select_channel=None, fs=None, range=None, dtype=None, **kwargs):
        self.force_format = format
        self.select_channel = select_channel
        self.fs = fs
        self.range = range
        self.dtype = dtype
    def process(self, file, **kwargs):
        # get read params
        args = dict()
        if self.range is not None:
            args.update({'start': self.range[0],
                         'stop': self.range[1]})
        if 'range' in kwargs:
            args.update({'start': kwargs['range'][0],
                         'stop': kwargs['range'][1]})
        if hasattr(self,'dtype'):
            args.update({'dtype': self.dtype})

        # read
        data, fs = read_wav(file, **args)
        if self.fs is not None:
            if fs != self.fs:
                print('Input fs and provided fs different. Downsampling not supported currently.')
                sys.exit()

        # data selection
        if self.select_channel is not None:
            data = data[:, self.select_channel]

        # updata self info
        return data, {'fs': fs}

class NumpyDatareader(processor):
    def __init__(self, format=None,  range=None, **kwargs):
        self.force_format = format
        self.range = range

    def process(self, file, **kwargs):
        # get read params
        args = dict()
        if self.range is not None:
            args.update({'range': self.range})
        if 'range' in kwargs:
            args.update({'range': kwargs['range']})

        if 'range' in args:
            data = np.load(file, mmap_mode='r')
            data = data[args['range'][0]:args['range'][1],:]
        else:
            data = np.load(file)
        return data, {}

class Normalizer(processor):
    def __init__(self, type=None, init_subsample=None, feature_range=[0,1], **kwargs):
        if type is None:
            print('Specify normalization type in dp.py/Normalizer')
            sys.exit()
        self.type = type
        self.feature_range = feature_range
        self.init_subsample = init_subsample

    def fit(self, data, info):
        if self.type == 'minmax':
            self.scaler = pp.MinMaxScaler(feature_range=self.feature_range)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) == 3:  # based on the assumption that we will at most read 3D data
                data = flatten_nested_lst(data)
            self.scaler.fit(data)
        elif self.type == 'standard':
            self.scaler = pp.StandardScaler()
            if len(data.shape) >= 3:  # based on the assumption that we will at most read 3D data
                data = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
            self.scaler.fit(data)

    def process(self, data, **kwargs):
        if (self.type == 'minmax') | (self.type == 'standard'):
            if len(data.shape)<=1:
                data = data.reshape(1, -1)
                data = self.scaler.transform(data)
                return data.reshape(-1), {}
            elif len(data.shape) == 2:
                 return self.scaler.transform(data), {}
            elif len(data.shape) == 3:
                for k in range(data.shape[0]):
                     data[k] = self.scaler.transform(data[k])
                return data, {}
            elif len(data.shape) == 4:
                for k in range(data.shape[0]):
                    for i in range(data[k].shape[2]):
                        data[k, :, :, i] = self.scaler.transform(data[k, :, :, i])
                return data, {}
        else:
            print('Not supported.')
            sys.exit()

    def inv_process(self, data, **kwargs):
        if (self.type == 'minmax') | (self.type == 'standard'):
            if len(data.shape)<=1:
                data = data.reshape(1, -1)
                data = self.scaler.inverse_transform(data)
                return data.reshape(-1)
            elif len(data.shape) == 2:
                return self.scaler.inverse_transform(data)
            elif len(data.shape) == 3:
                for k in range(data.shape[0]):
                    data[k] = self.scaler.inverse_transform(data[k])
            elif len(data.shape) == 4:
                for k in range(data.shape[0]):
                    for i in range(data[k].shape[2]):
                        data[k, :, :, i] = self.scaler.inverse_transform(data[k, :, :, i])
            else:
                print('Not supported.')
                sys.exit()
        else:
            print('Not supported.')
            sys.exit()

        return data

class Fixed_Normalizer(processor):
    def __init__(self, **kwargs):
        self.type = kwargs['type']

    def process(self, data, **kwargs):
        if self.type == 'uint16':
            data = data / 2 ** 16
        elif self.type == 'int16':
            data = data / (pow(2, 15) - 1)
        elif self.type == 'wav_2_01':
            data = (data + 1) / 2
        else:
            print('Not supported.')
            sys.exit()

        return data, {}

    def inv_process(self, data):
        if self.type == 'uint16_scaler':
            data = data * 2 ** 16
        elif self.type == 'int16':
            data = data * (pow(2, 15) - 1)
        else:
            print('Not supported.')
            sys.exit()
        return data

class Framing(processor):
    def __init__(self, windowsize = None, stepsize = None, window_func = 'hamming', axis=-1, **kwargs):
        # inits
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.window_func = Windowing(window_func=window_func,axis=axis)
        self.axis = axis
        if 'fs' in kwargs:
            self.fs = kwargs['fs']

    def process(self, data, **kwargs):
        # inits
        if 'time_step' in kwargs:
            if kwargs['time_step'] is not None:
                fs = 1/kwargs['time_step']
        elif 'fs' in kwargs:
            fs = kwargs['fs']
        elif hasattr(self,'fs'):
            fs = self.fs
        else:
            assert 0, "fs not provided in Framing"
        assert self.windowsize is not None
        frame_length = int(round(self.windowsize * fs))
        if self.stepsize is None:
            frame_step = 1
        else:
            frame_step = int(round(self.stepsize * fs))
        if self.axis == -1:
            axis = len(data.shape) - 1
        else:
            axis = self.axis
        signal_length = data.shape[axis]

        # segment
        num_frames = int(np.floor(((signal_length - (frame_length - 1) - 1) / frame_step) + 1))
        assert num_frames>0, 'num of frames is 0 in Framing()'
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = np.take(data, indices, axis=axis)
        data_lo = np.take(data, np.setdiff1d(np.arange(signal_length), indices), axis=axis)

        # window fct
        self.window_func.axis = axis+1
        frames = self.window_func.process(frames)[0]

        # return
        return frames, {'time_step': self.stepsize} if axis==0 else dict()

class Windowing(processor):
    def __init__(self, axis=-1, window_func='hamming', **kwargs):
        self.axis = axis
        self.window_func = window_func

    def process(self, data, **kwargs):
        # init
        if self.axis == -1:
            axis = len(data.shape) - 1
        else:
            axis = self.axis
        # hamming
        if self.window_func == 'hamming':
            hw = np.hamming(np.shape(data)[axis]).astype(float)
            data *= np.reshape(hw, [(data.shape[k] if k == axis else 1) for k in range(len(data.shape))])
        elif self.window_func == 'none' or self.window_func == 'None':
            pass
        elif self.window_func is None:
            pass
        else:
            print('No other windows supported.')
            sys.exit()

        # return
        return data, dict()

class FFT(processor):
    def __init__(self, type='real', Nfft = 'nextpow2', format='magnitude', dc_reset=False, axis=-1,**kwargs):
        self.format = format
        self.dc_reset = dc_reset
        self.axis = axis
        self.Nfft = Nfft
        self.type = type

    def process(self, data, **kwargs):
        # do fft
        if self.Nfft=='nextpow2':
            NFFT = 2 ** np.ceil(np.log2(np.shape(data)[self.axis]))
        elif self.Nfft=='original':
            NFFT = np.shape(data)[self.axis]

        if self.type=='real':
            data = np.fft.rfft(data, n=int(NFFT), axis=self.axis)
        elif self.type=='full':
            data = np.fft.fft(data, n=int(NFFT), axis=self.axis)

        # agg complex
        if self.format=='magnitude':
            data = np.absolute(data)
        elif self.format=='power':
            data = (1.0 / NFFT) * (np.absolute(data) ** 2)
        elif self.format=='split':
            data = np.concatenate((np.real(data),np.imag(data)),axis=self.axis)

        # remove dc
        if self.dc_reset:
            # do sel slicing
            if self.axis == -1:
                axis = len(np.shape(data)) - 1
            else:
                axis = self.axis
            sel_tuple = tuple([(slice(0, data.shape[k]) if k != axis else 0) for k in range(len(data.shape))])
            data[sel_tuple] = 0

        return data, {}

class Filterbank(processor):
    def __init__(self, n_bands=40, scale='linear', Nfft = 'nextpow2', fmin=0, fmax=np.Inf, axis=-1, **kwargs):
        self.n_bands = n_bands
        self.scale = scale
        self.axis = axis
        self.fmin = fmin
        self.fmax = fmax
        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        self.Nfft = Nfft

    def process(self,data,**kwargs):
        # inits
        if 'fs' in kwargs:
            fs = kwargs['fs']
            if self.fmax=='half_fs':
                fmax=fs/2
        elif hasattr(self,'fs'):
            fs = self.fs
        else:
            print('No fs given in Filterbank()')
            sys.exit()

        if self.Nfft == 'nextpow2':
            NFFT = np.shape(data)[self.axis] * 2 - 2
        elif self.Nfft == 'original':
            NFFT = np.shape(data)[self.axis]

        low_freq = self.fmin
        high_freq = np.min((fs / 2, self.fmax))

        # create filterbank
        if not (hasattr(self, 'fbank')):
            if self.scale == 'mel':
                # Define the Mel frequency of high_freq and low_freq
                low_freq_mel = (2595 * np.log10(1 + low_freq / 700))
                high_freq_mel = (2595 * np.log10(1 + high_freq / 700))
                # Define the start Mel frequencies, start frequencies and start bins
                start_freq_mel = low_freq_mel + np.arange(0, self.n_bands, 1) / (self.n_bands + 1) * (high_freq_mel - low_freq_mel)
                start_freq_hz = 700 * (10 ** (start_freq_mel / 2595) - 1)
                # Define the stop Mel frequencies, start frequencies and start bins
                stop_freq_mel = low_freq_mel + np.arange(2, self.n_bands + 2, 1) / (self.n_bands + 1) * (high_freq_mel - low_freq_mel)
                stop_freq_hz = 700 * (10 ** (stop_freq_mel / 2595) - 1)
            elif self.scale == 'linear':
                # linear spacing
                hz_points = np.linspace(low_freq, high_freq, self.n_bands + 2)
                start_freq_hz = hz_points[0:-2]
                stop_freq_hz = hz_points[2:]

            # get bins
            start_bin = np.round(NFFT / fs * start_freq_hz)
            stop_bin = np.round(NFFT / fs * stop_freq_hz)
            # The middle bins of the filters are the start frequencies of the next filter.
            middle_bin = np.append(start_bin[1:], stop_bin[-2])
            # Compute the width of the filters
            tot_len = stop_bin - start_bin + 1
            low_len = middle_bin - start_bin + 1
            high_len = tot_len - low_len + 1
            # Allocate the empty filterbank
            self.fbank = np.zeros((self.n_bands, int(np.floor(NFFT / 2 + 1))))
            # Compute the filter weights matrix
            for m in range(1, self.n_bands + 1):
                weights_low = np.arange(1, low_len[m - 1] + 1) / (low_len[m - 1])
                for k in range(0, int(low_len[m - 1])):
                    self.fbank[m - 1, int(start_bin[m - 1] + k)] = weights_low[k]
                weights_high = np.arange(high_len[m - 1], 0, -1) / (high_len[m - 1])
                for k in range(0, int(high_len[m - 1])):
                    self.fbank[m - 1, int(middle_bin[m - 1] + k)] = weights_high[k]

        # Apply the mel/linear warping
        filter_banks = np.dot(data, self.fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability

        return filter_banks, {}

class Logarithm(processor):
    def __init__(self, type='base10', **kwargs):
        self.type = type
    def process(self, data, **kwargs):
        if self.type == 'base10':
            return 20*np.log10(data), {}
        elif self.type == 'natural':
            return np.log(data), {}
    def inv_process(self, data, **kwargs):
        if self.type == 'base10':
            return (10**data)/20, {}
        elif self.type == 'natural':
            return np.exp(data), {}

class Aggregation(processor):
    def __init__(self, methods = ['mean','std'], axis = 0, combine=None, combine_axis=None):
        self.methods = methods
        self.axis = axis
        self.combine = combine
        if combine_axis is None:
            self.combine_axis = axis
        else:
            self.combine_axis = combine_axis

    def process(self, data, **kwargs):
        # aggregate data
        tmp = [None] * len(self.methods)
        for k in range(len(self.methods)):
            if self.methods[k] == 'mean':
                tmp[k] = np.mean(data, axis=self.axis)
            elif self.methods[k] == 'std':
                tmp[k] = np.std(data, axis=self.axis)
            elif self.methods[k] == 'kurtosis':
                tmp[k] = scipy.stats.kurtosis(data, axis=self.axis)
            else:
                print('Aggregation method not supported')
                sys.exit()
        output = (listnp_combine(tmp, method=self.combine, axis=self.combine_axis) if self.combine is not None else tmp)

        return output, {'time_step': 0} if self.axis == 0 else dict()

class FIR_filter(processor):
    def __init__(self, type=type, f=None, taps=None, axis=1, fs=None, window='hamming'):
        self.type = type
        self.f = f
        self.taps = taps
        self.taps |= 1 #make uneven
        self.axis = axis
        self.window = window
        self.fs = fs

    def get_filter(self,fs):
        if self.type=='bandstop':
            self.filter = signal.firwin(self.taps, self.f, window=self.window,fs=fs)
        elif self.type=='bandpass':
            self.filter = signal.firwin(self.taps, self.f, window=self.window,fs=fs,pass_zero=False)
        elif self.type == 'highpass':
            self.filter = signal.firwin(self.taps, self.f, window=self.window, fs=fs, pass_zero=False)
        elif self.type == 'lowpass':
            self.filter = signal.firwin(self.taps, self.f, window=self.window, fs=fs)

    def process(self, data, **kwargs):
        if not hasattr(self,'filter'):
            if 'fs' in kwargs:
                self.get_filter(kwargs['fs'])
            elif hasattr(self,'fs'):
                self.get_filter(self.fs)
            else:
                raise Exception("Sampling frequency should be provided to FIR_filter as init or passed on the process()")
        return signal.lfilter(self.filter, 1.0, data, axis=self.axis), {}

class AD_std(processor):
    def __init__(self, windowsize=0.0025, stepsize=0.0025, threshold=4,forgetting_factor=0.9, init_subsample=0.01):
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.threshold = threshold
        self.forgetting_factor = forgetting_factor
        self.init_subsample = init_subsample
        self.framer = Framing(windowsize=self.windowsize, stepsize=self.stepsize)
    def process(self, data, **kwargs):
        data_fr = self.framer.process(data,**kwargs)[0]
        max_val = np.max(np.abs(data_fr), axis=1)
        det = max_val > self.max_mean + (self.threshold * self.max_std)
        output = np.stack((np.where(det)[0], max_val[det == True]), axis=1)
        if len(np.where(det)[0])==0:
            output = np.expand_dims(np.array([np.nan, np.nan]),0)
        return np.expand_dims(output,0), {}

    def fit(self, data, info, **kwargs):
        max_mean, max_std = np.zeros(len(data)),  np.zeros(len(data))
        for k in range(len(data)):
            data_fr = self.framer.process(data[k],**info[k])[0]
            tmp = np.max(np.abs(data_fr), axis=1)
            max_mean[k], max_std[k] = np.mean(tmp), np.std(tmp)
        self.max_mean, self.max_std = np.mean(max_mean), np.mean(max_std)


class expand_dims(processor):
    def __init__(self, axis = -1):
        self.axis = axis

    def process(self, data, **kwargs):
        data = np.expand_dims(data, axis=self.axis)
        return data, {}

class MeanEnergy(processor):
    def process(self, data, **kwargs):
        return np.reshape(np.nanmean(data**2,axis=tuple(np.arange(0,len(data.shape)))),[1,1]), {}

class none(processor):
    pass
