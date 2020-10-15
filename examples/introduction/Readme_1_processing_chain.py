import numpy as np
import os
from scipy.io.wavfile import write as audio_write

### Generate data
data = np.random.uniform(size=(10000)) # single example
DATA = np.random.uniform(size=(10,10000)) # multi example
wavfiles, numpyfiles = [], []
datafolder = 'data_intro/data'
os.makedirs(datafolder,exist_ok=True)
os.makedirs(datafolder + '_numpy',exist_ok=True)
for k,D in enumerate(DATA):
    wavfiles.append(os.path.join(datafolder,str(k) + '.wav'))
    numpyfiles.append(os.path.join(datafolder + '_numpy',str(k) + '.npy'))
    np.save(numpyfiles[k], D)
    audio_write(wavfiles[k], rate=1, data=D)

# -------------------------------------------------------------------------
### Create an STFT, get mean and std over time
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# create processing chain
dp = processing_chain()
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.summary()
# apply processing chain to data
# make sure to provide sampling frequency to dp. Kwargs are always accessible for
# all processing layer. Therefore, you should make sure naming DOES NOT overlap
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Create an STFT, get mean and std over time (alternative)
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# create processing chain
# in this example, fs is already set in the processing chain
dp = processing_chain()
dp.add(Framing(windowsize=10,stepsize=10,axis=0,fs=1))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.summary()
# apply processing chain to data
output_data = dp(data)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Create an STFT, get mean and std over time and fit this to normalization
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# create processing chain
dp = processing_chain()
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
dp.summary()
# fit processing chain as Normalizer contains a 'fit' method to init parameters
dp.fit(DATA, fs=1)
# apply processing chain to data
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Same as before but the data is loaded from wav file
### As a consequence no extra fs information needs to be provided for processing. This read from the wav.
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define processing chain
dp = processing_chain()
dp.add(WavDatareader())
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
dp.summary()
# fit to wavfiles
dp.fit(wavfiles) #fit from wav files
#dp.fit(['data_intro/data_numpy/0.wav', 'data_intro/data_numpy/1.wav', 'data_intro/data_numpy/3.wav', ...], fs=1)
output_data = dp(wavfiles[2]) # process from wavfiles
#output_data = dp('data_intro/data_numpy/2.wav',fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Same as before but the data is loaded from numpy file \
### As a consequence extra fs information needs to be provided for processing.
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define processing chain
dp = processing_chain()
dp.add(NumpyDatareader())
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
# fit to numpy files
dp.fit(numpyfiles, fs=1) #fit from npy files
#dp.fit(['data_intro/data_numpy/0.npy', 'data_intro/data_numpy/1.npy', 'data_intro/data_numpy/3.npy', ...], fs=1)
output_data = dp(numpyfiles[2],fs=1) #fit from npy files
#output_data = dp('data_intro/data_numpy/2.npy',fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Create an STFT, get mean and std over time and fit this to normalization (created from hardcoded configuration)
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

config = {'chain': [{'name': 'NumpyDatareader'},
                    {'name': 'Framing',
                     'parameters': {'axis': 0, 'stepsize': 10, 'windowsize': 10}},
                    {'name': 'FFT',
                     'parameters': {'axis': 1}},
                    {'name': 'Logarithm'},
                    {'name': 'Aggregation',
                     'parameters': {'axis': 0,
                                    'combine': 'concatenate',
                                    'methods': ['mean', 'std']}},
                    {'name': 'Normalizer',
                     'parameters': {'type': 'standard'}}]}
dp = processing_chain(config)
dp.summary()
# OR
# dp = processing_chain()
# dp.add(config)
dp.fit(numpyfiles, fs=1) #fit from npy files
#dp.fit(['data_intro/data_numpy/0.npy', 'data_intro/data_numpy/1.npy', 'data_intro/data_numpy/3.npy', ...], fs=1)
output_data = dp(numpyfiles[2],fs=1) #fit from npy files
#output_data = dp('data_intro/data_numpy/2.npy',fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Create an STFT, get mean and std over time and fit this to normalization (created from yaml config)
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# get yaml configuration
config = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'))
# create processing chain from the yaml config
dp = processing_chain(config)
# fit data
dp.fit(DATA, fs=1)
# process
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Same as before, but now the yaml loading fct and feed to processing_chain() is available in a one-liner.
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
# fit data
dp.fit(DATA, fs=1)
# process
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Example on how to add a custom processing layer
# -- processing chain from config BIS
from dabstract.dataprocessor import processing_chain, processor
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# custom processor.
# This is a minimal example of what a processor can do.
class custom_processor(processor):
    def process(self, data, **kwargs):
        return data * 100, {}
        # return data, information that can be propagated to consecutive layers

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.summary()
# add a custom processor to the dp.chain
dp.add(custom_processor())
dp.summary()
# Fit data to chain
dp.fit(DATA, fs=1)
# process0
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Example on how to add a custom processing with fit option
# -- processing chain from config BIS
from dabstract.dataprocessor import processing_chain, processor
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# custom processor.
# This is a minimal example of what a processor can do.
class custom_processor(processor):
    def process(self, data, **kwargs):
        return (data - self.mean) * 100, {}
        # return data, information that can be propagated to consecutive layers
    def fit(self, data, info, **kwargs):
        self.mean = np.mean(data)

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.summary()
# add custom processor
dp.add(custom_processor())
dp.summary()
# fit data (it's recursive, so both the normalizer and the custom_processor are fit'ed on the data)
dp.fit(DATA, fs=1)
# process data
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Example on how to use any function in a dabstract processing chain
# -- processing chain from config BIS
from dabstract.dataprocessor import processing_chain, processor
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

def custom_fct(data,**kwargs):
    return (data - 5) * 100

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.summary()
# add custom processors
dp.add(custom_fct)
dp.add(lambda x: x*100)
dp.summary()
# fit data (it's recursive, so both the normalizer and the custom_processor are fit'ed on the data)
dp.fit(DATA, fs=1)
# process data
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Example on how to add a custom processing layer within configuration using !class
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config_custom', dir=os.path.join('configs','dp'),post_process=processing_chain)
# fit data (it's recursive, so both the normalizer and the custom_processor are fit'ed on the data)
dp.fit(DATA, fs=1)
# process data
output_data = dp(data, fs=1)
print(output_data.shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Create a lazy data source from disk with additional processing
### Adds a lazy mapping function to DATA and allow multi-example indexing
# -- processing chain for multiple examples
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config
from dabstract.dataset.abstract import MapAbstract, DataAbstract

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
# Fit data
dp.fit(DATA, fs=1)
# Make and abstract data source
# you can now access data as with typical indexing
# e.g. datab[0], data[1]
# in this way it accesses DATA[0] and DATA[1] respectively with the additional dp
datab = MapAbstract(DATA,dp, fs=1)
print(datab)
# allow for multi indexing, e.g. data[:] or data[0,1]
datab = DataAbstract(datab, fs=1)
print(datab)

print('\n\n\n')
# -------------------------------------------------------------------------
### Add multi-processing to lazy data source
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

# get yaml configuration and process with processing_chain()
dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
# Fit data
dp.fit(DATA, fs=1)
# Make and abstract data source
# you can now access data as with typical indexing
# e.g. datab[0], data[1]
# in this way it accesses DATA[0] and DATA[1] respectively with the additional dp
datab = MapAbstract(DATA,dp)
print(datab)
# allow for multi indexing, e.g. data[:] or data[0,1]
# and allow for multiprocessing with the workers and buffer_len flag
# indexing is paralellized, but also the iterator is
datab = DataAbstract(datab, fs=1, workers=2, buffer_len=2)
print(datab)
for k,d in enumerate(datab):
    print('Example ' + str(k))
    print(d)


