import numpy as np
import os
from scipy.io.wavfile import write as audio_write

# -- data
data = np.random.uniform(size=(10000)) # single example
DATA = np.random.uniform(size=(10,10000)) # multi example
datafolder = 'data_intro/data'
os.makedirs(datafolder,exist_ok=True)
os.makedirs(datafolder + '_numpy',exist_ok=True)
for k,D in enumerate(DATA):
    np.save(os.path.join(datafolder + '_numpy',str(k) + '.npy'), D)
    audio_write(os.path.join(datafolder,str(k) + '.wav'), rate=1, data=D)

# EXAMPLES with processing chain
# -------------------------------------------------------------------------
# -- processing_chain hardcoded
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
#from dabstract.dataset.abstract import MapAbstract, DataAbstract,SelectAbstract

dp = processing_chain()
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
dp.fit(DATA, fs=1)
output_data = dp(data, fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing_chain hardcoded from file (numpy)
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

dp = processing_chain()
dp.add(NumpyDatareader())
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
dp.fit(['data_intro/data_numpy/0.npy', 'data_intro/data_numpy/1.npy', 'data_intro/data_numpy/3.npy'], fs=1)
output_data = dp('data_intro/data_numpy/2.npy',fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing_chain hardcoded from file (numpy)
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

dp = processing_chain()
dp.add(WavDatareader())
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.add(Normalizer(type='standard'))
dp.fit(['data_intro/data/0.wav', 'data_intro/data/1.wav', 'data_intro/data/3.wav'], fs=1)
output_data = dp('data_intro/data/2.wav',fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing chain from config
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

config = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'))
dp = processing_chain(config)
dp.fit(DATA, fs=1)
output_data = dp(data, fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing chain from config BIS
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.fit(DATA, fs=1)
output_data = dp(data, fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing chain from config BIS
from dabstract.dataprocessor import processing_chain, processor
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

class custom_processor(processor):
    def process(self, data, **kwargs):
        return data * 100, {}

dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.fit(DATA, fs=1)
dp.add(custom_processor)
output_data = dp(data, fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- add custom class without config file
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

dp = load_yaml_config(filename='Readme_1_dp_config_custom', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.fit(DATA, fs=1)
output_data = dp(data, fs=1)
print(output_data.shape)

# -------------------------------------------------------------------------
# -- processing chain for multiple examples
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config
from dabstract.dataset.abstract import MapAbstract, DataAbstract

dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.fit(DATA, fs=1)
_data = MapAbstract(DATA,dp)
_data = DataAbstract(_data, fs=1, multi_processing=True)

# -------------------------------------------------------------------------
# -- processing chain with multi_processing
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.utils import load_yaml_config

dp = load_yaml_config(filename='Readme_1_dp_config', dir=os.path.join('configs','dp'),post_process=processing_chain)
dp.fit(DATA, fs=1)
_data = MapAbstract(DATA,dp)
_data = DataAbstract(_data, fs=1, multi_processing=True, workers=2, buffer_len=2)
for d in _data:
    print(d)


