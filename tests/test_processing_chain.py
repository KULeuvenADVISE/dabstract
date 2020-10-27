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
from dabstract.dataprocessor import ProcessingChain
from dabstract.dataprocessor.processors import *

# create processing chain
dp = ProcessingChain()
dp.add(Framing(windowsize=10,stepsize=10,axis=0))
dp.add(FFT(axis=1))
dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
dp.summary()
# apply processing chain to data
# make sure to provide sampling frequency to dp. Kwargs are always accessible for
# all processing layer. Therefore, you should make sure naming DOES NOT overlap
output_data = dp(data, fs=1)
print(output_data.shape)