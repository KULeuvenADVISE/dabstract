import numpy as np
import os
from pprint import pprint

from scipy.io.wavfile import write as audio_write

### Generate data
data = np.random.uniform(size=(10000)) # single example
DATA = np.random.uniform(size=(10,10000)) # multi example
wavfiles, numpyfiles = [], []
datafolder = 'data_intro/data'
os.makedirs(datafolder,exist_ok=True)
os.makedirs(datafolder + '_numpy',exist_ok=True)
for k,D in enumerate(DATA):
    wavfiles.append(os.path.join(datafolder, str(k) + '.wav'))
    numpyfiles.append(os.path.join(datafolder + '_numpy',str(k) + '.npy'))
    np.save(numpyfiles[k], D)
    audio_write(wavfiles[k], rate=1, data=D)

# -------------------------------------------------------------------------
### Example of a dabstract Sequential object
from dabstract.dataset.abstract import SeqAbstract

# standard usage
# you can basically concatenate any sequential data
SA = SeqAbstract()
SA.concat(DATA)
SA.concat(wavfiles)
print(SA)
print(len(SA))
print(SA[0])
print(SA[-1])

# or concatenate sequential objects with eachother
SA.concat(SA)
print(SA)
print(len(SA))

print('\n\n\n')
# -------------------------------------------------------------------------
### Same example, different syntax
from dabstract.dataset.abstract import SeqAbstract

# standard usage
# you can basically concatenate any sequential data
SA = SeqAbstract([DATA,wavfiles])
print(SA)
print(len(SA))
print(SA[0])
print(SA[-1])

# or concatenate sequential objects with eachother
SA = SA + SA
print(SA)
print(len(SA))

print('\n\n\n')
# -------------------------------------------------------------------------
### A dictionary like sequential object.
### Similar to dictionary as you can use index by key
### However, each entry contains a sequential object of equal length
### Additionally you can index it using integers such that you get a dictionary for each example
from dabstract.dataset.abstract import DictSeqAbstract

# create a dictseqabstract object and fill it with two keys
DSA = DictSeqAbstract()
DSA.add('data',DATA) # 'data' key with numpy item
DSA.add('filepath',wavfiles) # 'filepath' with a list of strings !equal length!
print(DSA[0])
print(DSA['data'])
print(DSA['data'][0])

# concat and error
DSA = DSA.concat(DSA) # concat two dictseq's
#DSA.add('failure',DATA) # failure due to length requirement

print('\n\n\n')
# -------------------------------------------------------------------------
### Example of a DictSeq is a sequential dictionary
from dabstract.dataset.abstract import DictSeqAbstract

# Create the dictseq
DSA = DictSeqAbstract()
DSA.add('data',DATA)
DSA.add('filepath',wavfiles)
DSA.add('filename',[os.path.split(filepath)[0] for filepath in wavfiles])

# Set an active key
# active keys are used to define which keys will be shown when a dictseq is indexed with an integer
DSA.set_active_keys(['data','filepath'])
print(DSA[0])
print(DSA['filename'])
print(DSA['filename'][0])

# reset keys such that all keys are active
DSA.reset_active_key()

# active key (single)
# If only a single KEY is active, it will not return a dictionary, but a single example of that item
# therefore, it mimics a SeqAbstract. Different is that you can still access other information using indexing by key
# Therefore, this is a way to embed extra information which for example is not needed for learning
DSA.set_active_keys('data')
print(DSA[0])
print(DSA['filepath'])
print(DSA['filepath'][0])

print('\n\n\n')
# -------------------------------------------------------------------------
### Add a mapping to your data using MapAbstract
from dabstract.dataset.abstract import SeqAbstract, MapAbstract

# create seq abstract
SA = SeqAbstract()
SA.concat(DATA)
SA.concat(wavfiles)

# apply map
# in this example a lambda function is used. But similarly, a processing chain can be used
SA = MapAbstract(SA,(lambda x: x+100))
print(SA)
print(SA[0])
# This example doesnt work as the mapping expects an integer, and it is given a string
#print(SA[-1]) #failure example

# Therefore, you can add the mapping first to DATA and later on add wavfiles
SA = SeqAbstract()
SA.concat(MapAbstract(DATA, (lambda x: x+100)))
SA.concat(wavfiles)

print('\n\n\n')
# -------------------------------------------------------------------------
### Again an example of MapAbstract, but this time with the processing_chain
from dabstract.dataset.abstract import SeqAbstract, MapAbstract
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define chains
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(WavDatareader).add(Framing(windowsize=10, stepsize=10))

# Create a sequential abstract with two concatenated datasources that have different
# processing steps. Difference is that the first one is avauilable in memory while for the second
# it first needs to be loaded from disk. The object can therefor handle both data from memory and disk
# in a single object
SA = SeqAbstract()
SA.concat(MapAbstract(DATA, processor,fs=1))
SA.concat(MapAbstract(wavfiles, processor2,fs=1))
print(SA[0]-SA[10])

print('\n\n\n')
# -------------------------------------------------------------------------
### Again an example of MapAbstract, but this time with the processing_chain and DICTSEQ
from dabstract.dataset.abstract import SeqAbstract, MapAbstract
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define chains
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(WavDatareader).add(Framing(windowsize=10, stepsize=10))

# same example as previously, but now added to a dictseq instead of Seq
DSA = DictSeqAbstract()
DSA.add('data', MapAbstract(DATA, processor,fs=1))
DSA.add('data2', MapAbstract(wavfiles, processor2,fs=1))
print(DSA[0]['data']-DSA[0]['data2'])

# or ...
DSA2 = DictSeqAbstract()
DSA2.add('data', DATA)
DSA2.add('data2', wavfiles)
DSA2.add_map('data',processor, fs=1)
DSA2.add_map('data2',processor2, fs=1)

print(DSA['data'])
print(DSA2['data'])

print('\n\n\n')
# -------------------------------------------------------------------------
### Another example to show how you could play aroud with a DictSeq
### Also shows how to use the SelectAbstract for subsetting datasets
from dabstract.dataset.abstract import SeqAbstract, SelectAbstract

# chain
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(WavDatareader).add(Framing(windowsize=10, stepsize=10))
# dataset 1
DSA = DictSeqAbstract()
DSA.add('data',MapAbstract(DATA,processor,fs=1))
DSA.add('filepath',wavfiles)
DSA.add('filename',[os.path.split(filepath)[0] for filepath in wavfiles])
DSA.add('class',np.ones(len(DSA)))
# dataset 2
DSA2 = DictSeqAbstract()
DSA2.add('data',MapAbstract(wavfiles,processor2,fs=1))
DSA2.add('filepath',wavfiles)
DSA2.add('filename',[os.path.split(filepath)[0] for filepath in wavfiles])
DSA2.add('class',np.zeros(len(DSA)))
# concat
DSA3 = DSA + DSA2 # DSA.concat(DSA2)
DSA4 = DSA3 + DSA3
print(DSA4)

## this shows three different ways to apply a filter function to your data
# filter with lambda function
DSA4_filt = SelectAbstract(DSA4,(lambda x,k: x['class'][k]==1))
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

# filter with function
def filtfunc(x,k):
    return x['class'][k]==1
DSA4_filt = SelectAbstract(DSA4,filtfunc)
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

# filter with indices
DSA4_filt = SelectAbstract(DSA4,[1,2,3])
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

#Note! This is evaluated directly, not on the fly. So select only with lightweight get's
#In case on the fly (e.g. on data) is needed, you could use FilterAbstract but this does not provide a len()

print('\n\n\n')
# -------------------------------------------------------------------------
### This shows how you can do a nested dictionary. This could be relevant if
### for a single dataset you have multiple data sources (e.g. audio, pir, ...) and for
### each of them you would like to have a dict with meta info (e.g. filename)
from dabstract.dataset.helpers import get_dir_info

# get folder info
fileinfodict = get_dir_info(datafolder, extension='.wav')
pprint(fileinfodict)
pprint(fileinfodict.keys())

## augment with other keys such as data and class id
# first source
DSA1 = DictSeqAbstract().add_dict(fileinfodict)
DSA1.add('data', fileinfodict['filepath'])
DSA1.add('class', np.ones(len(DSA1)))
DSA1.add_map('data', processor2,fs=1)
# second source
DSA2 = DictSeqAbstract().add_dict(fileinfodict)
DSA2.add('data', fileinfodict['filepath'])
DSA2.add('class', np.ones(len(DSA2)))
DSA2.add_map('data', processor2,fs=1)
# put into new DictSeq
dataset = DictSeqAbstract()
dataset.add('audio', DSA1)
dataset.add('piezo', DSA2)

pprint(dataset[0])
DSA1.set_active_keys('data') #by reference (!!)
DSA2.set_active_keys('data')
pprint(dataset[0])
dataset.set_active_keys('audio')
pprint(dataset[0])

print('\n\n\n')
# -------------------------------------------------------------------------
### SplitAbstract
### Allows to split your data in a lazy way such that it's processed only when indexing
from dabstract.dataset.abstract import SplitAbstract

SA_split = SplitAbstract(DATA,split_size=1000,sample_len=DATA.shape[1],sample_period=1)
SA_split[-1]
print(len(DATA))
print(len(SA_split))

print('\n\n\n')
# -------------------------------------------------------------------------
### SplitAbstract from disk with info
### Allows to split your data in a lazy way from disk
### note that you always need to know apriori what the size is of the loaded data
from dabstract.dataset.abstract import SplitAbstract
from dabstract.dataset.helpers import get_dir_info
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# get information about the data
fileinfodict = get_dir_info(datafolder, extension='.wav')
# apply read wav operation to the filelist
SA = MapAbstract(fileinfodict['filepath'],map_fct=processing_chain().add(WavDatareader()))
# add a splitabstract, given that you know the expected sample_len (for each example) and the sample_period (single fs for all examples))
SA_split = SplitAbstract(SA,
                         split_size=1000,
                         sample_len=[info['output_shape'][0] for info in fileinfodict['info']],
                         sample_period=fileinfodict['info'][0]['fs'])
SA_split[-1]
print(len(DATA))
print(len(SA_split))
print(DATA.shape)
print(SA_split[0].shape)

print('\n\n\n')
# -------------------------------------------------------------------------
### Add multi indexing to the data
from dabstract.dataset.abstract import DataAbstract

# create dictseq
fileinfodict = get_dir_info(datafolder, extension='.wav')
DSA = DictSeqAbstract().add_dict(fileinfodict)

# multi indexing?
pprint(DSA[0])
#print(DSA[:]) #Does not work
#print(DSA[1:10]) #Does not work

# Use DataAbstract!
DSA2 = DataAbstract(DSA)
pprint(DSA2[0]) #Does not work
pprint(DSA2[:]) #Does not work
pprint(DSA2[1:5]) #Does not work

# -------------------------------------------------------------------------
### Add multiprocessing
DSA2 = DataAbstract(DSA, workers=2, buffer_len=2)
print(DSA2[0])
print(DSA2[:])
print(DSA2[1:5])

# or ...
for example in DSA2: #Prefetches data!!
    print(example)
    print("\n")

# # -------------------------------------------------------------------------
# ### Example of dictseq with active keys and mixed key/integer indexing
# # dict
# DSA = DictSeqAbstract().add_dict(fileinfodict)
# DSA.add('data', fileinfodict['filepath'])
# DSA.add('class', np.ones(len(DSA1)))
# DSA.add_map('data', processor2,fs=1)
# # merge
# #DSA_DA = SeqAbstract().concat(DSA).concat(DATA) # fails, output should be identical in a seqabstract
# DSA.set_active_keys('data')
# DSA_DA = SeqAbstract().concat(DSA).concat(DATA)
# print(DSA_DA)
# # ok that works, what about indexing?
# print(DSA_DA[0])
# print(DSA_DA[-1])
# # what with keys?
# DSA_DA['filepath']  #opens up a "KeyAbstract" (will never give an error, will just return None if not exists)
# print(DSA_DA['filepath'][0]) # returns correct up value
# print(DSA_DA['filepath'][-1]) # returns None
