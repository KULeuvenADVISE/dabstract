import numpy as np
import os
from pprint import pprint

# -- data
data = np.random.uniform(size=(10000)) # single example
DATA = np.random.uniform(size=(10,10000)) # multi example
datafolder = 'data_intro/data'
filepaths = [os.path.join(datafolder,str(k) + '.npy') for k in range(len(DATA))]
os.makedirs(datafolder,exist_ok=True)
for k,D in enumerate(DATA):
    np.save(filepaths[k], D)

# EXAMPLES with Abstract's
# -------------------------------------------------------------------------
# -- Sequential
from dabstract.dataset.abstract import SeqAbstract

# standard usage
SA = SeqAbstract()
SA.concat(DATA)
SA.concat(filepaths)
print(SA)
print(len(SA))
print(SA[0])
print(SA[-1])

# concat
SA.concat(SA)
print(SA)
print(len(SA))
#SA.concat(0) #expected failure, only with __getitem__ method

# -------------------------------------------------------------------------
# -- DictSequential
from dabstract.dataset.abstract import DictSeqAbstract

# standard usage
DSA = DictSeqAbstract()
DSA.add('data',DATA)
DSA.add('filepath',filepaths)
print(DSA[0])
print(DSA['data'])
print(DSA['data'][0])

# concat and error
DSA = DSA.concat(DSA) # concat two dictseq's
#DSA.add('failure',DATA) # failure

# -------------------------------------------------------------------------
# -- DictSequential and active key(s)
from dabstract.dataset.abstract import DictSeqAbstract

# standard usage
DSA = DictSeqAbstract()
DSA.add('data',DATA)
DSA.add('filepath',filepaths)
DSA.add('filename',[os.path.split(filepath)[0] for filepath in filepaths])

# active key (multi)
DSA.set_active_keys(['data','filepath'])
print(DSA[0])
print(DSA['filename'])
print(DSA['filename'][0])

DSA.reset_active_key()

# active key (single)
DSA.set_active_keys('data')
print(DSA[0])
print(DSA['filepath'])
print(DSA['filepath'][0])

# -------------------------------------------------------------------------
# -- MapAbstract
from dabstract.dataset.abstract import SeqAbstract, MapAbstract

# create seq abstract
SA = SeqAbstract()
SA.concat(DATA)
SA.concat(filepaths)

# apply map
SA = MapAbstract(SA,(lambda x: x+100))
print(SA)
print(SA[0])
#print(SA[-1]) #failure example

# better...
SA = SeqAbstract()
SA.concat(MapAbstract(DATA, (lambda x: x+100)))
SA.concat(filepaths)

# -------------------------------------------------------------------------
# -- MapAbstract and processing chain
from dabstract.dataset.abstract import SeqAbstract, MapAbstract
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define chains
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(NumpyDatareader).add(Framing(windowsize=10, stepsize=10))

SA = SeqAbstract()
SA.concat(MapAbstract(DATA, processor,fs=1))
SA.concat(MapAbstract(filepaths, processor2,fs=1))
print(SA[0]-SA[10])

# -------------------------------------------------------------------------
# -- MapAbstract/DictSeq and processing chain
from dabstract.dataset.abstract import SeqAbstract, MapAbstract
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# define chains
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(NumpyDatareader).add(Framing(windowsize=10, stepsize=10))

DSA = DictSeqAbstract()
DSA.add('data', MapAbstract(DATA, processor,fs=1))
DSA.add('data2', MapAbstract(filepaths, processor2,fs=1))
print(DSA[0]['data']-DSA[0]['data2'])

# or ...
DSA2 = DictSeqAbstract()
DSA2.add('data', DATA)
DSA2.add('data2', filepaths)
DSA2.add_map('data',processor, fs=1)
DSA2.add_map('data2',processor2, fs=1)

print(DSA['data'])
print(DSA2['data'])

# -------------------------------------------------------------------------
# -- Select and DictSeq + extended example
from dabstract.dataset.abstract import SeqAbstract, SelectAbstract

# Create some data
# chain
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
processor2 = processing_chain().add(NumpyDatareader).add(Framing(windowsize=10, stepsize=10))
# dataset 1
DSA = DictSeqAbstract()
DSA.add('data',MapAbstract(DATA,processor,fs=1))
DSA.add('filepath',filepaths)
DSA.add('filename',[os.path.split(filepath)[0] for filepath in filepaths])
DSA.add('class',np.ones(len(DSA)))
# dataset 2
DSA2 = DictSeqAbstract()
DSA2.add('data',MapAbstract(filepaths,processor2,fs=1))
DSA2.add('filepath',filepaths)
DSA2.add('filename',[os.path.split(filepath)[0] for filepath in filepaths])
DSA2.add('class',np.zeros(len(DSA)))
# concat
DSA3 = DSA + DSA2 # DSA.concat(DSA2)
DSA4 = DSA3 + DSA3
print(DSA4)

# filter that shit with lambda
#DSA4_filt = SelectAbstract(DSA4,(lambda x,k: x['class']==1)) #Does not work!
DSA4_filt = SelectAbstract(DSA4,(lambda x,k: x['class'][k]==1))
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

# filter that shit with function
def filtfunc(x,k):
    return x['class'][k]==1
DSA4_filt = SelectAbstract(DSA4,filtfunc)
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

# filter that shit with indices
DSA4_filt = SelectAbstract(DSA4,[1,2,3])
print(DSA4_filt)
print(len(DSA4))
print(len(DSA4_filt))
print(DSA4_filt['class'])

#Note! This is evaluated directly, not on the fly. So select only with lightweight get's
#In case on the fly (e.g. on data) is needed, you could use FilterAbstract but this does not provide a len()

# -------------------------------------------------------------------------
# -- Nested dict and folder info
from dabstract.dataset.dataset import dataset

# get folder info
fileinfodict = dataset._get_dir_info([], datafolder,extension='npy',save_info=False)
pprint(fileinfodict)

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

# -------------------------------------------------------------------------
# -- SplitAbstract
from dabstract.dataset.abstract import SplitAbstract
# standard usage
SA = SeqAbstract()
SA.concat(DATA)
SA.concat(filepaths)

SA_split = SplitAbstract(SA,split_size=1000,sample_len=DATA.shape[1],sample_period=1)
SA_split[-1]

# -------------------------------------------------------------------------
# -- KeyAbstract
# so what if a SeqAbstract with a DistSeq and other data?

# dict
DSA = DictSeqAbstract().add_dict(fileinfodict)
DSA.add('data', fileinfodict['filepath'])
DSA.add('class', np.ones(len(DSA1)))
DSA.add_map('data', processor2,fs=1)

# merge
#DSA_DA = SeqAbstract().concat(DSA).concat(DATA) # fails, output should be identical in a seqabstract
DSA.set_active_keys('data')
DSA_DA = SeqAbstract().concat(DSA).concat(DATA)
print(DSA_DA)

# ok that works, what about indexing?
print(DSA_DA[0])
print(DSA_DA[-1])

# what with keys?
DSA_DA['filepath']  #opens up a "KeyAbstract" (will never give an error, will just return None if not exists)
print(DSA_DA['filepath'][0]) # returns correct up value
print(DSA_DA['filepath'][-1]) # returns None

# -------------------------------------------------------------------------
# -- Indexing and Multiprocessing
# so what if a SeqAbstract with a DistSeq and other data?
from dabstract.dataset.abstract import DataAbstract

# multi indexing?
print(DSA[0].shape)
#print(DSA[:]) #Does not work
#print(DSA[1:10]) #Does not work

# Use DataAbstract!
DSA2 = DataAbstract(DSA)
print(DSA2[0].shape) #Does not work
print(DSA2[:].shape) #Does not work
print(DSA2[1:5].shape) #Does not work

# Add multiprocessing...
DSA2 = DataAbstract(DSA, multi_processing=True, workers=2, buffer_len=2)
print(DSA2[0].shape) #Does not work
print(DSA2[:].shape) #Does not work
print(DSA2[1:5].shape) #Does not work

# or ...
for example in DSA2: #Prefetches data!!
    print(example)
    print("\n")

# -------------------------------------------------------------------------
# -- Left to do
# ToDo (gert): SplitAbstract
# ToDo (gert): Loading in memory
