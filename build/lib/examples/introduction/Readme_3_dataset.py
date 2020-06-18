import numpy as np
import os
from pprint import pprint

# -- data
data = np.random.uniform(size=(10,10000))*100

# -------------------------------------------------------------------------
# -- class example
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'tmp': os.path.join('data','data')})
db.summary()

# -------------------------------------------------------------------------
# -- class example with data selection
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataset.select import random_subsample

db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'tmp': os.path.join('data','data')},
             select = random_subsample(ratio=0.3))
db.summary()

# -------------------------------------------------------------------------
# -- class example and xval
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataset.xval import random
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'tmp': os.path.join('data','data')},
             xval_func=random(folds=4,val_frac=1/3))
xval = db.get_xval_set(fold=0,set='train')

# -------------------------------------------------------------------------
# -- Feature extraction
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'tmp': os.path.join('data','data')})
# define processor
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
# add to db
db.prepare_feat('data','Framing1010',processor,path=os.path.join('features'),new_key='feat')