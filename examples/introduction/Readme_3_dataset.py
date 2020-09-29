import numpy as np
import os

# Readme_2_data_abstracts showed how the data abstracts work.
# Technically, they embody all functionality to work with data
# This part introduces the dataset class, which is based on a dictseqabstract
# This means that you can use it in a similar way.
# However it has additional functionality such as cross validation loading/saving, subsetting, ..
# This part mainly forces you to define datasets in terms of classes in a generic framework to allow
# easy reuse between researchers.

# -------------------------------------------------------------------------
### class example
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data')})
db.summary()

# -------------------------------------------------------------------------
### class example with data selection
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataset.select import random_subsample

db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data')},
             select = random_subsample(ratio=0.3))
db.summary()

# -------------------------------------------------------------------------
### class example and xval
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataset.xval import random_kfold
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data')})
db.set_xval(random_kfold(folds=4,val_frac=1/3))
xval = db.get_xval_set(fold=0,set='train')

# -------------------------------------------------------------------------
### class example and xval with xval saving for later reuse
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataset.xval import random_kfold
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data')})
db.set_xval(random_kfold(folds=4,val_frac=1/3), save_dir='xval')
xval = db.get_xval_set(fold=0,set='train')

# -------------------------------------------------------------------------
### Feature extraction
### paths/feat is a mandatory field that should be added when doing feature extraction
### as it determines where the features are stored
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# init db
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'feat': os.path.join('data', 'feat')}) #mandatory
# define processor
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
# do feature extraction on 'data'  if not already performed and add it to the dataset as a 'feat' key
# if new_key is not specified, the item of 'data' is replaced by the feature extracted version
db.prepare_feat('data',fe_name='Framing1010', fe_dp=processor, new_key='feat')
# again you can specify multiprocessing as:
# db.prepare_feat('data',fe_name='Framing1010', fe_dp=processor, new_key='feat', workers=2)

# from dabstract.dataset.abstract import MapAbstract
# from dabstract.dataprocessor.processing_chain import processor
# class custom_processor(processor):
#     def process(self, data, **kwargs):
#         return (data - 5) * 100, {'time_step': kwargs['time_step']}
#         # return data, information that can be propagated to consecutive layers
# pc = processing_chain()
# pc.add(custom_processor)
# db.add_map('feat', map_fct=pc)

# -------------------------------------------------------------------------
### Load data from memory
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# init db
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'feat': os.path.join('data', 'feat')}) #mandatory
# define processor
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
# do feature extraction on 'data'
db.prepare_feat('data',fe_name='Framing1010', fe_dp=processor, new_key='feat')
# load features into memory
db.load_memory('feat')

# -------------------------------------------------------------------------
### Load data from memory and keep internal structure
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# init db
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'feat': os.path.join('data', 'feat')}) #mandatory
# define processor
processor = processing_chain().add(Framing(windowsize=10, stepsize=10))
# do feature extraction on 'data'
db.prepare_feat('data',fe_name='Framing1010', fe_dp=processor, new_key='feat')
# load features into memory
db.load_memory('feat', keep_structure=True)

# -------------------------------------------------------------------------
### Splitting
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# init db
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'feat': os.path.join('data', 'feat')}) #mandatory
db.summary()
# define processor
processor = processing_chain().add(Framing(windowsize=0.1, stepsize=0.1))
# prepare features
db.prepare_feat('data',fe_name='Framing0101', fe_dp=processor, new_key='feat')
# add splitting
db.add_split(split_size=0.5)
# show summary
db.summary()
# both feat and data are timesplitted and read from disk
print(db['data'][0].shape)
print(db['feat'][0].shape)

# -------------------------------------------------------------------------
### Splitting (per frame)
from examples.introduction.custom.dataset.dbs.EXAMPLE import EXAMPLE
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *

# init db
db = EXAMPLE(paths = {   'data': os.path.join('data','data'),
                         'meta': os.path.join('data','data'),
                         'feat': os.path.join('data', 'feat')}) #mandatory
db.summary()
# define processor
processor = processing_chain().add(Framing(windowsize=0.1, stepsize=0.1))
# prepare features
db.prepare_feat('data',fe_name='Framing0101', fe_dp=processor, new_key='feat')
# add splitting
db.add_split(split_size=1, type='samples', reference_key='feat')
# show summary
db.summary()
# both feat and data are timesplitted and read from disk
print(db['data'][0].shape)
print(db['feat'][0].shape)

# -------------------------------------------------------------------------
### Dataset from config
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

## Loads the following yaml file
# datasets:
#   - name: !class [custom.dataset.dbs.EXAMPLE]
#     parameters:
#       paths:
#         data: !pathjoin [data,data]
#         meta: !pathjoin [data,data]
#         tmp: !pathjoin [data,tmp]
# !class is a custom constructor that load_yaml_config uses to replace that item by that class
data = load_yaml_config(filename='EXAMPLE_anomaly2', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()

# -------------------------------------------------------------------------
### Dataset from config through custom_dir
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable to indicate where custom fct are (custom, as in, not present in dabstract)
# This can be used instead of !class [] depending on what you think is most convenient
# structure of the custom should be:
# /dbs/.. for datasets
# /dp/.. for processing layers
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()

# -------------------------------------------------------------------------
### Dataset from config through custom_dir with xval
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly_xval', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()
print(data.get_xval_set(set='train',fold=0))

# -------------------------------------------------------------------------
### Dataset from config with two datasets and splitting
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.add_split(0.5)
data.summary()

# -------------------------------------------------------------------------
### Dataset from config with two datasets and splitting from config
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly_split', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()

# -------------------------------------------------------------------------
### Dataset from config with two datasets and subsampling on string
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()
print([subdb for subdb in data['data']['subdb']])
data.add_select((lambda x,k: x['data']['subdb'][k]=='normal'))
data.summary()
print([subdb for subdb in data['data']['subdb']])

# -------------------------------------------------------------------------
### Dataset from config with two datasets and random subsampling
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config
from dabstract.dataset.select import random_subsample

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()
print([subdb for subdb in data['data']['subdb']])
data.add_select(random_subsample(0.5))
data.summary()
print([subdb for subdb in data['data']['subdb']])

# -------------------------------------------------------------------------
### Dataset from config with two datasets and subsampling on a list and random from config
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data = load_yaml_config(filename='EXAMPLE_anomaly_subsample', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data.summary()

# -------------------------------------------------------------------------
### Merge two datasets
from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config
from dabstract.dataset.select import random_subsample

# define custom variable
os.environ["dabstract_CUSTOM_DIR"] = "custom"
# load dataset
data0 = load_yaml_config(filename='EXAMPLE_anomaly', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data0.summary()
# load dataset
data1 = load_yaml_config(filename='EXAMPLE_anomaly2', dir=os.path.join('configs', 'dbs'), walk=True,
                        post_process=dataset_from_config)
data1.summary()
# merge
data = data0+data1
