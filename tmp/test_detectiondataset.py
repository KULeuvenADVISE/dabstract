import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

from tmp.detectiondataset import DetectionDataset


data = DetectionDataset(paths = {'data': 'tmp/data/','meta': 'tmp/data/'})