import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

from tmp.ICBHI import ICBHI


data = ICBHI(paths = {'data': 'tmp/ICBHI/','meta': 'tmp/ICBHI/'})