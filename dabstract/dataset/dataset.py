import pathlib
import pickle
from tqdm import tqdm
import copy
import types
import os
import numpy as np
from pprint import pprint
from pylab import plot, imshow, show

from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.dataset.abstract import *
from dabstract.dataset import xval
from dabstract.utils import safe_import_module

class dataset():
    def __init__(self,
                 paths=None,
                 split=None,
                 select=None,
                 test_only=False,
                 **kwargs):
        # Inits
        self.prepare(paths)
        self._data = DictSeqAbstract()
        self.set_data(paths)
        self._nr_datasets = 0
        # Add other database meta
        if len(self.keys()) != 0:
            self.add('test_only', [test_only] * len(self))
            self.add('dataset_id', np.zeros(len(self),np.int))
            self._nr_datasets += 1
        # filter
        if select is not None:
            if not isinstance(select, list): select = [select]
            for f in select:
                self.add_select(f)
        # split
        if split is not None:
            self.add_split(**split)
        # dataset meta
        if len(self.keys()) != 0:
            # get default param
            self._param = [{'name': self.__class__.__name__,
                          'test_only': test_only,
                          'paths': paths,
                          'filter': filter,
                          'split': split,
                           **kwargs}]
            # add other meta
            self._param[0] = self.set_meta(self._param[0])
        else:
            self._param = []
        # other
        self.data_init = {}
            
    def __getitem__(self,index):
        return self._data[index]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __len__(self):
        return len(self._data)

    def __add__(self,data):
        return self.concat(data)

    def add(self, key, data, **kwargs):
        self._data.add(key, data, **kwargs)

    def add_dict(self,data,**kwargs):
        self._data.add_dict(data,**kwargs)

    def concat(self, data, intersect=False):
        # check
        # if data.xval is not None:
        #     Warning("Concatenated dataset has xval. Be aware that xval on the base dataset is used.")
        assert isinstance(data, dataset), "You can only concatenate with a dict_dataset instance"
        # prep
        data = copy.deepcopy(data)
        for par in data._param:
            self._param.append(par)
        nr_datasets = self._nr_datasets
        data.add_map('dataset_id', (lambda x: x + nr_datasets))
        self._nr_datasets += data._nr_datasets
        # concat
        self._data.concat(data._data, intersect=intersect)
        return self

    def remove(self, key):
        self._data.remove(key)

    def add_map(self, key, map_fct, *arg, **kwargs):
        # assert key in self.keys()
        # if isinstance(self._data[key],DictSeqAbstract):
        #     assert len(self._data[key].active_key)==1
        #     self._data[key].add_map(self._data[key].active_key[0], map_fct, *arg, **kwargs)
        # else:
        self._data[key] = MapAbstract(copy.deepcopy(self._data[key]), map_fct=map_fct)

    def set_meta(self,param):
        return param

    def add_split(self, split_size=None, constraint=None, type='seconds', **kwargs):
        # prep sample lengths
        sample_len, sample_period, sample_duration = dict(), dict(), dict()
        for key in self.keys():
            # check if info available
            if isinstance(self[key],DictSeqAbstract):
                if 'info' in self[key].keys():
                    if all([(('output_shape' in info) and ('time_step' in info)) for info in self[key]['info']]):
                        sample_duration[key] = np.array([info['output_shape'][0]*info['time_step'] for info in self[key]['info']])
                        sample_len[key] = np.array([info['output_shape'][0] for info in self[key]['info']])
                        sample_period[key] = np.array([info['time_step'] for info in self[key]['info']])
                        assert [sample_period[key][0]==time_step for time_step in sample_period[key]], "sample_period should be uniform"
                        sample_period[key] = sample_period[key][0]
                        continue
            sample_len[key], sample_period[key], sample_duration[key] = None, None, None
        # adjust sample_len based on minimum covered duration (e.g. framing at edges vs raw audio)
        min_duration = np.min(np.array([sample_duration[key] for key in self.keys() if sample_duration[key] is not None]),axis=0)
        for key in sample_len:
            if sample_len[key] is not None:
                sample_len[key] = (sample_len[key] * min_duration/sample_duration[key]).astype(int)
        # Apply split for the ones with sample_len information
        new_data = DictSeqAbstract()
        for key in self.keys():
            if sample_len[key] is not None:
                new_data[key] = SplitAbstract(self[key], split_size=split_size, sample_len=sample_len[key], sample_period=sample_period[key], type=type, constraint=constraint)
        # check split lengths
        for k, key in enumerate(new_data.keys()):
            if k == 0:
                ref = new_data[key]._split_len
            assert np.all(ref == new_data[key]._split_len), "split length are not equal. Please check why!"
        # do other keys (replicating)
        for key in self.keys():
            if sample_len[key] is None:
                new_data[key] = SampleReplicateAbstract(self[key],factor=ref)
        # replace existing dataset
        self._data = new_data

    def add_select(self, select, *arg, **kwargs):
        orig_data = copy.deepcopy(self._data)
        self._data = DictSeqAbstract()
        for key in orig_data.keys():
            self[key] = SelectAbstract(orig_data[key], select,  *arg, eval_data=orig_data, **kwargs)

    def add_alias(self,key, new_key):
        self._data.add_alias(key, new_key)

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def set_active_keys(self,keys):
        self._data.set_active_keys(keys)

    def reset_active_key(self):
        self._data.reset_active_key()

    def unpack(self, keys):
        return self._data.unpack(keys)

    def set_data(self,paths):
        pass

    def get_xval_set(self, set=None, fold=None, keys='all', **kwargs):
        # checks
        if set is not None:
            assert hasattr(self,'xval'), "xval is not set. Please exec self.set_xval()"
            assert set in list(self.xval.keys()), "xval_set not in xval sets. Available sets are: " + str(list(self.xval_dict.keys()))
            assert fold<self.xval['folds']
        assert fold is not None
        assert fold < self.xval['folds']
        if keys is 'all':
            if set is None:
                def get_xval_set(set=None):
                    return SelectAbstract(self._data,self.xval[set][fold])
                return get_xval_set
            else:
                return SelectAbstract(self._data,self.xval[set][fold])
        else:
            raise NotImplementedError("In future release zipping will be addded")

    def prepare_feat(self,key,fe_name,fe_dp, new_key=None, overwrite=False, verbose=True,
                     multi_processing=False, workers=2, buffer_len=2):
        # checks
        assert [file is not None for file in self[key]['filepath']], "not all entries contain filepath"
        assert [item is not None for item in self[key]['example']], "not all entries contain filename"
        assert [file is not None for file in self[key]['subdb']], "not all entries contain subdb"
        assert [file is not None for file in self[key]['info']], "not all entries contain info"
        # inits
        data = copy.deepcopy(self)
        data.add_map(key, fe_dp)
        subdb = [subdb for subdb in data[key]['subdb']]
        example = [os.path.splitext(example)[0] + '.npy' for example in data[key]['example']]
        subdbs = list(np.unique(subdb))

        # extract
        featfilelist, infofilelist = [], []
        for dataset_id in range(self._nr_datasets):
            print('Dataset ' + self._param[dataset_id]['name'])
            featpath_base = os.path.join(self._param[dataset_id]['paths']['feat'], self._param[dataset_id]['name'], key, fe_name)
            for subdb in subdbs: # for every subdb
                os.makedirs(os.path.join(featpath_base, subdb), exist_ok=True)
                sel_ind = np.where([i==subdb and j==dataset_id for i,j in zip(data[key]['subdb'],data['dataset_id'])])[0] # get indices
                if verbose: print('Preparing ' + str(len(sel_ind)) + ' examples in ' + self._param[dataset_id]['name'] + ' - ' + subdb)

                tmp_featfilelist = [os.path.join(featpath_base, example[k]) for k in sel_ind]
                if np.any([not pathlib.Path(tmp_featfile).is_file() for tmp_featfile in tmp_featfilelist]) or overwrite: #if all does not exist
                    output_info = [None] * len(sel_ind)
                    # extract for every example
                    for k, data_tmp in enumerate(tqdm(DataAbstract(data[key]).get(sel_ind, return_generator=True, return_info=True, \
                                                                                         multi_processing=multi_processing, workers=workers, buffer_len=buffer_len), \
                                                     disable=(not verbose))): # for every sample
                        data_tmp, info_tmp = data_tmp
                        # save data
                        #np.save(os.path.join(featpath_base,os.path.splitext(data[key]['example'][sel_ind[k]])[0] + '.npy'), data_tmp)
                        np.save(tmp_featfilelist[k],data_tmp)
                        # keep info
                        output_info[k] = info_tmp
                    # save info
                    if (not pathlib.Path(featpath_base, subdb, 'file_info.pickle').is_file()) or overwrite:
                        with open(os.path.join(featpath_base, subdb, 'file_info.pickle'),"wb") as fp: pickle.dump((output_info,example), fp)
                featfilelist += tmp_featfilelist
                # open file info
                #with open(os.path.join(featpath_base, subdb, 'file_info.pickle'), "rb") as fp: infofilelist += pickle.load(fp)

        # save chain config
        feconfdir = pathlib.Path(featpath_base, 'config.pickle')
        if (not feconfdir.is_file()) or overwrite:
            with open(feconfdir, "wb") as fp: pickle.dump(fe_dp.summary(verbose=False), fp)
        else:
            with open(feconfdir, "rb") as fp: feconf = pickle.load(fp)
            #assert fe_dp.summary(verbose=False)==feconf, "Feature configuration in " + str(feconfdir) + " does not match the provided processing chain. Please check."
            #ToDo(gert): check why this check does not work after serialization. Should be identical...

        # adjust features
        if new_key is None:
            new_key = key
            self.remove(key)
        if isinstance(key,str):
            self.add(new_key,self.dict_from_folder(featpath_base, filepath=featfilelist, extension='.npy', map_fct=processing_chain().add(NumpyDatareader)))
            #self[new_key]['info']._data = [infofilelist]
        else:
            assert 0, "new_key should be a str or None. In case of str a new key is added to the dataset, in case of None the original item is replaced."

    def get_dir_info(self,path, extension='.wav', save_path=None, filepath=None, overwrite_file_info=False):
        # get dirs
        if not isinstance(filepath,list):
            filepath = []
            for root, dirs, files in os.walk(path):
                tmp = [os.path.join(root, file) for file in files if extension in file]
                if len(tmp)>0:
                    tmp.sort()
                    filepath += tmp
        example = [os.path.relpath(file, path) for file in filepath if extension in file]
        filename = [os.path.split(file)[1] for file in example if extension in file]
        subdb = [os.path.split(file)[0] for file in example if extension in file]

        if save_path is not None:
            path = save_path

        # get additional info
        if not os.path.isfile(os.path.join(path, 'file_info.pickle')) or overwrite_file_info:
            info = self._get_dir_info(filepath,extension)
            if (save_path is not None) and (info is not None):
                os.makedirs(path, exist_ok=True)
                with open(pathlib.Path(path, 'file_info.pickle'), "wb") as fp: pickle.dump((info, example),  fp)
        else:
            with open(os.path.join(path, 'file_info.pickle'), "rb") as fp:
                info, example_in = pickle.load(fp)
            info = [info[k] for k in range(len(example)) if example[k] in example_in]
            assert len(example_in) == len(filepath), "info file not of same size as directory"
        return {'filepath': filepath, 'example': example, 'filename': filename, 'subdb': subdb, 'info': info}

    def _get_dir_info(self, filepath, extension):
        info = [dict()] * len(filepath)
        if extension == '.wav':
            import soundfile as sf
            for k in range(len(filepath)):
                import soundfile as sf
                f = sf.SoundFile(filepath[k])
                info[k]['output_shape'] = np.array([len(f), f.channels])
                info[k]['fs'] = f.samplerate
                info[k]['time_step'] = 1 / f.samplerate
        return info

    def dict_from_folder(self,path, extension='.wav', map_fct=None, save_path=None, filepath=None, overwrite_file_info=False):
        data = DictSeqAbstract()
        # get info
        fileinfo = self.get_dir_info(path,extension=extension, save_path=save_path, filepath=filepath, overwrite_file_info=overwrite_file_info)
        # add data
        data.add('data', fileinfo['filepath'], active_key=True)
        # add meta
        for key in fileinfo:
            data.add(key,fileinfo[key])
        # add map
        data['data'] = MapAbstract(data['data'],map_fct=map_fct)
        # set active key
        data.set_active_keys('data')
        return data

    def get_featconf(self):
        raise NotImplementedError
        #ToDo(gert): add functionality to return current feature config

    def set_xval(self, name, parameters = dict(), save_dir=None, overwrite=True):
        assert name is not None
        test_only = np.array([k for k in self['test_only']])
        sel_vect_train = np.where(test_only == 0)[0]
        sel_vect_test = np.where(test_only == 1)[0]

        self_train = DataAbstract(SelectAbstract(self._data,sel_vect_train))

        # checks
        get_xval = True
        if save_dir is not None:
            savefile_xval = os.path.join(save_dir, 'xval.pickle')
            if os.path.isfile(savefile_xval):
                get_xval = False

        # get
        if get_xval | overwrite:
            # get xval class
            if isinstance(name, str):
                module = xval
                if not hasattr(module, name):
                    module = safe_import_module(os.environ['dabstract_CUSTOM_DIR'] + '.evaluation.xval')
                    assert hasattr(module,name), "Xval " + name + " is not supported in both dabstract and custom xvals. Please check"
                func = getattr(module, name)(**parameters)
            elif isinstance(name, (type, (types.FunctionType,types.ClassType))):
                func = name(**parameters)

            self.xval = func(self_train)
            assert 'test' in self.xval, "please return a dict with minimally a test key"

            if save_dir is not None:
                os.makedirs(os.path.split(savefile_xval)[0], exist_ok=True)
                with open(savefile_xval, 'wb') as f: pickle.dump(self.xval, f)
        elif save_dir is not None:
            with open(savefile_xval, "rb") as f: self.xval = pickle.load(f)  # load

        # sanity check
        keys = list(self.xval.keys())
        for key in keys:
            assert isinstance(self.xval[key], list), 'Crossvalidation indices should be formatted in a list (for each fold).'
            assert len(self.xval[keys[0]]) == len(self.xval[key]), 'Amount of folds (items in list) should be the same for each test phase (train/val/test).'

        # add other test data
        for k in range(len(self.xval['test'])):
            self.xval['test'][k] = np.append(self.xval['test'][k], sel_vect_test)

        # add info
        self.xval['folds'] = len(self.xval['train'])

        return self.xval

    def load_memory(self, key, workers=2,buffer_len=2, verbose=True):
        if verbose:
            print('Loading data in memory of key ' + key + ' containing ' + str(len(self)) + ' examples.')
        self[key] = SeqAbstract().concat(DataAbstract(self[key]).get(slice(0,len(self)),
                                              verbose=True,
                                              workers=workers,
                                              buffer_len=buffer_len))

    def summary(self):
        summary = {'keys': self._data.keys(),
                'database': [par['name'] for par in self._param],
                'test_only': [par['test_only'] for par in self._param],
                'len': [np.sum([dataset_id==id for dataset_id in self._data['dataset_id']]) for id in range(self._nr_datasets)]}
        pprint(summary)

    def __repr__(self):
        return 'dataset containing: ' + str(self.keys())

    def prepare(self,paths):
        pass