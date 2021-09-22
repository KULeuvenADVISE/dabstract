import pathlib
import pickle
import types
from pprint import pprint

from dabstract.dataprocessor.processors import *
from dabstract.abstract import *
from dabstract.dataset import xval
from dabstract.dataset import select as selectm
from dabstract.utils import safe_import_module

from typing import Union, Any, List, Optional, TypeVar, Callable, Dict

tvDataset = TypeVar("Dataset")


class Dataset:
    """Dataset base class

    This is the dataset base class. It essentially is a DictSeqAbstract with additional functionality,
    such as management for: crossvalidation, feature extraction, example splitting and sample selection.

    This class should not be used on it's own. It is a base class for other datasets. When using this class as a base
    for your own dataset, one should use the following structure::

        $ class EXAMPLE(dataset):
        $     def __init__(self,
        $                  paths=None,
        $                  test_only=0,
        $                  other=...
        $                  **kwargs):
        $         # init dict abstract
        $         super().__init__(name=self.__class__.__name__,
        $                          filter=filter,
        $                          test_only=test_only)
        $         #init other variables
        $
        $     # Data: get data
        $     def set_data(self, paths):
        $         # set up dataset containing the data and optional lazy mapping and so on
        $         # the dataset is essentially a wrapped DictSeqAbstract. All your data is
        $         # is accessible through self.. e.g. len(self), self.add, self.concat, ...
        $         self.add('data', ... )
        $         self.add('label', ... )
        $         return self
        $
        $     def prepare(self,paths):
        $         # prepare data here, i.e. download

    One is advised to check the examples in dabstract/examples/introduction on how to work with datasets before reading
    the rest of this help.

    To initialise this dataset the only mandatory field is paths and paths['feat'] specifically.
    Paths should be provided as such::

        $   paths={'data': path_to_data,
        $          'meta': path_to_meta,
        $          'feat': path_to_feat}
        $   dataset = EXAMPLE(paths={...})

    The other entries
    for 'data' and 'meta' are just a suggestion and one can add as much as they like. However, it is advised to keep this convention
    if possible.

    The class offers the following key functionality on top of your dataset definition, which can be called by
    the following methods::

        .add - Add another key to the dataset
        .add_dict - Add the keys and fields of an existing dataset or DictSeqAbstract to this one
        .concat - concat dataset with dataset
        .remove - remove key from dataset
        .add_map - add mapping to a key
        .add_split - add a splitting operation to your dataset
        .add_select - apply a selection to your dataset
        .add_alias - add an alias to another key
        .keys - show the set of keys
        .set_active_keys - set an active key
        .reset_active_keys - reset the active keys
        .unpack - unpack DictSeq to a list representation
        .set_data - overwrite this method with yours to set your data
        .load_memory - load a particular key into memory
        .summary - show a summary of the dataset
        .prepare_feat - compute the features and save to disk
        .set_xval - set crossvalidation folds
        .get_xval_set - get a subdataset givin the folds

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    paths : dict or str:
        Path configuration in the form of a dictionary.
        For example::

            $   paths={ 'data': path_to_data,
            $           'meta': path_to_meta,
            $           'feat': path_to_feat}

    test_only : bool
        To specify if this dataset should be used for testing or both testing and train.
        This is only relevant if multiple datasets are combined and set_xval() is used.
        For example::

            test_only = 0 -> use for both train and test
            test_only = 1 -> use only for test

    Returns
    -------
        dataset class
    """

    def __init__(self, paths: list = None, test_only: Optional[bool] = False, **kwargs):
        # Init dataset
        self._data = DictSeqAbstract(allow_dive=True)
        self._data.name = self.__class__.__name__
        # set restricted keys
        self._data.restricted_keys = ('xval', 'test_only', 'dataset_id', 'dataset_str')
        # prepare paths
        self.prepare(paths, **kwargs)
        # add data
        self.set_data(paths, **kwargs)
        # update internals
        self._add_internal(test_only=test_only)
        self._init_meta(paths=paths, test_only=test_only)

    def __getitem__(self, index: numbers.Integral or str) -> Any:
        """Allow indexing in the form of dataset[id]"""
        return self._data[index]

    def __setitem__(self, k: str, v: Any) -> None:
        """Allow key assigment in the form of dataset[key] = Seq/DictSeq"""
        self._data[k] = v

    def __len__(self):
        """get length of dataset using len(dataset)"""
        return len(self._data)

    def __add__(self, data: tvDataset):
        """Combine datasets using the following syntax... dataset = dataset0+dataset1"""
        return self.concat(data, adjust_base=False)

    def add(
            self, key: str, data: Any, info: List[dict] = None, lazy: bool = True, **kwargs
    ) -> None:
        """Add key to dataset.
        Requirement: data should be as long as len(self)

        Parameters
        ----------
        key : str
            key to add
        data : seq/dictseq/np/list
            data to add
        info : list
            additional information that can be added that will be progated along with the data
        lazy : bool
            apply lazily or not
        """
        self._data.add(key, data, info=info, lazy=lazy, **kwargs)

    def add_dict(
            self, data: dict or DictSeqAbstract, lazy: bool = True, **kwargs
    ) -> None:
        """Add the keys of a dictionary to the existing dataset
        Requirement: length of each item in the dict should be as long as len(self)

        Parameters
        ----------
        lazy : bool
            let this dict be lazy or not
        data : dictseq/dict
            dict to add
        """
        self._data.add_dict(data, lazy=lazy, **kwargs)
        self._set_internal_meta()

    def concat(
            self, data: tvDataset, intersect: bool = False, adjust_base: bool = True
    ) -> tvDataset:
        """Add the keys of a dictionary to the existing dataset
        Requirement: data should be as long as len(self)

        Parameters
        ----------
        data : dictseq/dict
            dict to add
        intersect : bool
            keep intersection of the two dicts based on the keys
        adjust_base : bool
            protect the original dataset from adjusting.

        Returns
        -------
        dataset : Dataset class
        """
        assert isinstance(
            data, Dataset
        ), "You can only concatenate with a dabstract.Dataset instance"
        # prep the to-add dataset
        data['dataset_id'] += self.nr_datasets
        # safe-copy depending on adjust_base
        if adjust_base:
            self2 = self
        else:
            self2 = copy.deepcopy(self)
        # concat the new dataset
        self2._data.concat(
            data._data,
            intersect=intersect,
            adjust_base=True,
            allow_dive=True
        )
        # update meta information of the dataset
        assert all([key not in self2._meta.keys() for key in data._meta.keys()])
        self2._meta.update(data._meta)
        self2._meta[data.__hash__()].update({'dataset_id': self2.nr_datasets})

        return self2

    def remove(self, key: str) -> None:
        """Remove a particular key in the dataset"""
        self._data.remove(key)

    def add_map(self, key: str, map_fct: Callable, lazy: bool = None) -> None:
        """Add a mapping to a key

        Parameters
        ----------
        lazy : bool
            apply lazily or not
        key : str
            key to apply the mapping to
        map_fct : Callable
            fct which performs y = f(x)
        """

        self._data[key] = Map(
            copy.deepcopy(self._data[key]),
            lazy=(self._data._lazy[key] if lazy is None else lazy),
            map_fct=map_fct,
        )

    def _init_meta(
            self, paths: dict = dict(), test_only: bool = False, **kwargs
    ) -> None:
        # get class's hash (before any other additional steps)
        self_hash = self.__hash__()

        # set meta
        self._meta = {self_hash: {}}
        self._meta[self_hash]['name'] = self.name
        self._meta[self_hash]['dataset_id'] = 1
        self._meta[self_hash]['test_only'] = test_only
        self._meta[self_hash]['paths'] = paths
        self._meta[self_hash]['length'] = len(self)
        self._meta[self_hash]['keys'] = self.keys(dive=True)

        # set groups
        for key in self._data.keys():
            if self._data._abstract[key]:
                self._data[key].group = self_hash

    def summary(self) -> None:
        print('Dataset summary')
        for key in self._meta:
            print('-- Dataset %s:' % key)
            pprint(self._meta[key])

    def _add_internal(self, test_only: List[int]) -> None:
        # internal data
        self.adjust_mode = True
        assert 'test_only' not in self.keys(), "test_only is a protected key."
        self.add("test_only", test_only * np.ones((len(self), 1)), lazy=False)
        assert 'dataset_id' not in self.keys(), "dataset_id is a protected key."
        self.add("dataset_id", np.zeros((len(self), 1), np.int), lazy=False)
        assert 'dataset_str' not in self.keys(), "dataset_str is a protected key."
        self.add("dataset_str", [self.name] * len(self), lazy=False)
        self.adjust_mode = False

    def add_split(
            self,
            split_value: Union[float, int] = None,
            type: str = "seconds",
            reference_key: str = None,
            **kwargs
    ) -> None:
        """Add a splitting operation to the dataset

        This is a functionality handy if you for example have a dataset with chunks of 60s while you
        want examples of 1s but you do not want to reformat your entire dataset. This functionality does it
        in a lazy manner, e.g. splitting is only performed when needed. For this it needs apriori information on the
        output_shape of each example and the sampling frequency. This is automatically available IF you use
        FolderContainer data structure, as this creates DictSeq to your dataset containing filepath, filename, .. and info.
        The info entry contains the output_shape, sampling rate of your data. This work for folders containing .wav files
        AND for extracted features in the numpy format when this was performed using self.prepare_feat in this class.
        This class basically uses SplitAbstract and SampleReplicateAbstract. Key's including information, will be splitted,
        while keys including only data will be replicated depending on the splitting rate.

        Parameters
        ----------
        split_size : float/int
            split size in seconds/samples depending on 'metric'
        type : str
            split_size type ('seconds','samples')
        reference_key : str
            if samples is set as a size, one needs to provide a key reference to acquire
            time_step information from.
        """

        from dabstract.dataset.containers import Container

        # get sample information
        sample_len, sample_period, sample_duration, is_splittable = dict(), dict(), dict(), False
        for key in self.keys(dive=True):
            # check if info available
            if isinstance(self[key], Container):
                if self[key].is_splittable:
                    sample_duration[key] = self[key].get_duration()
                    sample_period[key] = self[key].get_time_step()
                    sample_len[key] = self[key].get_split_len()
                    if reference_key is None:  # select A key as a reference. Should all be equal in duration
                        reference_key = key
                    is_splittable = True
                    continue
            sample_len[key], sample_period[key], sample_duration[key] = None, None, None
        assert is_splittable, "None of the entries appear to be splittable. " \
                              "Are you certain they contain time information? " \
                              "Numpy's are not directly splittable and " \
                              "need to contained in a FeatureContainer if desired."

        # sanity checks
        if type == "samples":
            assert (
                    reference_key is not None
            ), "When choosing for samples, you should select a reference key."
            assert isinstance(reference_key, str), \
                "reference_key should be a str"
            assert isinstance(self[reference_key], Container), \
                "reference set should be of type Container"
            assert self[reference_key].is_splittable, \
                "%s is not usable to extract split_size reference from. Make sure it contains a split_len." % key
            assert sample_len[reference_key] is not None, \
                "%s does not have split_len." % reference_key
        elif type == 'seconds':
            pass

        # adjust sample_len and durations based on minimum covered duration (e.g. framing at edges vs raw audio)
        min_duration = np.min(
            np.array(
                [
                    sample_duration[key]
                    for key in self.keys(dive=True)
                    if sample_duration[key] is not None
                ]
            ),
            axis=0,
        )
        for key in sample_len:
            if sample_len[key] is not None:
                sample_len[key] = (
                        sample_len[key] * min_duration / sample_duration[key]
                ).astype(int)
            sample_duration[key] = min_duration

        # get relative split_len
        if type == "samples":
            split_ratio = split_value * sample_period[reference_key] / sample_duration[reference_key]
        elif type == 'seconds':
            split_ratio = split_value / sample_duration[reference_key]
        split_len = dict()
        for key in self.keys(dive=True):
            if sample_len[key] is not None:
                split_len[key] = split_ratio * sample_len[key]

        # Apply split for the ones with sample_len information
        self._data.adjust_mode = True
        for key in self.keys(dive=True):
            if isinstance(self[key], Container):
                if self[key].is_splittable:
                    try:
                        self[key] = Split(self[key],
                                          split_len=split_len[key],
                                          sample_len=sample_len[key],
                                          lazy=self.is_lazy(dive=True)[key])
                    except:
                        sample_len[key] = None
        # check split lengths
        for key in self._data.keys(dive=True):
            ref = None
            if sample_len[key] is not None:
                if ref is None:
                    ref = self._data[key]._splits
                assert np.all(
                    ref == self._data[key]._splits
                ), "Amount of splits are not equal. Please check why!"

        # do other keys (replicating)
        for key in self.keys(dive=True):
            if sample_len[key] is None:
                self[key] = SampleReplicate(self[key], factor=ref, lazy=self.is_lazy(dive=True)[key])
        self._data.adjust_mode = False

    def add_select(
            self,
            name: Any,
            *arg,
            parameters: Optional[dict] = dict,
            eval_data: Any = None,
            **kwargs
    ) -> None:
        """Add a selection to the dataset

        This function add a selector to the dataset. The input to this function can either be a function that does the
        selection or a name/parameter pair that is used to search for that function in dabstract.dataset.select
        AND in the specified os.environ["dabstract_CUSTOM_DIR"]. When defining custom selector functions, one can either provide
        this function directly OR place them in os.environ["dabstract_CUSTOM_DIR"] / dataset / select.py.
        Any usage for custom function uses the same directory structure as dabstract.

        Besides a function one can also directly provide indices.

        dabstract already has a set of build-in selectors in dabstract.dataset.select such
        that one can simply do::

            $  self.add_select(random_subsample, parameters=dict('ratio': 0.5))

        for random subsampling, and::

            $  self.add_select(subsample_by_str, parameters=dict('key': ..., 'keep': ...))

        for selecting based on a key and a particular value
        One can also also use the lambda function such as::

            $  self.add_select((lambda x,k: x['data']['subdb'][k]))

        Or directly use indices such as::

            $  indices = np.array[0,1,2,3,4])
            $  self.add_select(indices)

        Parameters
        ----------
        selector : Callable/str/List[int]/np.ndarray
            selector defined as a str (translated to fct internally) or function or indices
        parameters : dict
            additional parameters in case name is a str to init the function/class
        eval_data : Any
            data which could be used to available selector on in case no indices but a function is used.
            Note that if no eval_data is selected it simply assumes the dataset itself to evaluate on.
        arg/kwargs:
            additional param to provide to the function if needed
        """

        # get selector
        if isinstance(name, dict):
            if "parameters" in name:
                parameters = name["parameters"]
            assert "name" in name
            selector = name["name"]
        if isinstance(name, str):
            module = selectm
            if not hasattr(module, name):
                module = safe_import_module(
                    os.environ["dabstract_CUSTOM_DIR"] + ".dataset.select"
                )
                assert hasattr(module, name), (
                        "Select "
                        + name
                        + " is not supported in both dabstract and custom xvals. Please check"
                )
            selector = getattr(module, name)(**parameters)
        elif isinstance(name, type):
            selector = name(**parameters)
        else:
            raise NotImplementedError

        # apply selection
        self._data.add_select(selector, *arg, eval_data=eval_data, **kwargs)

    def add_alias(self, key: str, new_key: str) -> None:
        """Add an alias to a particular key. Handy if you would like to use a dataset and add e.g. data/target referring to
        something.
        """
        self._data.add_alias(key, new_key)

    def keys(self, dive: bool = False) -> List[str]:
        """Show the keys in the dataset
        """
        return self._data.keys(dive=dive)

    def set_active_keys(self, keys: Union[List[str], str]) -> None:
        """Set an active key.
        An active key simply lets a DictSeq mimic a Seq. When integer indexing a dataset it return a dictionary.
        In some cases it is desired that it only return the data from one particular key OR a set of keys.
        """
        self._data.set_active_keys(keys)

    def reset_active_key(self) -> None:
        """Reset active keys (DEPRECATED)"""
        warnings.warn(
            "reset_active_key() in dataset is deprecated. Please use reset_active_keys()"
        )
        self._data.reset_active_key()

    def reset_active_keys(self) -> None:
        """Reset active keys"""
        self._data.reset_active_keys()

    def unpack(self, keys: List[str]) -> UnpackAbstract:
        """Unpack the dictionary into a sequence
        This function return a dataset that, when indexed, return a list containing the items of 'keys' in that order.
        """
        return self._data.unpack(keys)

    @abstractmethod
    def set_data(self, paths: Dict[str, str]) -> None:
        """Placeholder that should be used to set your data in your own database class
        E.g. self.add(..) and so on
        """
        pass

    def pop(self, key: str = None) -> Any:
        return self._data.pop(key = key)

    def load_memory(
            self,
            key: str,
            workers: int = 2,
            buffer_len: int = 2,
            keep_structure: bool = False,
            verbose: bool = True,
    ) -> None:
        """Load data of a particular key from memory

        If you want to already load some data in memory as this might be the faster option you can use function.

        Parameters
        ----------
        key : str
            key to be loaded in memory
        workers : int
            amount of workers used for loading the data
        buffer_len : int
            buffer_len of the pool
        keep_structure : bool
            keep structure up another class than DictSeqAbstract
        verbose : bool
            provide print feedback
        """
        if verbose:
            print(
                "Loading data in memory of key "
                + key
                + " containing "
                + str(len(self))
                + " examples."
            )

        if keep_structure:

            def load_data(data):
                return SeqAbstract().concat(
                    DataAbstract(data,
                                 workers=workers,
                                 buffer_len=buffer_len).get(
                        slice(0, len(self)),
                        verbose=verbose
                    )
                )

            def iterative_load(data, key_str):
                if isinstance(data, DictSeqAbstract):
                    active_keys = data.get_active_keys()
                    for key in data.keys():
                        key_str2 = key_str + "/" + key
                        if isinstance(data[key], DictSeqAbstract):
                            data[key] = iterative_load(data[key], key_str2)
                        else:
                            if verbose:
                                print("Loading key " + key_str2)
                            data[key] = load_data(data[key])
                    data._set_active_keys(active_keys)
                else:
                    if verbose:
                        print("Loading key " + key_str)
                    data = load_data(data)
                return data

            self[key] = iterative_load(self[key], key_str=key)
        else:
            self[key] = DataAbstract(self[key],
                                     workers=workers,
                                     buffer_len=buffer_len).get(
                slice(0, len(self)),
                verbose=verbose
            )

    def __repr__(self) -> str:
        """String representation of the class"""
        return "dataset containing: " + str(self.keys())

    def prepare(self, paths: Dict[str, str]) -> None:
        """Placeholder for the dataset. You can add dataset download ops here."""
        pass

    def get_unique(self, key: str, fold: int = None, set: str = None, return_idx=False) -> List[Any]:
        """returns the unique values and corresponding ids  to the examples that belong to a unique group for a particular key/item.

        If not fold/set is specified, it will return the unique value and ids for all data.
        If both are specified, i.e. fold = 1 and set = 'test' it will return those associated with that dataset.
        Note that this only works if xval is initialised in set_xval().

        While get_unique(.., return_idx=False) returns the unique values of a dataset, e.g.::

            $   print(data['example'])
                    [1,2,3,1]
            $   print(data.get_unique('example'))
                    [1,2,3]

        get_unique(.., return_idx=True) also returns the associated indices::

            $   print(data.get_unique('example', return_idx=True))
                    [[1,2,3], [[0,3],[2],[3]], [[1,2],[3],[4]]

        This is primarily useful for plotting data based on a particular separating variable.

        Parameters
        ----------
        key : str
            key to get unique values from
        fold : int
            fold to get unique content of
        set : str
            set to get unique content of
        return_idx: bool
            returns the idx corresponding to the unique values or not

        Returns
        -------
        unique_values : List[np.ndarray]
            Unique value ids of that key corresponding to the output of .get_unique(...)
        data_ids : List[np.ndarray]
            idx of data matching a particular unique_value (optional if return_idx = True)
        plot_ids : List[np.ndarray]
            sequential plot idx for a particular unique_value (optional if return_idx = True)
        """

        # get data
        if fold is None: assert set is None
        if set is None: assert fold is None
        if fold is None:
            data_key = DataAbstract(self[key])[:]
        else:
            data_key = DataAbstract(self.get_xval_set(set=set, fold=fold)[key])[:]

        # get location of ids
        data_unique = np.unique(data_key)

        # get idx
        if return_idx:
            data_ids = [None] * len(data_unique)
            for k in range(len(data_unique)):
                data_ids[k] = np.where([data_tmp == data_unique[k] for data_tmp in data_key])[0]

            # get plot ids
            plot_ids = [np.in1d(np.concatenate(data_ids), data_id).nonzero()[0] for data_id in data_ids]

            return (data_unique, data_ids, plot_ids)

        else:
            return data_unique

    def prepare_feat(
            self,
            key: str,
            fe_name: str,
            fe_dp: ProcessingChain,
            new_key: str = None,
            overwrite: bool = False,
            allow_dive: bool = True,
            verbose: bool = True,
            workers: int = 2,
            buffer_len: int = 2,
    ) -> None:
        """Utility function to manage feature saving and loading.

        This function manages the feature extraction and loading for you. What it basically does it when you provide a
        particular feature extraction it processes, saves and keeps some information. Next time when this is called, it
        does not compute again, but initiates the dataset in a lazy way such that features are read from disk. If you
        want to be read in memory, you can use self.load_memory(key,...). Additionally, it also offers multi-processing
        when extracting the features. The features are added as an additional key to your dataset OR replaces the
        key containing the source data. Files are writting away in the following order:
            self['path']['feat'] / key / fe_name / ...
        the files inside that folder will have the same structure as the original files have, except that now they are writting
        as npy files.

        `It is required that 'key' contains a dictionary containing filepath, example, subdb and info in order to
        make this functionality work. This means that you should use self.add_subdict_from_folder() for the raw data.`

        Parameters
        ----------
        key : str
            key to extract features from.
        fe_name : str
            the name of the feature extraction, which will be used to define the foldername
        fe_dp : ProcessingChain
            processing_chain applied to the data
        new_key : str/None
            If None, then key will be overwritten with the data.
            If a string, then a new key is added to the dataset.
        overwrite : bool
            overwrite the features that already saved
        workers : int
            amount of workers used for loading data and extracting features
        buffer_len : int
            buffer_len of the pool
        """
        # checks
        from dabstract.dataset.containers import FolderContainer

        # pop diving to search for a FolderContainer
        data = self[key]
        if allow_dive:
            def dive_and_pop(tmp):
                data_recs = []
                containers = []
                tree = [tmp.name]
                while isinstance(tmp, Abstract):
                    if isinstance(tmp, (SelectAbstract, SplitAbstract)):
                        data_recs.append(('op', tmp.__class__, tmp.get_param()))
                        tree[0] += "." + tmp.name
                        tmp = tmp.pop()
                    elif isinstance(tmp, (DictSeqAbstract, SeqAbstract)):
                        if tmp.allow_dive:
                            data_recs.append(('dive', tmp.__class__, []))
                            tree_tmp = []
                            for key in tmp.keys(dive=True):
                                data_rec_tmp, containers_tmp, branch_tmp = dive_and_pop(tmp[key])
                                data_recs[-1][2].append((key, data_rec_tmp))
                                containers += containers_tmp
                                for branch_tmp_tmp in branch_tmp:
                                    tree_tmp.append(tree[0] + branch_tmp_tmp)
                            tree = tree_tmp
                        elif isinstance(tmp, FolderContainer):
                            data_recs.append(('container', tmp.__class__, tmp))
                            containers.append(tmp)
                            tree[0] += "." + tmp.name
                        break
                    else:
                        break
                return data_recs, containers, tree

            assert isinstance(data, Abstract), "If you want to dive to extract features from a FolderContainer your " \
                                               " objects it needs to dive through should atleast be of class Abstract."
            data_recs, containers, trees = dive_and_pop(data)
            assert len(containers), "The item in key %s should atleast contain one FolderContainer." % key
        else:
            assert isinstance(data, FolderContainer), "The item in key %s is not a container. " \
                                                "One can only use prepare_feat if a Container is given."

        # Extract features for each container
        fe_containers = [None] * len(containers)
        for k, (container, tree) in enumerate(zip(containers, trees)):
            # Add MapAbstract to container
            fe_container = MapAbstract(container, fe_dp)

            # init filepath
            assert container.group in self._meta.keys(), "The group %s of container %s does not match the groups in the" \
                                                         " meta info: %s" % \
                                                         (container.group, str(container), str(self._meta.keys()))

            meta = self._meta[container.group]
            base = os.path.join(meta['paths']['feat'],meta['name'],key,fe_name)
            rfilepaths = [os.path.join(rfolder, (identifier + postfix + ".npy")) for rfolder, identifier, postfix in \
                         zip(container['info.rfolder'],container['info.identifier'],container["info.postfix"])]
            filepaths = [os.path.join(base,filename) for filename in rfilepaths]
            for ufolder in np.unique(container['info.rfolder']):
                os.makedirs(os.path.join(base,ufolder), exist_ok=True)

            # print
            if verbose:
                print(" ")
                print("Doing dataset %s/%s of %s examples" % (container.group, self._meta[container.group]['name'], len(container)))
                print("Diving structure: %s" % tree)
                print("Feature extraction: %s" % fe_name)
                print("Saving at location: %s" % base)

            # check is it is needed to extract
            extract = False
            if overwrite:
                extract = True
            else:
                extract = np.any([not pathlib.Path(tmp).is_file() for tmp in filepaths])

            # extract loop
            if extract:
                # init dataloader
                dataloader = DataAbstract(fe_container,
                                          workers=workers,
                                          buffer_len=buffer_len).get(index=None, return_generator=True,return_info=True)
                # loop over the data
                output_infos = [None] * len(container)
                for j, data_tmp in enumerate(tqdm(dataloader, disable=(not verbose))):  # for every sample
                    # unpack
                    data_tmp, info_tmp = data_tmp
                    # save data
                    np.save(filepaths[j], data_tmp)
                    # keep info
                    output_infos[j] = info_tmp
                # save info
                with open(
                        os.path.join(base, "file_info.pickle"), "wb"
                ) as fp:
                    pickle.dump((output_infos, rfilepaths), fp)

            # load in container
            fe_containers[k] = container.get_feature_container()(base, map_fct=ProcessingChain().add(NumpyDatareader()))

            # print
            if verbose:
                print("Features loaded.")

        # save chain config
        feconfdir = os.path.join(base, "config.pickle")
        if (not os.path.isfile(feconfdir)) or overwrite:
            fe_dp.save(feconfdir)
        else:
            fe_dp.save(feconfdir)
            #fe_dp2 = ProcessingChain().load(feconfdir)
            #ToDo add a sanity check that these are equal

        # add features to dataset
        if new_key is None:
            new_key = key
            self.remove(key)

        if isinstance(key, str):
            # reconstruct function
            def reconstruct(data_recs2, containers2, fe_containers2):
                data = None
                for data_rec in reversed(data_recs2):
                    if data_rec[0] == 'dive':
                        data = data_rec[1]()
                        for branch in data_rec[2]:
                            data_tmp, containers2, fe_containers2 = reconstruct(branch[1], containers2, fe_containers2)
                            if isinstance(data, SeqAbstract):
                                data.concat(data_tmp)
                            elif isinstance(data, DictSeqAbstract):
                                data.add(branch[0], data_tmp)
                    elif data_rec[0] == 'op':
                        assert data is not None
                        if SelectAbstract in data_rec[1].__bases__:
                            data = data_rec[1](data, **data_rec[2])
                        elif SplitAbstract in data_rec[1].__bases__:
                            #ToDo: is this the most elegant way?
                            if data.is_splittable:
                                # if still splittable
                                step = np.round(data['info.length'] * data_rec[2]['split_len'] / data_rec[2]['sample_len'])
                                data = data_rec[1](data, split_len=step, sample_len=data['info.length'])
                            else:
                                # if time_axis is gone, it is replaced by a SampleReplicate
                                data = SampleReplicate(data, factor=data_rec[2]['splits'])
                    elif data_rec[0] == 'container':
                        assert data_rec[2] == containers2[0], "Sanity check went wrong. Oops... . What did you do?"
                        _, data = containers2.pop(0), fe_containers2.pop(0)
                    else:
                        raise NotImplementedError
                return data, containers2, fe_containers2
            # add to dataset
            self.add(new_key, reconstruct(data_recs, containers, fe_containers)[0])
        else:
            raise Exception(
                "new_key should be a str or None. In case of str a new key is added to the dataset, in case of None the original item is replaced."
            )

    def set_xval(
            self,
            name: Union[str, types.FunctionType, List[int], np.ndarray],
            parameters: Dict = dict(),
            save_path: str = None,
            overwrite: bool = True,
    ) -> None:
        """Set the cross-validation folds

        This function sets the crossvalidation folds. This works similar as with self.add_select().
        You can either provide a name/parameters pair where name is a string that refers to a particular function available
        in either dabstract.dataset.xval OR os.environ["dabstract_CUSTOM_DIR"] / dataset / xval.py. The former is a build-in
        xval while the latter offers you to add a custom function, which might be added to dabstract later on if validated.
        An other option is to provide the function directly through 'name'. Finally, it also offers
        to save your xval configuration such that it's identical to last experiment OR depending on where you save,
        use the same xval for different experiments.

        dabstract already has a set of build-in selectors in dabstract.dataset.xval
        such that one can simply do::

            $  self.set_xval(group_random_kfold, parameters=dict('folds': 4, 'val_frac=1/3, group_key='group'))

        for random crossvalidation with a group constraint, and::

            $  self.set_xval(sequential_kfold, parameters=dict('folds': 4, 'val_frac=1/3, group_key='group'))

        for sequential crossvalidation with a group constraint, and::

            $  self.set_xval(stratified_kfold, parameters=dict('folds': 4, 'val_frac=1/3))

        for stratified crossvalidation, and::

            $  self.set_xval(stratified_kfold, parameters=dict('folds': 4, 'val_frac=1/3))

        for random crossvalidation.

        Parameters
        ----------
        name : Callable/xval_func/str/List[int],np.ndarray
            xval defined as a str (translated to fct internally) or function
        parameters : dict
            additional parameters in case name is a str to init the function/class
        save_dir : str
            filepath to where to pickle the xval folds
        overwrite : bool
            overwrite the saved file
        """

        assert name is not None
        test_only = np.array([k for k in self["test_only"]])
        sel_vect_train = np.where(test_only == 0)[0]
        sel_vect_test = np.where(test_only == 1)[0]

        self_train = Select(self._data, sel_vect_train)

        # checks
        get_xval = True
        if save_path is not None:
            savefile_xval = os.path.join(save_path, "xval.pickle")
            if os.path.isfile(savefile_xval):
                get_xval = False

        # get
        if get_xval | overwrite:
            # get xval class
            if isinstance(name, str):
                module = xval
                if not hasattr(module, name):
                    module = safe_import_module(
                        os.environ["dabstract_CUSTOM_DIR"] + ".dataset.xval"
                    )
                    assert hasattr(module, name), (
                            "Xval "
                            + name
                            + " is not supported in both dabstract and custom xvals. Please check"
                    )
                func = getattr(module, name)(**parameters)

            elif isinstance(name, type):
                func = name(**parameters)

            elif isinstance(name, types.FunctionType):
                func = name

            xval_inds = func(self_train)
            assert "test" in xval_inds, "please return a dict with minimally a test key"

            if save_path is not None:
                os.makedirs(os.path.split(savefile_xval)[0], exist_ok=True)
                with open(savefile_xval, "wb") as f:
                    pickle.dump(xval_inds, f)
        elif save_path is not None:
            with open(savefile_xval, "rb") as f:
                xval_inds = pickle.load(f)  # load

        # sanity checks
        keys = list(xval_inds.keys())
        for key in keys:
            assert isinstance(
                xval_inds[key], list
            ), "Crossvalidation indices should be formatted in a list (for each fold)."
            assert len(xval_inds[keys[0]]) == len(
                xval_inds[key]
            ), "Amount of folds (items in list) should be the same for each test phase (train/val/test)."
            assert np.max([np.max(set) for set in xval_inds[key]]) <= len(
                self), "Crossvalidation indices contain values larger than the dataset. Please check."

        # update indices based on sel_vect_train
        for key in xval_inds:
            for k in range(len(xval_inds[key])):
                xval_inds[key][k] = sel_vect_train[xval_inds[key][k]]

        # add other test data
        for k in range(len(xval_inds["test"])):
            xval_inds["test"][k] = np.append(xval_inds["test"][k], sel_vect_test)

        # add info
        self._folds = len(xval_inds["test"])

        # add to dataset
        xval_dict = DictSeqAbstract()
        for key in xval_inds:
            fold_dict = DictSeqAbstract()
            for fold, fold_ids in enumerate(xval_inds[key]):
                bool_array = np.zeros(len(self), dtype=bool)
                bool_array[xval_inds[key][fold]] = True
                fold_dict.add('fold_' + str(fold), bool_array)
            xval_dict.add(key, fold_dict)
        self.add('xval', xval_dict)

    def get_folds(self) -> int:
        """ get the amount of folds after .set_xval() is done """
        assert 'xval' in self.keys(), "You have not yet set crossvalidation using set_xval()."
        return self._folds

    def get_xval_set(
            self,
            set: str = None,
            fold: int = None,
            keys: str = "all",
            batch_size: int = 1,
            drop_last: bool = True,
            unzip: bool = False,
            zip: bool = False,
            shuffle: bool = False,
            lazy: bool = True,
            workers: int = 1,
            buffer_len: int = 3,
    ) -> Select:
        """Get a crossvalidation subset of your dataset

        This function return a subdataset of the original one based on which set you want and which fold

        Parameters
        ----------
        set : str
            set should be in ('train','test','val') depending on what the crossvalidation fct returned
        fold : int
            get a particular fold
        keys : str
            get a subset of the keys, e.g. only input and target
        batch_size : int
            size of the batches. If None than no batching is performed.
        drop_last : bool
            when batching is performed (batch_size is >1), the last one might not be of size equal to the batch_size.
            If that is the case and, this last batch is dropped.
        lazy : bool
            apply lazily
        workers : int
            amount of workers in case lazy is false
        buffer_len : int
            used buffer length for multiprocessing in case lazy is false
        """

        # sanity checks
        if set is not None:
            assert isinstance(set, str), "set should be of type str"
            assert 'xval' in self.keys(), "Crossvalidation is not initialised. Please execute self.set_xval(..)"
            assert set in list(
                self['xval'].keys()
            ), set + " not in xval sets. Available sets are: " + str(
                list(self['xval'].keys())
            )
        assert fold is not None, "You have to select a fold."
        assert fold < self.get_folds(), "The fold you've chosen is above the total available folds."
        kwargs = {'lazy': lazy, 'workers': workers, 'buffer_len': buffer_len}
        if keys == "all":
            if set is None:
                def get_xval_set(set=None, keys="all"):
                    data = Select(
                        self._data,
                        np.where(self['xval'][set]['fold_' + str(fold)])[0],
                        **kwargs
                    )
                    if keys != "all":
                        data = UnpackAbstract(data, keys)
                    if shuffle:
                        data = ShuffleAbstract(data)
                    if batch_size > 1:
                        data = BatchAbstract(data, batch_size=batch_size, drop_last=drop_last, unzip=unzip, zip=zip)
                    return data

                return get_xval_set
            else:
                data = self._data
        else:
            data = self._data

        data = Select(data,
                      np.where(self['xval'][set]['fold_' + str(fold)])[0],
                      **kwargs)
        if keys != "all":
            data = UnpackAbstract(data, keys)
        if shuffle:
            data = ShuffleAbstract(data)
        if batch_size > 1:
            data = BatchAbstract(data, batch_size=batch_size, drop_last=drop_last, unzip=unzip, zip=zip)

        return data

    def is_lazy(self, dive: bool = False):
        return self._data.is_lazy(dive=dive)

    @property
    def nr_datasets(self):
        return len(self._meta.keys())

    @property
    def name(self):
        return self._data._name

    @name.setter
    def name(self, name: str):
        raise NotImplementedError("Name of a dataset is equal to the name of the class. Please select an unique name.")

    @property
    def adjust_mode(self):
        return self._data._adjust_mode

    @adjust_mode.setter
    def adjust_mode(self, value: bool):
        self._data._adjust_mode = value