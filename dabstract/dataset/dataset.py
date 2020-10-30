from dabstract.dataprocessor.processors import *
from dabstract.dataset.abstract import *
from dabstract.dataset import xval
from dabstract.dataset import select as selectm
from dabstract.utils import safe_import_module

import pathlib
import pickle
import types
from pprint import pprint

from typing import Union, Any, List, Optional, cast, Type, TypeVar, Callable, Dict

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
        # Inits
        self.prepare(paths)
        self._data = DictSeqAbstract()
        self.set_data(paths)
        self._nr_datasets = 1
        self._set_summary(paths=paths, test_only=test_only)

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
        self._update_internal_meta()

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
        self._update_internal_meta()


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
        ), "You can only concatenate with a dict_dataset instance"
        # prep
        data = copy.deepcopy(data)
        for par in data._param:
            self._param.append(par)
        data['dataset_id'] = MapAbstract(data['dataset_id'],lambda x: x+1)
        self._nr_datasets += data._nr_datasets
        # concat
        self2 = self if adjust_base else copy.deepcopy(self)
        self2._data = self2._data.concat(
            data._data, intersect=intersect, adjust_base=adjust_base
        )
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

    def _set_summary(self, paths: dict = dict(), test_only: bool = False, **kwargs) -> None:
        """Internal function to set the summary
        """
        self._param = [
            {
                "name": self.__class__.__name__,
                "test_only": test_only,
                "paths": paths,
                **kwargs,
            }
        ]

    def _update_internal_meta(self, test_only: bool = False):
        """Internal function to set some internal meta
        """
        # Set other database meta
        if "test_only" not in self.keys():
            self.add("test_only", test_only * np.ones(len(self)))
        if "dataset_id" not in self.keys():
            self.add("dataset_id", np.zeros(len(self), np.int))


    def add_split(
        self,
        split_size: Union[float, int] = None,
        constraint: Optional[str] = None,
        type: str = "seconds",
        reference_key: str = None,
        **kwargs
    ) -> None:
        """Add a splitting operation to the dataset

        This is a functionality handy if you for example have a dataset with chunks of 60s while you
        want examples of 1s but you do not want to reformat your entire dataset. This functionality does it
        in a lazy manner, e.g. splitting is only performed when needed. For this it needs apriori information on the
        output_shape of each example and the sampling frequency. This is automatically available IF you use
        FolderDictSeqAbstract data structure, as this creates DictSeq to your dataset containing filepath, filename, .. and info.
        The info entry contains the output_shape, sampling rate of your data. This work for folders containing .wav files
        AND for extracted features in the numpy format when this was performed using self.prepare_feat in this class.
        This class basically uses SplitAbstract and SampleReplicateAbstract. Key's including information, will be splitted,
        while keys including only data will be replicated depending on the splitting rate.

        Parameters
        ----------
        split_size : float/int
            split size in seconds/samples depending on 'metric'
        constraint : None/str
            Option 'power2' creates sizes with a order of 2 (used for autoencoders)
        type : str
            split_size type ('seconds','samples')
        reference_key : str
            if samples is set as a size, one needs to provide a key reference to acquire
            time_step information from.
        """

        from dabstract.dataset.helpers import FolderDictSeqAbstract

        # get time_step in case of samples
        if type == "samples":
            assert (
                reference_key is not None
            ), "When choosing for samples, you should select a reference key."
            assert isinstance(reference_key, str), "reference_key should be a str"
            assert isinstance(self[reference_key], FolderDictSeqAbstract)
            assert (
                "time_step" in self[reference_key]["info"][0]
            ), "time_step should be a key in self[reference_key]['info'][..]. Splitting is currently only supported when that information is available"
            assert (
                "output_shape" in self[reference_key]["info"][0]
            ), "output_shape should be a key in self[reference_key]['info'][..]. Splitting is currently only supported when that information is available"
            type = "seconds"
            split_size = (
                np.unique(
                    np.array(
                        [info["time_step"] for info in self[reference_key]["info"]]
                    )
                )
                * split_size
            )
            assert (
                len(split_size) == 1
            ), "can only do splitting when the time_steps in each example of your dataset/key are uniform."

        # prep sample lengths
        sample_len, sample_period, sample_duration = dict(), dict(), dict()
        for key in self.keys():
            # check if info available
            if isinstance(self[key], FolderDictSeqAbstract):
                if all(
                    [
                        (("output_shape" in info) and ("time_step" in info))
                        for info in self[key]["info"]
                    ]
                ):
                    sample_duration[key] = np.array(
                        [
                            info["output_shape"][0] * info["time_step"]
                            for info in self[key]["info"]
                        ]
                    )
                    sample_len[key] = np.array(
                        [info["output_shape"][0] for info in self[key]["info"]]
                    )
                    sample_period[key] = np.array(
                        [info["time_step"] for info in self[key]["info"]]
                    )
                    assert [
                        sample_period[key][0] == time_step
                        for time_step in sample_period[key]
                    ], "sample_period should be uniform"
                    sample_period[key] = sample_period[key][0]
                    continue
            sample_len[key], sample_period[key], sample_duration[key] = None, None, None
        # adjust sample_len based on minimum covered duration (e.g. framing at edges vs raw audio)
        min_duration = np.min(
            np.array(
                [
                    sample_duration[key]
                    for key in self.keys()
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
        # Apply split for the ones with sample_len information
        new_data = DictSeqAbstract()
        for key in self.keys():
            if sample_len[key] is not None:
                try:
                    tmp = Split(
                        self[key],
                        split_size=split_size,
                        sample_len=sample_len[key],
                        sample_period=sample_period[key],
                        type=type,
                        constraint=constraint,
                        lazy=self._data._lazy[key],
                    )
                    new_data[key] = tmp
                except:
                    sample_len[key] = None
        # check split lengths
        for k, key in enumerate(new_data.keys()):
            if k == 0:
                ref = new_data[key]._split_len
            assert np.all(
                ref == new_data[key]._split_len
            ), "split length are not equal. Please check why!"
        # do other keys (replicating)
        for key in self.keys():
            if sample_len[key] is None:
                new_data.add(
                    key,
                    SampleReplicate(self[key], factor=ref, lazy=self._data._lazy[key]),
                    lazy=self._data._lazy[key],
                )
        # replace existing dataset
        self._data = new_data

    def add_select(
        self, selector: Any, parameters=Optional[dict], *arg, **kwargs
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

        for random subsampling,
        and::
            $  self.add_select(subsample_by_str, parameters=dict('key': ..., 'keep': ...))

        for selecting based on a key and a particular value
        One can also also use the lambda function such as::
            $  self.add_select((lambda x,k: x['data']['subdb'][k]))

        Or directly use indices
        such as::
            $  indices = np.array[0,1,2,3,4])
            $  self.add_select(indices)

        Parameters
        ----------
        selector : Callable/str/List[int]/np.ndarray
            selector defined as a str (translated to fct internally) or function or indices
        parameters : dict
            additional parameters in case name is a str to init the function/class
        arg/kwargs:
            additional param to provide to the function if needed
        """

        # get fct
        if isinstance(selector, dict):
            if "parameters" in selector:
                parameters = selector["parameters"]
            if "name" in selector:
                selector = selector["name"]
        if isinstance(selector, str):
            module = selectm
            if not hasattr(module, selector):
                module = safe_import_module(
                    os.environ["dabstract_CUSTOM_DIR"] + ".dataset.select"
                )
                assert hasattr(module, selectm), (
                    "Select "
                    + selectm
                    + " is not supported in both dabstract and custom xvals. Please check"
                )
            func = getattr(module, selector)(**parameters)
        elif isinstance(selector, type):
            func = selector(**parameters)
        else:
            func = selector
        # apply selection
        orig_data = copy.deepcopy(self._data)
        self._data = DictSeqAbstract()
        for key in orig_data.keys():
            self[key] = Select(
                orig_data[key],
                func,
                *arg,
                lazy=orig_data._lazy[key],
                eval_data=orig_data,
                **kwargs
            )

    def add_alias(self, key: str, new_key: str) -> None:
        """Add an alias to a particular key. Handy if you would like to use a dataset and add e.g. data/target referring to
        something.
        """
        self._data.add_alias(key, new_key)

    def keys(self) -> None:
        """Show the keys in the dataset"""
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

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

    def set_data(self, paths: Dict[str, str]) -> None:
        """Placeholder that should be used to set your data in your own database class
        E.g. self.add(..) and so on
        """
        pass

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
                    DataAbstract(data).get(
                        slice(0, len(self)),
                        verbose=verbose,
                        workers=workers,
                        buffer_len=buffer_len,
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
            self[key] = DataAbstract(self[key]).get(
                slice(0, len(self)),
                verbose=verbose,
                workers=workers,
                buffer_len=buffer_len,
            )

    def summary(self) -> None:
        """Print a dataset summary"""
        summary = {
            "keys": self._data.keys(),
            "database": [par["name"] for par in self._param],
            "test_only": [par["test_only"] for par in self._param],
            "len": [
                np.sum([dataset_id == id for dataset_id in self._data["dataset_id"]])
                for id in range(self._nr_datasets)
            ],
        }
        pprint(summary)

    def __repr__(self) -> str:
        """String representation of the class"""
        return "dataset containing: " + str(self.keys())

    def prepare(self, paths: Dict[str, str]) -> None:
        """Placeholder for the dataset. You can add dataset download ops here."""
        pass

    def prepare_feat(
        self,
        key: str,
        fe_name: str,
        fe_dp: ProcessingChain,
        new_key: str = None,
        overwrite: bool = False,
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
        from dabstract.dataset.helpers import FolderDictSeqAbstract

        assert isinstance(self[key], FolderDictSeqAbstract), (
            key + " should be of type FolderDictSeqAbstract"
        )
        # inits
        data = copy.deepcopy(self)
        data.add_map(key, fe_dp)
        subdb = [subdb for subdb in data[key]["subdb"]]
        example = [
            os.path.splitext(example)[0] + ".npy" for example in data[key]["example"]
        ]
        subdbs = list(np.unique(subdb))

        # extract
        featfilelist, infofilelist = [], []
        for dataset_id in range(self._nr_datasets):
            if verbose:
                print("Dataset " + self._param[dataset_id]["name"])
            featpath_base = os.path.join(
                self._param[dataset_id]["paths"]["feat"],
                self._param[dataset_id]["name"],
                key,
                fe_name,
            )
            for subdb in subdbs:  # for every subdb
                os.makedirs(os.path.join(featpath_base, subdb), exist_ok=True)
                sel_ind = np.where(
                    [
                        i == subdb and j == dataset_id
                        for i, j in zip(data[key]["subdb"], data["dataset_id"])
                    ]
                )[
                    0
                ]  # get indices
                if verbose:
                    print(
                        "Preparing "
                        + str(len(sel_ind))
                        + " examples in "
                        + self._param[dataset_id]["name"]
                        + " - "
                        + subdb
                    )

                tmp_featfilelist = [
                    os.path.join(featpath_base, example[k]) for k in sel_ind
                ]
                tmp_example = [example[k] for k in sel_ind]
                if (
                    np.any(
                        [
                            not pathlib.Path(tmp_featfile).is_file()
                            for tmp_featfile in tmp_featfilelist
                        ]
                    )
                    or overwrite
                ):  # if all does not exist
                    output_info = [None] * len(sel_ind)
                    # extract for every example
                    for k, data_tmp in enumerate(
                        tqdm(
                            DataAbstract(data[key]).get(
                                sel_ind,
                                return_generator=True,
                                return_info=True,
                                workers=workers,
                                buffer_len=buffer_len,
                            ),
                            disable=(not verbose),
                        )
                    ):  # for every sample
                        data_tmp, info_tmp = data_tmp
                        # save data
                        np.save(tmp_featfilelist[k], data_tmp)
                        # keep info
                        output_info[k] = info_tmp
                    # save info
                    if (
                        not pathlib.Path(
                            featpath_base, subdb, "file_info.pickle"
                        ).is_file()
                    ) or overwrite:
                        with open(
                            os.path.join(featpath_base, subdb, "file_info.pickle"), "wb"
                        ) as fp:
                            pickle.dump((output_info, tmp_example), fp)

                with open(
                    os.path.join(featpath_base, subdb, "file_info.pickle"), "rb"
                ) as fp:
                    info_in, example_in = pickle.load(fp)
                infofilelist += [
                    info_in[k]
                    for k in range(len(tmp_example))
                    if tmp_example[k] in example_in
                ]
                featfilelist += tmp_featfilelist

        # save chain config
        feconfdir = pathlib.Path(featpath_base, "config.pickle")
        if (not feconfdir.is_file()) or overwrite:
            with open(feconfdir, "wb") as fp:
                pickle.dump(fe_dp._info, fp)
        else:
            with open(feconfdir, "rb") as fp:
                feconf = pickle.load(fp)
            # assert fe_dp.summary(verbose=False)==feconf, "Feature configuration in " + str(feconfdir) + " does not match the provided processing chain. Please check."
            # ToDo(gert): check why this check does not work after serialization. Should be identical...

        # adjust features
        if new_key is None:
            new_key = key
            self.remove(key)
        if isinstance(key, str):
            self.add(
                new_key,
                FolderDictSeqAbstract(
                    featpath_base,
                    filepath=featfilelist,
                    extension=".npy",
                    map_fct=ProcessingChain().add(NumpyDatareader()),
                    info=infofilelist,
                ),
            )
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

        for random crossvalidation with a group constraint,
        and::
            $  self.set_xval(sequential_kfold, parameters=dict('folds': 4, 'val_frac=1/3, group_key='group'))

        for sequential crossvalidation with a group constraint,
        and::
            $  self.set_xval(stratified_kfold, parameters=dict('folds': 4, 'val_frac=1/3))

        for stratified crossvalidation,
        and::
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

            self.xval = func(self_train)
            assert "test" in self.xval, "please return a dict with minimally a test key"

            if save_path is not None:
                os.makedirs(os.path.split(savefile_xval)[0], exist_ok=True)
                with open(savefile_xval, "wb") as f:
                    pickle.dump(self.xval, f)
        elif save_path is not None:
            with open(savefile_xval, "rb") as f:
                self.xval = pickle.load(f)  # load

        # sanity check
        keys = list(self.xval.keys())
        for key in keys:
            assert isinstance(
                self.xval[key], list
            ), "Crossvalidation indices should be formatted in a list (for each fold)."
            assert len(self.xval[keys[0]]) == len(
                self.xval[key]
            ), "Amount of folds (items in list) should be the same for each test phase (train/val/test)."

        # update indices based on sel_vect_train
        for key in self.xval:
            for k in range(len(self.xval[key])):
                self.xval[key][k] = sel_vect_train[self.xval[key][k]]

        # add other test data
        for k in range(len(self.xval["test"])):
            self.xval["test"][k] = np.append(self.xval["test"][k], sel_vect_test)

        # add info
        self.xval["folds"] = len(self.xval["test"])

        return self.xval

    def get_xval_set(
        self,
        set: str = None,
        fold: int = None,
        keys: str = "all",
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
        lazy : bool
            apply lazily
        workers : int
            amount of workers in case lazy is false
        buffer_len : int
            used buffer length for multiprocessing in case lazy is false
        """

        # checks
        if set is not None:
            assert hasattr(self, "xval"), "xval is not set. Please exec self.set_xval()"
            assert set in list(
                self.xval.keys()
            ), "xval_set not in xval sets. Available sets are: " + str(
                list(self.xval.keys())
            )
            assert fold < self.xval["folds"]
        assert fold is not None
        assert fold < self.xval["folds"]
        if keys is "all":
            if set is None:

                def get_xval_set(set=None, keys="all"):
                    if keys is "all":
                        return Select(self._data, self.xval[set][fold])
                    else:
                        return Select(
                            UnpackAbstract(self._data, keys),
                            self.xval[set][fold],
                            lazy=False,
                            workers=workers,
                            buffer_len=buffer_len,
                        )

                return get_xval_set
            else:
                return Select(
                    self._data,
                    self.xval[set][fold],
                    lazy=lazy,
                    workers=workers,
                    buffer_len=buffer_len,
                )
        else:
            return Select(
                UnpackAbstract(self._data, keys),
                self.xval[set][fold],
                lazy=lazy,
                workers=workers,
                buffer_len=buffer_len,
            )
