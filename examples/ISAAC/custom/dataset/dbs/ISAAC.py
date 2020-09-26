import sqlite3
import struct
from datetime import datetime

from dabstract.utils import str_in_list, stringlist2ind
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.dataset.dataset import dataset
from dabstract.dataset.abstract import DataAbstract

class ISAAC_dataset_base(dataset):
    """ISAAC dataset base class

    This is the ISAAC dataset base class. It inherits the dabstract dataset base class.
    This is a baseclass as ISAAC had multiple dataset runs, of which some were formatted in a slightly different manner.
    However, as many functionality is the same. All datasets start from this one. This baseclass cannot be used on it's own.

    Arguments:
        paths (dict): required information, read dabstract.dataset.dataset for more info
        split (): default functionality, read dabstract.dataset.dataset for more info
        select (): default functionality, read dabstract.dataset.dataset for more info
        test_only (): default functionality, read dabstract.dataset.dataset for more info
        subdb (str/list): which subdb to use, e.g. (1mA,100uA, ...)
        select_channel: 1/0 or piezo/microphone
                    first experiments contained piezo and microphone
                    while remaning experiments contained only piezo with multiple channels.
                    ! This parameters changes depending on the dataset !
        equistab (true/false): Neglect the first 10h of data to make sure measurement is stable
        no_audio (true/false): provide audio or not

    Returns:
        dataset class
    """

    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        select_channel=None,
                        equistab=True,
                        no_audio=False,
                        **kwargs):
        self.subdb = subdb
        self.select_channel = select_channel
        self.equistab = equistab
        self.no_audio = no_audio
        super().__init__(paths=paths,split=split,select=select,test_only=test_only)

    # Set dataset
    def set_data(self, paths):
        if not self.no_audio:
            # set data
            chain = processing_chain().add(WavDatareader(select_channel=self.select_channel, dtype='int16')) \
                .add(Fixed_Normalizer(type='uint16'))
            self.add_subdict_from_folder('data', paths['data'], map_fct=chain, file_info_save_path=paths['feat'])
            # filter subdb
            self.add_select((lambda x,k: x['data']['subdb'][k] in self.subdb))
        # add to database
        meta = self._get_meta(paths)
        self.add_dict(meta)
        # equistab
        if self.equistab:
            self.add_select(self._equistab_func())

    # Meta: load corrosion resistance
    def _get_meta(self, paths):
        subdb, subdbs = self.get_subdb()
        subdb_map = str_in_list(subdbs,subdb)

        # get corrosion resistance
        corrosion_resistance_ref_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            if subdb[k] != 'None_f':
                sqlReader = SqliteReader_corrosion2dataset(os.path.join(paths['meta'], 'database_' + subdb[k] + '.db'))
                tmp = sqlReader.getPotentiostatData()
                corrosion_resistance_ref_sub[k] = np.empty((len(tmp),4))
                for k2 in range(len(tmp)):
                    corrosion_resistance_ref_sub[k][k2, 0] = tmp[k2][1]  # filename UNIX timestamp
                    corrosion_resistance_ref_sub[k][k2, 1] = tmp[k2][2]  # Potentiostat data
                    corrosion_resistance_ref_sub[k][k2, 2] = tmp[k2][3]  # Equilibrium voltage
                    corrosion_resistance_ref_sub[k][k2, 3] = tmp[k2][6]  # Lin Equi averaging
            else:
                corrosion_resistance_ref_sub[k] = np.ones((1, 4)) * np.nan

        # get unix timestamps
        if self.no_audio:
            subdbs = listnp_combine([([subdb[k]] * corrosion_resistance_ref_sub[k].shape[0]) for k in range(len(subdb))])
            unix_timelist_sub = [corrosion_resistance_ref_sub[k][:,0] for k in range(len(subdb))]
            unix_timelist = listnp_combine(unix_timelist_sub)
        else:
            filepath = [filepath for filepath in self['data']['filepath']]
            unix_timelist, unix_timelist_sub = self._get_unix_time(subdb, filepath, subdb_map, corrosion_resistance_ref_sub)

        # interpolate corrosion resistance
        corrosion_resistance_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            if subdb[k] != 'None_f':
                corrosion_resistance_sub[k] = linear_interpolation(unix_timelist_sub[k], corrosion_resistance_ref_sub[k][:,0], corrosion_resistance_ref_sub[k][:,1],missing_comp=True)
            else:
                corrosion_resistance_sub[k] = np.empty(len(unix_timelist_sub[k]))
                corrosion_resistance_sub[k][:] = np.nan
        corrosion_resistance = listnp_combine(corrosion_resistance_sub)

        # equivoltage
        equivoltage_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            if subdb[k] != 'None_f':
                equivoltage_sub[k] = linear_interpolation(unix_timelist_sub[k], corrosion_resistance_ref_sub[k][:,0], corrosion_resistance_ref_sub[k][:,2],missing_comp=True)
            else:
                equivoltage_sub[k] = np.empty(len(unix_timelist_sub[k]))
                equivoltage_sub[k][:] = np.nan
        equivoltage = listnp_combine(equivoltage_sub)

        # equivoltage stab
        equivoltagestab_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            if subdb[k] != 'None_f':
                equivoltagestab_sub[k] = linear_interpolation(unix_timelist_sub[k], corrosion_resistance_ref_sub[k][:,0], corrosion_resistance_ref_sub[k][:,3],missing_comp=True)
            else:
                equivoltagestab_sub[k] = np.empty(len(unix_timelist_sub[k]))
                equivoltagestab_sub[k][:] = np.nan
        equivoltagestab = listnp_combine(equivoltagestab_sub)

        # binary anomaly
        binary_anomaly_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            binary_anomaly_sub[k] = np.ones(len(unix_timelist_sub[k])) * (subdb[k] != 'None_f')
        binary_anomaly = listnp_combine(binary_anomaly_sub)

        # wrapped unix time
        # get unix time + reference data
        weekwrap_time_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            weekwrap_time_sub[k] = wrap_time(unix_timelist_sub[k])
        weekwrap_time = listnp_combine(weekwrap_time_sub)

        return {'unix_time': unix_timelist,
                'weekwrap_time': weekwrap_time,
                'corrosion_resistance': corrosion_resistance,
                'equivoltage': equivoltage,
                'equivoltagestab': equivoltagestab,
                'binary_anomaly': binary_anomaly,
                'subdb': subdbs}

    def _get_unix_time(self, subdb, filepath, subdb_map, corrosion_resistance_ref_sub):
        # get unix time
        unix_timelist_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            unix_timelist_sub[k] = [None] * len(subdb_map[k])
            for k2 in range(len(subdb_map[k])):
                unix_timelist_sub[k][k2] = float(os.path.split(filepath[subdb_map[k][k2]])[1].replace('.wav', ''))
            # if subdb[k] != 'None_f':
            #     # unix times are aligned with start of potentiostat as
            #     unix_timelist_sub[k] = unix_timelist_sub[k] - (unix_timelist_sub[k][0] - corrosion_resistance_ref_sub[k][0, 0])
        return listnp_combine(unix_timelist_sub), unix_timelist_sub

    def _equistab_func(self):
        start = 10
        subdb,subdbs = self.get_subdb()
        unix_time = np.array([unix_time for unix_time in self['unix_time']])
        sel_vect = [None]*len(subdb)
        for k in range(len(subdb)):
            sel_vect[k] = np.array(str_in_list(subdbs,subdb[k]))
            sel_vect[k] = sel_vect[k][np.argmin(np.abs(unix_time[sel_vect[k]] - (unix_time[sel_vect[k][0]] + start * 60 * 60))):-1]
        return listnp_combine(sel_vect).astype(int)

    def get_subdb(self):
        if self.no_audio:
            subdb, subdbs = self.subdb, self.subdb
        else:
            subdbs = [subdb for subdb in self['data']['subdb']]
            _, subdb_idx = np.unique(subdbs,return_index=True)
            subdb = [subdbs[k] for k in np.sort(subdb_idx)]
        return subdb, subdbs

class ISAAC_July2019_SS316(ISAAC_dataset_base):
    """class for first ISAAC dataset (SS316)

    Dataset contains bad measurements. Do not use.
    """
    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        sensor=None,
                        equistab=True,
                        no_audio=False,
                        **kwargs):
        if subdb is None:
            self.subdb = ('0uA_f', '10uA_f', '100uA_f', '1mA_f','5mA_f')
        else:
            self.subdb = subdb
        assert sensor is not None, 'Please provide a sensor key (piezo/microphone) in database config.'
        select_channel = (0 if sensor == 'microphone' else 1)
        super(dataset).__init__(paths=paths,split=split,select=select,test_only=test_only, select_channel=select_channel, equistab=equistab, no_audio=no_audio)

class ISAAC_July2019_Steel(ISAAC_July2019_SS316):
    """class for first ISAAC dataset (Steel)

    Dataset contains bad measurements. Do not use.
    """
    pass

class ISAAC_July2020_2D_SS316_ch0(ISAAC_dataset_base):
    """class for first 2 day measurement exploration (SS316)

    Dataset contains bad measurements. Do not use.
    """
    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        equistab=True,
                        no_audio=False,
                        select_channel=0,
                        **kwargs):
        if subdb is None:
            subdb = ('None_f', '4uA_f', '100uA_f', '250uA_f','500uA_f','1mA_f')
        else:
            subdb = subdb
        super().__init__(paths=paths,split=split,select=select,test_only=test_only, select_channel=select_channel, equistab=equistab,subdb=subdb, no_audio=no_audio)

    # Set dataset
    def set_data(self, paths):
        if not self.no_audio:
            # set data
            chain = processing_chain()  .add(WavDatareader(select_channel=self.select_channel, dtype='int16')) \
                                        .add(Fixed_Normalizer(type='uint16'))
            self.add_subdict_from_folder('data', paths['data'], map_fct=chain, file_info_save_path=os.path.join(paths['feat'],self.__class__.__name__,'data','raw'))
            # filter subdb
            self.add_select((lambda x,k: x['data']['subdb'][k] in self.subdb))
        # add to database
        meta = self._get_meta(paths)
        self.add_dict(meta)
        # equistab
        if self.equistab:
            self.add_select(self.equistab_func())
        # add group (for xval purposes)
        self.add('group', self['subdb'])
        self.add('group_id', stringlist2ind(DataAbstract(self['group'])[:]).astype(int))

    def set_meta(self,param):
        param.update({'subdb': self.subdb})
        return param

    # Meta: get time of the files
    def _get_unix_time(self, subdb, filepath, subdb_map, corrosion_resistance_ref_sub):
        # get unix time
        unix_timelist_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            unix_timelist_sub[k] = [None] * len(subdb_map[k])
            for k2 in range(len(subdb_map[k])):
                unix_timelist_sub[k][k2] = (datetime.strptime(os.path.split(filepath[subdb_map[k][k2]])[1].replace('.wav',''), "%Y-%m-%d_%H.%M.%S") - datetime(1970, 1, 1)).total_seconds()
            if subdb[k] not in ('None','None_f'):
                # unix times are aligned with start of potentiostat as
                unix_timelist_sub[k] = unix_timelist_sub[k] - (unix_timelist_sub[k][0] - corrosion_resistance_ref_sub[k][0, 0])
        return listnp_combine(unix_timelist_sub), unix_timelist_sub

class ISAAC_July2020_week2_SS316_ch0(ISAAC_July2020_2D_SS316_ch0):
    """class for week 2 measurement exploration (SS316)

    Dataset contains bad measurements. Do not use.
    """
    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        no_audio=False,
                        equistab=True,
                        select_channel=0,
                        **kwargs):
        if subdb is None:
            subdb = ['420uA_f']
        else:
            subdb = subdb
        super().__init__(paths=paths,split=split,select=select,test_only=test_only, equistab=equistab,subdb=subdb, no_audio=no_audio, select_channel=select_channel)

class ISAAC_July2020_week4_SS316_ch0(ISAAC_July2020_week2_SS316_ch0):
    """class for week4 measurement exploration (SS316)

    Dataset contains bad measurements. Do not use.
    """
    pass

class ISAAC_July2020_week1_Steel_ch0(ISAAC_July2020_week2_SS316_ch0):
    """class for week1 measurement exploration (Steel)

    Dataset contains bad measurements. Do not use.
    """
    pass

class ISAAC_July2020_week3_Steel_ch0(ISAAC_July2020_week2_SS316_ch0):
    """class for week3 day measurement exploration (Steel)

    Dataset contains bad measurements. Do not use.
    """
    pass

##-----------------------------------------------------------------------------------
## MAGICS INSTRUMENTS
## Bert Van Den Broeck
## initial creation: 18/04/2019
##
## v1 (10/05/2019, BVA):
##    - high level SQLite functions for reading the "Corrosion 2 dataset" metadata
##-----------------------------------------------------------------------------------
class SqliteReader_corrosion2dataset():
    def __init__(self,sqlFile):
        # class variables
        self.sqlFile = sqlFile
        # create connection
        self.conn = sqlite3.connect(self.sqlFile)
        self.c = self.conn.cursor()
    def __delete__(self):
        # close connection on deletion of class-member
        self.conn.close()
    def getPicData(self):
        # execute query and fetch returned data
        self.c.execute("SELECT time, UNIX_time, rawData FROM camera_data")
        data = self.c.fetchall()
        # convert BLOB data to numpy arrays
        for i in range(len(data)):
            data[i] = list(data[i])
            data[i][2] = unpackNumpyArrayFromSQLite(data[i][2])
        return data
    def getPotentiostatData(self):
        # execute query and fetch returned data
        self.c.execute("SELECT time, UNIX_time, PR, Eq, VData, IData, VperS FROM potentiostat_data")
        data = self.c.fetchall()
        # convert BLOBs data to numpy arrays
        for i in range(len(data)):
            data[i] = list(data[i])
            data[i][4] = unpackNumpyArrayFromSQLite(data[i][4])
            data[i][5] = unpackNumpyArrayFromSQLite(data[i][5])
        return data

##-----------------------------------------------------------------------------------
## MAGICS INSTRUMENTS
## Bert Van Den Broeck
## initial creation: 18/04/2019
##
## v1 (18/04/2019, BVA):
##    - first creation of functions unpackNumpyArrayFromSQLite and ackNumpyArrayForSQLite
##    - support for float, np.float, np.uint8, np.uint16, np.uint32, int (not all tested)
##    - only supports NxMxOx... (square) numpy.arrays
## v2 (10/05/2019, BVA):
##    - removed write lib
##-----------------------------------------------------------------------------------
def unpackNumpyArrayFromSQLite(data):
    # ID,format,      byte, conversionCode(struct.unpack)
    npDataTypes = [[1, float, 8, 'd'],
                   [2, np.float, 8, 'd'],
                   [3, np.uint8, 1, 'B'],
                   [4, np.int16, 2, 'h'],
                   [5, np.int32, 4, 'i'],
                   [6, int, 4, 'i'],
                   [7, np.float64, 8, 'd']]

    # unpack header (headerLength + formatCode + shape) coded in int32
    headerLength = struct.unpack('i', data[0:4])[0]
    formatCode = struct.unpack('i', data[4:8])[0]
    shape = tuple([struct.unpack('i', data[i:i + 4])[0] for i in range(8, headerLength * 4, 4)])
    # information about format
    dataFormat = [i[1] for i in npDataTypes if i[0] == formatCode][0]
    bytesPerSample = [i[2] for i in npDataTypes if i[0] == formatCode][0]
    conversionCode = [i[3] for i in npDataTypes if i[0] == formatCode][0]
    # remaining data
    data = data[headerLength * 4:]
    if len(data) % bytesPerSample != 0:
        raise Exception("bad structure: data is corrupt")
    # decode
    data = np.array([struct.unpack(conversionCode, data[i:i + bytesPerSample])[0] for i in
                     range(0, len(data) - bytesPerSample + 1, bytesPerSample)]).astype(dataFormat)
    # fit into shape
    data = data.reshape(shape)
    return data

def linear_interpolation(new_time, orig_time, orig_data, missing_comp=True):
    new_data = np.zeros(len(new_time))
    for k in range(len(new_time)):
        diff = orig_time - new_time[k]
        diffs = np.abs(diff)
        lower_ids = np.where(diff <= 0)[0]
        higher_ids = np.where(diff >= 0)[0]
        if (missing_comp != 0) & (len(higher_ids) == 0):
            higher_ids = lower_ids
        if len(higher_ids) != 0:
            ids1 = np.argmin(diffs[lower_ids])
            ids2 = np.argmin(diffs[higher_ids])
            ids1 = lower_ids[ids1]
            ids2 = higher_ids[ids2]
            if np.sum(diffs[[ids1, ids2]]) != 0:
                new_value = orig_data[ids1] * diffs[ids2] / np.sum([diffs[[ids1, ids2]]]) + orig_data[ids2] * diffs[ids1] / np.sum([diffs[[ids1, ids2]]])
            else:
                new_value = orig_data[ids1]
        else:
            new_value = np.nan
        new_data[k] = new_value

    return new_data

def wrap_time(timelist):
    wrapped = np.empty(len(timelist))
    for k2 in range(len(timelist)):
        dt_object = datetime.fromtimestamp(timelist[k2])
        wrapped[k2] = dt_object.weekday() + (dt_object.hour * 60 * 60 + dt_object.minute * 60 + dt_object.second) / (24 * 60 * 60)
    return wrapped
