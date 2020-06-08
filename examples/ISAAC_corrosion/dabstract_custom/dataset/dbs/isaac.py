import sqlite3
import struct
import copy
from datetime import datetime

from dabstract.utils import listnp_combine, linear_interpolation, wrap_time, str_in_list
from dabstract.dataprocessor import processing_chain
from dabstract.dataprocessor.processors import *
from dabstract.dataset.dataset import dataset
from dabstract.dataset.abstract import SelectAbstract

class ISAAC_dataset_base(dataset):
    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        sensor=None,
                        equistab=True,
                        **kwargs):
        if subdb is None:
            self.subdb = ('0uA_f', '10uA_f', '100uA_f', '1mA_f','5mA_f')
        else:
            self.subdb = subdb
        assert sensor is not None, 'Please provide a sensor key (piezo/microphone) in database config.'
        self.select_channel = (0 if sensor == 'microphone' else 1)
        self.equistab = equistab
        super().__init__(paths=paths,split=split,select=select,test_only=test_only)

    # Set dataset
    def set_data(self, paths):
        # set data
        chain = processing_chain()  .add(WavDatareader(select_channel=self.select_channel, dtype='int16')) \
                                    .add(Fixed_Normalizer(type='uint16'))
        self.add('data',self.dict_from_folder(paths['data'],map_fct=chain,save_info=True,save_path=paths['feat']))
        # add to database
        self.add_dict(self._get_meta(paths))
        # filter subdb
        self.add_select((lambda x,k: x['data']['subdb'][k] in self.subdb))
        # equistab
        if self.equistab:
            self.add_select(self.equistab_func())

    # Meta: load corrosion resistance
    def _get_meta(self, paths):
        subdb = np.unique([subdb for subdb in self['data']['subdb']._data[0]])
        subdbs = self['data']['subdb']._data[0]
        subdb_map = str_in_list(subdbs,subdb)
        filepath = self['data']['filepath']._data[0]

        # get corrosion resistance
        corrosion_resistance_ref_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            if subdb[k] != 'None_f':
                sqlReader = SqliteReader_corrosion2dataset(os.path.join(paths['meta'], 'database_' + subdb[k] + '.db'))
                tmp = sqlReader.getPotentiostatData()
                corrosion_resistance_ref_sub[k] = np.empty((len(tmp), 4))
                for k2 in range(len(tmp)):
                    corrosion_resistance_ref_sub[k][k2, 0] = tmp[k2][1]  # filename UNIX timestamp
                    corrosion_resistance_ref_sub[k][k2, 1] = tmp[k2][2]  # Potentiostat data
                    corrosion_resistance_ref_sub[k][k2, 2] = tmp[k2][3]  # Equilibrium voltage
                    corrosion_resistance_ref_sub[k][k2, 3] = tmp[k2][6]  # Lin Equi averaging
            else:
                corrosion_resistance_ref_sub[k] = np.ones((1, 4)) * np.nan

        # get unix timestamps
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
                'binary_anomaly': binary_anomaly}

    def _get_unix_time(self, subdb, filepath, subdb_map, corrosion_resistance_ref_sub):
        # get unix time
        unix_timelist_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            unix_timelist_sub[k] = [None] * len(subdb_map[k])
            for k2 in range(len(subdb_map[k])):
                unix_timelist_sub[k][k2] = float(os.path.split(filepath[subdb_map[k][k2]])[1].replace('.wav', ''))
            if subdb[k] != 'None_f':
                # unix times are aligned with start of potentiostat as
                unix_timelist_sub[k] = unix_timelist_sub[k] - (unix_timelist_sub[k][0] - corrosion_resistance_ref_sub[k][0, 0])
        return listnp_combine(unix_timelist_sub), unix_timelist_sub

    def equistab_func(self):
        start = 10
        a = self['data']
        b = a['subdb']
        c = b[0]
        subdb = [subdb for subdb in self['data']['subdb']]
        unix_time = np.array([unix_time for unix_time in self['unix_time']])
        subdbs = np.unique(subdb)
        sel_vect = [None]*len(subdbs)
        for k in range(len(subdbs)):
            sel_vect[k] = np.array(str_in_list(subdb,subdbs[k]))
            sel_vect[k] = sel_vect[k][np.argmin(np.abs(unix_time[sel_vect[k]] - (unix_time[sel_vect[k][0]] + start * 60 * 60))):-1]
        return listnp_combine(sel_vect).astype(int)

class ISAAC_SS316(ISAAC_dataset_base):
    pass

class ISAAC_steel(ISAAC_dataset_base):
    pass

class ISAAC_SS316_filt(ISAAC_dataset_base):
    def set_data(self, paths):
        # set data
        chain = processing_chain()  .add(WavDatareader(select_channel=self.select_channel, dtype='int16')) \
                                    .add(FIR_filter(type='bandstop', f=[28000,32000], taps=100, axis=1, window='hamming')) \
                                    .add(Fixed_Normalizer(type='uint16'))
        self.add('data', self.dict_from_folder(paths['data'], map_fct=chain, save_info=True, save_path=paths['feat']))
        # add to database
        self.add_dict(self._get_meta(paths))

class ISAAC_test0_Steel(ISAAC_dataset_base):
    def __init__(self,  paths=None,
                        split=None,
                        select=None,
                        test_only=False,
                        subdb=None,
                        sensor=None,
                        **kwargs):
        if subdb is None:
            self.subdb = ('0uA_f', '10uA_f', '100uA_f', '1mA_f','5mA_f')
        else:
            self.subdb = subdb
        assert sensor is not None, 'Please provide a sensor key (piezo0/piezo0) in database config.'
        self.select_channel = (0 if sensor == 'piezo0' else 1)
        super().__init__(paths=paths,split=split,select=select,test_only=test_only)

    # Set dataset
    def set_data(self, paths):
        # set data
        chain = processing_chain()  .add(WavDatareader(select_channel=self.select_channel, dtype='int16')) \
                                    .add(Fixed_Normalizer(type='uint16'))
        self.add('data',self.dict_from_folder(paths['data'],map_fct=chain,save_info=True,save_path=paths['feat']))
        # add to database
        self.add_dict(self._get_meta(paths))

    # Meta: get time of the files
    def _get_unix_time(self, subdb, filepath, subdb_map, corrosion_resistance_ref_sub):
        # get unix time
        unix_timelist_sub = [None] * len(subdb)
        for k in range(len(subdb)):
            unix_timelist_sub[k] = [None] * len(subdb_map[k])
            for k2 in range(len(subdb_map[k])):
                unix_timelist_sub[k][k2] = (datetime.strptime(os.path.split(filepath[subdb_map[k][k2]])[1].replace('.wav',''), "%Y-%m-%d_%H.%M.%S") - datetime(1970, 1, 1)).total_seconds()
            if subdb[k] != 'None_f':
                # unix times are aligned with start of potentiostat as
                unix_timelist_sub[k] = unix_timelist_sub[k] - (unix_timelist_sub[k][0] - corrosion_resistance_ref_sub[k][0, 0])
        return listnp_combine(unix_timelist_sub), unix_timelist_sub

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
    headerLength = struct.unpack('i', data[0:4])[0];
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

