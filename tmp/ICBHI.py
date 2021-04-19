import pandas as pd
import numpy as np
from pdl import pdl # For automatically downloading the ICBHI2017 dataset
import librosa as lb

from dabstract.dataprocessor.processing_chain import ProcessingChain
from dabstract.dataset.dataset import Dataset
from dabstract.abstract import *
from dabstract.dataprocessor.processors import *
from dabstract.utils import stringlist2ind, str_in_list

"""
Created by Michiel
ICBHI dataset (2017) (lung sound data): https://bhichallenge.med.auth.gr/
"""
# TODO: splitting: self.add_split in function 'set_data'?

class ICBHI(Dataset):
    def set_data(self, paths):
        """
        This function reads in the original audio and places it under 'audio_original'. Furthermore, it also
        reads in the meta information (labels, training/test set, etc.) and places all this information in the
        corresponding column. 
        """
        from dabstract.dataset.containers import WavFolderContainer
        
        # 1) Read in the original audio and place it under 'audio_original'.
        # Add WavDatareader to new processing chain and set it to mono. Resampling of the audio isn't done here ('fs=None').
        chain = ProcessingChain().add(WavDatareader(fs=None)) # This chain will be used as mapping function in 'data_temp'.

        data_temp = WavFolderContainer(
            path=os.path.join(paths['data'], 'ICBHI_final_database'),
            map_fct=chain,
            file_info_save_path=os.path.join(paths['data'], str(self.__class__.__name__)+'_audio_raw'),
            overwrite_file_info=True # Note that this can get costly when using larger datasets
        )
        self.add(key='data', data=data_temp)

        self.add(MetaContainer)

        self.add_split(5)

        
    def _check_dataset(self):
        """
        This function checks if all data is in the correct order.
        An example, 'dataset[audio_preprocessed][filename][0]' contains '1_0.npy', while 'dataset[filename][0]'
        contains '1_3'.
        """
        for i in range(len(self)):
            if(self['filename'][i] != self['audio_original']['filename'][i].replace('.wav', '')):
                raise ValueError('Error: The files are not in the correct order.')
                print('Error: ', self['filename'][i], self['audio_original']['filename'][i])

        # If no errors in dataset order.
        return


    def _add_meta(self, dataframe, order: list):
        """
        The goal of this function is to add the meta data to the correct places/rows/entries.
        The meta data contains the crackle and wheeze labels, whether the file belongs to the
        train/test set, the patient's diagnosis and other factors.
        """
        filenames = dataframe['filename_segment'].to_list()
        resort = np.array(
            [
                # filenames.index(filename[0:-4]) # The '-4' is to ignore the file extension
                # filenames.index(filename.replace('.wav', ''))
                filenames.index(filename)
                for filename in self['audio_original']['filename']
            ]
        )
        dataframe = dataframe.reindex(resort)

        # add meta
        self.add('crackle', dataframe['crackle'].to_list(), lazy=False) # TODO: same order as dataframe?
        self.add('wheeze', dataframe['wheeze'].to_list(), lazy=False)
        self.add('number_of_crackles', dataframe['number_of_crackles'].to_list(), lazy=False)
        self.add('number_of_wheezes', dataframe['number_of_wheezes'].to_list(), lazy=False)
        self.add('set', dataframe['set'].to_list(), lazy=False) # Add key 'set' to data to indicate train/test set
        self.add('patientID', dataframe['patientID'].to_list(), lazy=False)
        self.add('diagnosis', dataframe['diagnosis'].to_list(), lazy=False)
        self.add('stethoscope', dataframe['stethoscope'].to_list(), lazy=False)
        self.add('measurement_location', dataframe['measurement_location'].to_list(), lazy=False)
        self.add('filename', dataframe['filename_segment'].to_list(), lazy=False) # For debugging only!
        return


    # TODO: dabstract.dataset calls 'prepare(self, paths)'. It needs to call 'prepare(self, paths, split)'.
    def prepare(self, paths):
        """
        This function downloads the ICBHI dataset and unzips it.
        This function uses the following Python3 libraries:
          - Python Download Library (PDL)
          - requests
          - zipfile

        Source: https://www.zerotosingularity.com/blog/downloading-datasets-introducting-pdl/
        """
        # TODO: check if dir/file already exists. If not, then start download.

        # 1) Download all related files.
        # 1.1) Download audio and the weak labels (crackle, wheeze) annotated for each respiratory cycle.
        #      The output becomes the folder 'ICBHI_final_database'.
        if(not os.path.exists(os.path.join(paths['data'], 'ICBHI_final_database'))):
            pdl.download(
                url='https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip',
                data_dir=paths['data'],
                keep_download=True,
                overwrite_download=False,
                verbose=True
            )

            # 2) Change the filename of '226_1b1_Pl_sc_LittC2SE.wav'.
            #    This file is in the audio directory ('final_database'), but is not in the train/test file. In the train/test file
            #    there is a file called '226_1b1_Pl_sc_Meditron', but this cannot be found in the directory. If we look at the other
            #    filenames of patient 226, then we see that all measurements on this patient were performed using the same stethoscope.
            #    Since all other measurements on patient 226 use the Meditron stethoscope, we rename the file from 'LittC2SE' to 'Meditron'.
            os.rename(
                src=os.path.join(paths['data'], 'ICBHI_final_database', '226_1b1_Pl_sc_LittC2SE.wav'),
                dst=os.path.join(paths['data'], 'ICBHI_final_database', '226_1b1_Pl_sc_Meditron.wav')
            )

        # 1.2) Download the text file that contains the diagnoses for each patient.
        if(not os.path.exists(os.path.join(paths['data'], 'ICBHI_Challenge_diagnosis.txt'))):
            pdl.download(
                url='https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_diagnosis.txt',
                data_dir=paths['data'],
                keep_download=True,
                overwrite_download=False,
                verbose=True
            )

        # 1.3) Download the train/test set distribution as a text file.
        if(not os.path.exists(os.path.join(paths['data'], 'ICBHI_challenge_train_test.txt'))):
            pdl.download(
                url='https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt',
                data_dir=paths['data'],
                keep_download=True,
                overwrite_download=False,
                verbose=True
            )

        # 1.4) Download the detailed events
        if(not os.path.exists(os.path.join(paths['data'], 'events'))):
            pdl.download(
                url='https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip',
                data_dir=paths['data'],
                keep_download=True,
                overwrite_download=False,
                verbose=True
            )