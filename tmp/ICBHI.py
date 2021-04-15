import pandas as pd
import numpy as np
from pdl import pdl # For automatically downloading the ICBHI2017 dataset
import librosa as lb

from dabstract.dataprocessor.processing_chain import ProcessingChain
from dabstract.dataset.dataset import Dataset
from dabstract.abstract.abstract import *
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
        from dabstract.dataset.wrappers import WavFolderContainer
        
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

        self.add_split(5)
            
        # self.add(key='audio_original', data=data_temp) # Add key 'audio' to dataset, data=data_temp
        #
        # # 2) add labels
        # # The statement "self['audio_original']['filename'][:]" contains the filenames in their respective order.
        # self._add_meta(dataframe=meta, order=self['audio_original']['filename'][:])
        #
        #
        # # 3) Check if 'audio_original' and the meta data have the same order
        # try:
        #     self._check_dataset() # Check if the dataset is correct
        #     return
        # except ValueError('Error: The files are not in the correct order.'):
        #     print('Error: The files are not in the correct order.')
        #     raise ValueError
        
        
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

        # # The demographic information will not be used nor downloaded. However, it is available on the ICBHI2017 website.
        #
        # # 3) Process all downloaded files to create the meta data (labels).
        # # 3.1) Create empty DataFrame 'weak_labels' and fill it.
        # weak_labels = pd.DataFrame(
        #     columns=['filename_segment', 'start_segment_seconds', 'stop_segment_seconds']
        # )
        # weak_labels2 = self._set_weak_labels(weak_labels, paths, split)
        #
        # # 3.2) Add extra columns to DataFrame 'weak_labels'. Fill these columns.
        # weak_labels2['normal'] = 0
        # weak_labels2['crackle'] = 0
        # weak_labels2['wheeze'] = 0
        # weak_labels2['number_of_crackles'] = 0
        # weak_labels2['number_of_wheezes'] = 0
        # weak_labels3 = self._set_new_columns(weak_labels2, paths)
        #
        # # 3.3) Add training/test info and diagnosis as well.
        # train_test = pd.read_csv(os.path.join(paths['data'], 'ICBHI_challenge_train_test.txt'), index_col=None, names=['filename', 'set'], sep='\t')
        # diagnoses = pd.read_csv(os.path.join(paths['data'], 'ICBHI_Challenge_diagnosis.txt'), index_col=None, names=['patientID', 'diagnosis'], sep='\t')
        # weak_labels4 = weak_labels3.merge(train_test, left_on='filename_original', right_on='filename', how='inner')
        # weak_labels5 = weak_labels4.merge(diagnoses, left_on='patientID', right_on='patientID', how='inner')
        #
        # # 4) Save 'weak_labels' to disk.
        # if(not os.path.exists( os.path.join(paths['meta'], 'weak_labels.csv') )):
        #     # 'weak_labels.csv' file does not yet exist.
        #     if(not os.path.exists(paths['meta'])):
        #         # Paths['meta'] does not yet exist, so we create this directory.
        #         os.mkdir(paths['meta'])
        #
        #     # Finally, let's save the 'weak_labels' DataFrame to disk.
        #     weak_labels5.to_csv(os.path.join(paths['meta'], 'weak_labels.csv'), header=True, index=False)