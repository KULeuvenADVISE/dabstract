from dabstract.dataprocessor import MultiTimeLabelFilter, ProcessingChain
from tmp.detectiondataset import DetectionDataset


data = DetectionDataset(paths = {'data': 'tmp/data/','meta': 'tmp/data/'})

data.add_alias('time_anomaly', 'time_anomaly_label2')



meta_processor = ProcessingChain().add(MultiTimeLabelFilter(filter_type='multi_label'))
data.add_map('time_anomaly_label2', meta_processor)
print(data['time_anomaly_label2'][1])

