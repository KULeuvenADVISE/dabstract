from dabstract.dataprocessor import Processor

class custom_processor(Processor):
    def process(self, data, **kwargs):
        return data * 100, {}