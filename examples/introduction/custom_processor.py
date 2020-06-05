from dabstract.dataprocessor import processor

class custom_processor(processor):
    def process(self, data, **kwargs):
        return data * 100, {}