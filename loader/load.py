import os
import json
import nltk
from utils import ROOT_DIR


class SQuALITYLoader:

    def __init__(self, version='v1-2', format='txt'):
        self.data = {}
        self.version = version
        self.format = format
        self.load()

    def load(self):
        data_path = f'{ROOT_DIR}/data/SQuALITY/data/{self.version}/{self.format}/'
        for filename in os.listdir(data_path):
            name, file_extension = os.path.splitext(filename)
            if name not in self.data:
                self.data[name] = {}

            with open(data_path + filename) as f:
                for line in f:
                    entry = json.loads(line)
                    self.data[name][entry['metadata']['passage_id']] = entry['document']

    @staticmethod
    def get_sentences(document):
        return nltk.sent_tokenize(document)


loader = SQuALITYLoader()
loader.load()
