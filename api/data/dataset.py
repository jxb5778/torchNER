
import numpy as np

import torch

from api.data import io
from api.data import index
from api.data import prepare

from api.utils import util





class NERCorpus:

    def __init__(self, train_file: str = None, dev_file: str = None, test_file: str = None):

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self._reader = io.NERFileReader()
        self._tagger = index.TagIndexer()
        self._vocab = index.VocabIndexer()

    def train_data(self):
        sentence_list, tag_list = self._reader(self.train_file)
        return sentence_list, tag_list

    def dev_data(self):
        sentence_list, tag_list = self._reader(self.dev_file)
        return sentence_list, tag_list

    def test_data(self):
        sentence_list, tag_list = self._reader(self.test_file)
        return sentence_list, tag_list

    def make_tag_dictionary(self):

        train_sentences, train_tags = self.train_data()
        self._tagger = self._tagger.fit(train_tags)

        return self._tagger.val_to_index

    def make_vocab_dictionary(self):

        train_sentences, train_tags = self.train_data()
        self._vocab = self._vocab.fit(train_sentences)

        return self._vocab.val_to_index

    def gen_data_loader(self, sentence_list, tag_list, params):

        sent_list_padded = util.pad_sequences(sentence_list)
        enc_sent_list = self._vocab.encode_sequences(sent_list_padded)

        tag_list_padded = util.pad_sequences(tag_list)
        enc_tag_list = self._tagger.encode_sequences(tag_list_padded)

        prepared_data = prepare.PrepareTensorData(X=enc_sent_list, y=enc_tag_list)
        data_loader = torch.utils.data.DataLoader(prepared_data, **params)

        return data_loader

    def gen_transformer_data_loader(self, sentence_list, tag_list, params):

        prepared_data = prepare.PrepareTensorData(X=sentence_list, y=tag_list)
        data_loader = torch.utils.data.DataLoader(prepared_data, **params)

        return data_loader
