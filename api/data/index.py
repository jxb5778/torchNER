
import api.utils as utils


class SequenceIndexer:

    def __init__(self, unknown_tok=utils.UNK_TOK, pad_tok=utils.PAD_TOK):
        self.val_to_index = {pad_tok: 0, unknown_tok: 1, utils.START_TAG: 2, utils.STOP_TAG: 3}
        self.index_to_val = {0: pad_tok, 1: unknown_tok, 2: utils.START_TAG, 3: utils.STOP_TAG}

        self.unknown_tok = unknown_tok
        self.pad_tok = pad_tok

        self.start_tag = utils.START_TAG
        self.stop_tag = utils.STOP_TAG

    def values(self):
        return list(self.val_to_index.keys())

    def indexes(self):
        return self.index_to_val.keys()

    def fit(self, data):

        for sequence in data:
            for val in sequence:
                if val not in self.val_to_index:
                    self.val_to_index[val] = len(self.val_to_index)
                    self.index_to_val[self.val_to_index[val]] = val

        return self

    def _encode_sequence_value(self, val):

        enc_val = None

        if val in self.val_to_index.keys():

            enc_val = self.val_to_index[val]

        elif self.unknown_tok in self.val_to_index.keys():
            enc_val = self.val_to_index[self.unknown_tok]

        else:
            self.val_to_index[self.unknown_tok] = len(self.val_to_index)
            enc_val = self.val_to_index[self.unknown_tok]

        return enc_val

    def encode_sequences(self, sequence_list):

        seq_encoded = [
            [self._encode_sequence_value(val) for val in seq] for seq in sequence_list
        ]

        return seq_encoded

    def _decode_sequence(self, seq):
        sequence = [
            self.index_to_val[ix] if ix in self.index_to_val.keys() else self.unknown_tok for ix in seq
        ]

        return sequence

    def decode_sequences(self, sequence_list):

        seq_decoded = [self._decode_sequence(seq) for seq in sequence_list]

        return seq_decoded


class VocabIndexer(SequenceIndexer):

    def words(self):
        return self.values()

    def word_to_ix(self, word):
        return self.val_to_index[word]

    def ix_to_word(self, ix):
        return self.index_to_val[ix]


class TagIndexer(SequenceIndexer):

    def tags(self):
        return self.values()

    def tag_to_ix(self, tag):
        return self.val_to_index[tag]

    def ix_to_tag(self, ix):
        return self.index_to_val[ix]
