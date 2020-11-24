
import torch.nn as nn
from torch.nn.utils import rnn

import bi_lstm_crf as blc


class BiLSTM_CRF(nn.Module):
    """
        Credit
        - pytorch advanced tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
        - bi-lstm-crf: https://github.com/jidasheng/bi-lstm-crf
    """

    def __init__(self,
                 vocab_size,
                 tag_to_ix,
                 embedding_dim: int = 32,
                 hidden_dim: int = 256,
                 dropout_ratio: int = 0.5
                 ):

        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio

        self.tag_to_ix = tag_to_ix
        self.vocab_size = vocab_size
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.crf = blc.CRF(hidden_dim, self.tagset_size)

    def _get_lstm_features(self, sentences):

        masks = sentences.gt(0)
        embeds = self.word_embeds(sentences.long())

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = rnn.pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.lstm(pack_sequence)
        lstm_out, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, sentence, tags):

        features, masks = self._get_lstm_features(sentence)
        loss = self.crf.loss(features, tags, masks=masks)

        return loss

    def forward(self, sentence):
        lstm_feats, masks = self._get_lstm_features(sentence)
        scores, tag_seq = self.crf(lstm_feats, masks)

        return scores, tag_seq
