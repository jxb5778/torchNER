
import torch
import torch.nn as nn
from torch.nn.utils import rnn

import bi_lstm_crf as blc

import numpy as np

import api.utils as utils


class TransformerBiLSTM_CRF(nn.Module):
    """
        Credit
        - pytorch advanced tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
        - bi-lstm-crf: https://github.com/jidasheng/bi-lstm-crf
    """

    def __init__(self,
                 transformer_model: str,
                 tag_to_ix,
                 embedding_dim: int = 32,
                 hidden_dim: int = 256,
                 dropout_ratio: int = 0.5
                 ):

        super(TransformerBiLSTM_CRF, self).__init__()

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio

        self.embeddings, self.tokenizer = utils.transformer_from_pretrained(transformer_model)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.dropout = nn.Dropout(dropout_ratio)

        self.crf = blc.CRF(hidden_dim, self.tagset_size)

    def _get_lstm_features(self, sentences):

        masks = np.array([
            [tok == utils.PAD_TOK for tok in sent]
            for sent in sentences
        ])

        masks = torch.from_numpy(masks)

        sentences = [' '.join(tok for tok in sent) for sent in sentences]

        tok_sentences = self.tokenizer(sentences, return_tensors="pt", padding=True)

        embeds, _ = self.embeddings(**tok_sentences)

        seq_length = masks.sum(1)

        print("Seq length: ", seq_length)

        pack_sequence = rnn.pack_padded_sequence(embeds, lengths=seq_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(pack_sequence)
        lstm_out, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        return lstm_out, masks

    def loss(self, sentence, tags):

        features, masks = self._get_lstm_features(sentence)
        loss = self.crf.loss(features, tags, masks=masks)

        return loss

    def forward(self, sentence):
        lstm_feats, masks = self._get_lstm_features(sentence)
        scores, tag_seq = self.crf(lstm_feats, masks)

        return scores, tag_seq

