
import torch

from api.utils import const


def argmax(vec):
    """
    credit pytorch advanced tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    """
    credit pytorch advanced tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    # Compute log sum exp in a numerically stable way for the forward algorithm
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def upsample_tags(upsample_tags: list, sentence_list: list, tag_list: list):

    upsample_sentence_list = list()
    upsample_tag_list = list()

    for sentence, tags in zip(sentence_list, tag_list):
        if any(tag in tags for tag in upsample_tags):
            upsample_sentence_list.append(sentence)
            upsample_tag_list.append(tags)

    sentence_list = sentence_list + upsample_sentence_list
    tag_list = tag_list + upsample_tag_list

    return sentence_list, tag_list


def _pad_sequence(sequence, max_length, pad_tok, padding_pos='post'):

    seq_len = len(sequence)
    padding = [pad_tok for idx in range(max_length - seq_len)]

    if padding_pos == 'pre':
        padded_seq = padding + sequence

    else:
        padded_seq = sequence + padding

    return padded_seq


def pad_sequences(sequence_list, pad_tok=const.PAD_TOK):

    seq_lengths = [len(seq) for seq in sequence_list]
    max_length = max(seq_lengths)

    padded_seq = [_pad_sequence(seq, max_length, pad_tok) for seq in sequence_list]

    return padded_seq
