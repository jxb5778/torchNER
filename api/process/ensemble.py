
import os

import statistics

from api.data import io


def ensemble_model_results(input_directory: str, outfile: str):

    reader = io.NERPredFileReader()

    sent_list = None
    tag_list = None

    pred_list = list()

    file_list = [f'{input_directory}/{file}' for file in os.listdir(input_directory)]

    for file in file_list:

        file_sent_list, file_tag_list, file_pred_list = reader(file, sep=' ')

        if sent_list is None:
            sent_list = file_sent_list

        if tag_list is None:
            tag_list = file_tag_list

        pred_list.append(file_pred_list)

    ensemble_pred_list = list()

    for sent_idx in range(len(sent_list)):

        sent_preds = [pred[sent_idx] for pred in pred_list]

        len_sentence = len(sent_preds[0])

        sent_tok_preds = list()

        for tok_idx in range(len_sentence):
            tok_preds = [pred[tok_idx] for pred in sent_preds]
            sent_tok_preds.append(tok_preds)

        ensemble_preds = [statistics.mode(tok_preds) for tok_preds in sent_tok_preds]
        ensemble_pred_list.append(ensemble_preds)

    writer = io.NERPredFileWriter()

    writer(file_path=outfile, sentence_list=sent_list, tag_list=tag_list, pred_list=ensemble_pred_list)

    return sent_list, tag_list, ensemble_pred_list
