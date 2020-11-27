
from transformers import AutoTokenizer, AutoModel

import api


def transformer_from_pretrained(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    embedding = AutoModel.from_pretrained(model)

    return embedding, tokenizer


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3'

corpus = api.data.NERCorpus(
        train_file=f'{DIRECTORY}/data/train.txt',
        dev_file=f'{DIRECTORY}/data/dev.txt',
        test_file=f'{DIRECTORY}/data/test.txt'
    )

train_sent_list, train_tags = corpus.train_data()

emb, tok = transformer_from_pretrained('dmis-lab/biobert-v1.1')

train_sent_list = [' '.join(tok for tok in train_sent) for train_sent in train_sent_list]

train_sent_list = tok(train_sent_list[:5], return_tensors="pt", padding=True)

pt_outputs = emb(**train_sent_list)

print(pt_outputs[0])
print(pt_outputs[1])
