
import api


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3'

corpus = api.data.NERCorpus(
    train_file=f'{DIRECTORY}/data/train.txt',
    dev_file=f'{DIRECTORY}/data/dev.txt',
    test_file=f'{DIRECTORY}/data/test.txt'
)

upsample_tags = [
        'B-Generic-Measure',
        'B-Numerical',
        'B-Size'
    ]

train_sentences, train_tags = corpus.train_data()
train_sentences, train_tags = api.utils.upsample_tags(upsample_tags, train_sentences, train_tags)

print('Number of sentences after upsample: ', len(train_sentences))

writer = api.data.NERFileWriter()
writer(
    file_path=f'{DIRECTORY}/data/train_upsample-genmeasure-num-size.txt',
    sentence_list=train_sentences,
    tag_list=train_tags
)
