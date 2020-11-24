
import api


if __name__ == '__main__':

    DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3'

    corpus = api.data.NERCorpus(
        train_file=f'{DIRECTORY}/data/train.txt',
        dev_file=f'{DIRECTORY}/data/dev.txt',
        test_file=f'{DIRECTORY}/data/test.txt'
    )

    tag_to_ix = corpus.make_tag_dictionary()
    word_to_ix = corpus.make_vocab_dictionary()

    tagger = api.models.BiLSTM_CRF(
        hidden_dim=256,
        embedding_dim=32,
        vocab_size=len(word_to_ix),
        tag_to_ix=tag_to_ix
    )

    trainer = api.learn.NERTrainer(tagger, corpus)

    trainer.train(
        base_path=f'{DIRECTORY}/log/test/',
        mini_batch_size=32,
        max_epochs=5,
        num_workers=2
    )
