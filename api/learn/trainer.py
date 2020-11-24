
import torch
import torch.optim as optim

import numpy as np

from tqdm import tqdm


class NERTrainer:

    def __init__(self, tagger, corpus):

        self.tagger = tagger
        self.corpus = corpus

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              shuffle: bool = True,
              max_epochs: int = 10,
              num_workers: int = 1
              ):

        optimizer = optim.Adam(self.tagger.parameters(), lr=learning_rate)

        params = {
            'batch_size': mini_batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        train_sentences, train_tags = self.corpus.train_data()
        train_data_loader = self.corpus.gen_data_loader(train_sentences, train_tags, params)

        dev_sentences, dev_tags = self.corpus.dev_data()
        dev_data_loader = self.corpus.gen_data_loader(dev_sentences, dev_tags, params)

        test_sentences, test_tags = self.corpus.test_data()
        test_data_loader = self.corpus.gen_data_loader(test_sentences, test_tags, params)

        for epoch in range(max_epochs):
            print(f"Starting epoch {epoch + 1}...")

            epoch_train_loss_list = list()

            for train_data in tqdm(train_data_loader):
                self.tagger.zero_grad()

                train_loss = self._train_step(train_data)
                epoch_train_loss_list.append(train_loss.item())

                train_loss.backward()
                optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss_list)

            epoch_dev_loss_list = list()

            for dev_data in tqdm(dev_data_loader):
                dev_loss = self._evaluate(dev_data)
                epoch_dev_loss_list.append(dev_loss.item())

            epoch_dev_loss = np.mean(epoch_dev_loss_list)

            epoch_report = "\n------------------------------------------------------------\n" \
                           f"Epoch {epoch + 1} done: lr {learning_rate}\n" \
                           f"Train loss {epoch_train_loss} - Dev loss {epoch_dev_loss}\n" \
                           "------------------------------------------------------------\n"

            print(epoch_report)

    def _train_step(self, data_loader):

        batch_sents, batch_tags = data_loader
        batch_sents = batch_sents.to(torch.int64)
        batch_tags = batch_tags.to(torch.int64)

        loss = self.tagger.loss(batch_sents, batch_tags)

        return loss

    def _evaluate(self, data_loader):

        with torch.no_grad():
            batch_sents, batch_tags = data_loader
            batch_sents = batch_sents.to(torch.int64)
            batch_tags = batch_tags.to(torch.int64)

            eval_loss = self.tagger.loss(batch_sents, batch_tags)

            return eval_loss


class NERTransformerTrainer:

    def __init__(self, tagger, corpus):

        self.tagger = tagger
        self.corpus = corpus

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              shuffle: bool = True,
              max_epochs: int = 10,
              num_workers: int = 1
              ):

        optimizer = optim.Adam(self.tagger.parameters(), lr=learning_rate)

        params = {
            'batch_size': mini_batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        train_sentences, train_tags = self.corpus.train_data()
        train_data_loader = self.corpus.gen_data_loader(train_sentences, train_tags, params)

        dev_sentences, dev_tags = self.corpus.dev_data()
        dev_data_loader = self.corpus.gen_data_loader(dev_sentences, dev_tags, params)

        test_sentences, test_tags = self.corpus.test_data()
        test_data_loader = self.corpus.gen_data_loader(test_sentences, test_tags, params)

        for epoch in range(max_epochs):
            print(f"Starting epoch {epoch + 1}...")

            epoch_train_loss_list = list()

            for train_data in tqdm(train_data_loader):
                self.tagger.zero_grad()

                train_loss = self._train_step(train_data)
                epoch_train_loss_list.append(train_loss.item())

                train_loss.backward()
                optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss_list)

            epoch_dev_loss_list = list()

            for dev_data in tqdm(dev_data_loader):
                dev_loss = self._evaluate(dev_data)
                epoch_dev_loss_list.append(dev_loss.item())

            epoch_dev_loss = np.mean(epoch_dev_loss_list)

            epoch_report = "\n------------------------------------------------------------\n" \
                           f"Epoch {epoch + 1} done: lr {learning_rate}\n" \
                           f"Train loss {epoch_train_loss} - Dev loss {epoch_dev_loss}\n" \
                           "------------------------------------------------------------\n"

            print(epoch_report)

    def _train_step(self, data_loader):

        batch_sents, batch_tags = data_loader
        batch_sents = batch_sents.to(torch.int64)
        batch_tags = batch_tags.to(torch.int64)

        loss = self.tagger.loss(batch_sents, batch_tags)

        return loss

    def _evaluate(self, data_loader):

        with torch.no_grad():
            batch_sents, batch_tags = data_loader
            batch_sents = batch_sents.to(torch.int64)
            batch_tags = batch_tags.to(torch.int64)

            eval_loss = self.tagger.loss(batch_sents, batch_tags)

            return eval_loss
