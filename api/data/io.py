
class NERFileReader:

    def __call__(self, file_path, sep='\t'):

        file_word_list = list()
        file_tag_list = list()

        with open(file_path, 'r') as f:
            word_list = list()
            tag_list = list()

            for line in f:
                if line == '\n':
                    file_word_list.append(word_list)
                    file_tag_list.append(tag_list)

                    word_list = list()
                    tag_list = list()

                else:
                    word, tag = line.split(sep)

                    word_list.append(word.strip())
                    tag_list.append(tag.strip())

        return file_word_list, file_tag_list


class NERPredFileReader:

    def __call__(self, file_path, sep='\t'):

        file_word_list = list()
        file_tag_list = list()
        file_pred_list = list()

        with open(file_path, 'r') as f:
            word_list = list()
            tag_list = list()
            pred_list = list()

            for line in f:
                if line == '\n':
                    file_word_list.append(word_list)
                    file_tag_list.append(tag_list)
                    file_pred_list.append(pred_list)

                    word_list = list()
                    tag_list = list()
                    pred_list = list()

                else:
                    word, tag, pred = line.split(sep)

                    word_list.append(word.strip())
                    tag_list.append(tag.strip())
                    pred_list.append(pred.strip())

        return file_word_list, file_tag_list, file_pred_list


class NERFileWriter:

    def __call__(self, file_path: str, sentence_list: list, tag_list: list, sep='\t'):

        with open(file_path, 'w') as f:

            for sentence, tags in zip(sentence_list, tag_list):
                for word, tag in zip(sentence, tags):
                    f.write(f'{word}{sep}{tag}\n')

                f.write('\n')


class NERPredFileWriter:

    def __call__(self, file_path: str, sentence_list: list, tag_list: list, pred_list: list, sep='\t'):

        with open(file_path, 'w') as f:

            for sentence, tags, preds in zip(sentence_list, tag_list, pred_list):
                for word, tag, pred in zip(sentence, tags, preds):
                    f.write(f'{word}{sep}{tag}{sep}{pred}\n')

                f.write('\n')
