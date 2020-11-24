
from seqeval import metrics

from api.data import io


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3/test/'

input_file = f'{DIRECTORY}/test-ensemble.tsv'

reader = io.NERPredFileReader()

sent_list, tag_list, pred_list = reader(input_file)

print("F1: ", metrics.f1_score(tag_list, pred_list, mode='strict'), "\n")
print(metrics.classification_report(tag_list, pred_list, mode='strict', digits=5))
