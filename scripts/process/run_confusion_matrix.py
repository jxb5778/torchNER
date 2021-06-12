
from sklearn.metrics import confusion_matrix

import plotly.express as px
from plotly.offline import plot

from api.data import io


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3/test/'

input_file = f'{DIRECTORY}/test-ensemble_3.tsv'

reader = io.NERPredFileReader()

sent_list, tag_list, pred_list = reader(input_file)

pred_list = [pred for taglist in pred_list for pred in taglist]
tag_list = [tag for taglist in tag_list for tag in taglist]

labels = list(set(tag_list))
print(labels)

conf_matrix = confusion_matrix(tag_list, pred_list, labels=labels, normalize='true')

print(conf_matrix)

fig = px.imshow(
    conf_matrix,
    labels=dict(x='Predicted', y='Ground Truth', color='Percent'),
    x=labels,
    y=labels
)

fig.update_xaxes(side="top")

plot(fig)
