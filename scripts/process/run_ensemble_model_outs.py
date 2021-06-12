
from api.process import ensemble


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 695/Proj3/test'

input_dir = f'{DIRECTORY}/ensembles'
outfile = f'{DIRECTORY}/test-ensemble_ensemble.tsv'

ensemble_data = ensemble.ensemble_model_results(input_dir, outfile, sep='\t')
