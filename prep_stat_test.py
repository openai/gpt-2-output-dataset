import pandas as pd
from tqdm import tqdm

# Read entropy data with .nll extension
file_name = "data/small-117M.test.model=gpt2.nll"
data = {'entropy': [], 'series_id': []}
with open(file_name, 'r') as f:
    for i, line in enumerate(f):
        entropy = list(map(float, line.split(' ')))
        series_id = [i]*len(entropy)
        data['entropy'] = data['entropy'] + entropy
        data['series_id'] = data['series_id'] + series_id
