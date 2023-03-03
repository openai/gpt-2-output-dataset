import itertools
import os
import pandas as pd
from tqdm import tqdm

# Read entropy data with .nll extension
file_name = "data/small-117M.test.model=gpt2.nll"
data = {'entropy': [], 'series_id': []}
with open(file_name, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        entropy = list(map(float, line.split(' ')[:-1]))
        series_id = [i]*len(entropy)
        data['entropy'].append(entropy)
        data['series_id'].append(series_id)

data['entropy'] = itertools.chain.from_iterable(data['entropy'])
data['series_id'] = itertools.chain.from_iterable(data['series_id'])
df = pd.DataFrame.from_dict(data)
print(df.shape)

output_dir = './stat_test'
os.makedirs(output_dir, exist_ok=True)
output_file_name = os.path.basename(file_name) + '.stat.csv'
output_path = os.path.join(output_dir, output_file_name)
df.to_csv(output_path, index=False)