import os
import sys
import requests
from tqdm import tqdm

subdir = 'data'
os.makedirs(subdir, exist_ok=True)

for ds in [
    'webtext',
    'small-117M',  'small-117M-k40',
    'medium-345M', 'medium-345M-k40',
    'large-762M',  'large-762M-k40',
    'xl-1542M',    'xl-1542M-k40',
]:
    for split in ['train', 'valid', 'test']:
        filename = ds + "." + split + '.jsonl'
        r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)
        r.raise_for_status()

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=4194304):
                    f.write(chunk)
                    pbar.update(len(chunk))
