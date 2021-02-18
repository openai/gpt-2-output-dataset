import os

import requests
import torch.distributed as dist
from tqdm import tqdm

from .utils import distributed

ALL_DATASETS = [
    'webtext',
    'small-117M',  'small-117M-k40',  'small-117M-nucleus',
    'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
    'large-762M',  'large-762M-k40',  'large-762M-nucleus',
    'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
]


def download(*datasets, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)

    if distributed() and dist.get_rank() > 0:
        dist.barrier()

    for ds in datasets:
        assert ds in ALL_DATASETS, f'Unknown dataset {ds}'

        for split in ['train', 'valid', 'test']:
            filename = ds + "." + split + '.jsonl'
            output_file = os.path.join(data_dir, filename)
            if os.path.isfile(output_file):
                continue

            r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(output_file, 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

    if distributed() and dist.get_rank() == 0:
        dist.barrier()


if __name__ == '__main__':
    download(*ALL_DATASETS)
