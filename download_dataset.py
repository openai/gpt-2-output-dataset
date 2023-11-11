import os
import sys
import requests
from tqdm import tqdm

subdir = 'data'import requests
import json
import os 
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Union, Callable
from tqdm import tqdm 

@dataclass
class ChatData:
    id: int
    ended: bool
    length: int
    text: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatData':
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ChatDataEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, ChatData):
            return obj.to_dict()
        return super().default(obj)

class GPTData:
    def __init__(self, target_examples: Union[int, None] = None, output_file: str = "output.jsonl") -> None:
        self.target_examples = target_examples
        self.output_file = output_file
        self.chat_data_list: List[ChatData] = []

    @staticmethod
    def validate_data_size(data_size: str) -> None:
        valid_data_sizes = [
            'webtext',
            'small-117M', 'small-117M-k40',
            'medium-345M', 'medium-345M-k40',
            'large-762M', 'large-762M-k40',
            'xl-1542M', 'xl-1542M-k40',
        ]
        if data_size not in valid_data_sizes:
            raise ValueError(f"Invalid data size: {data_size}. Valid options are {valid_data_sizes}")

    @staticmethod
    def validate_split(split: str) -> None:
        valid_splits = ['train', 'valid', 'test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Valid options are {valid_splits}")

    def download_and_save_data(self, data_size_fn: Union[str, List[str], Callable], split_fn: Union[str, List[str], Callable]) -> None:
        base_url = "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
        
        data_sizes = data_size_fn() if callable(data_size_fn) else ([data_size_fn] if isinstance(data_size_fn, str) else data_size_fn)
        splits = split_fn() if callable(split_fn) else ([split_fn] if isinstance(split_fn, str) else split_fn)

        for data_size in data_sizes:
            self.validate_data_size(data_size) 

            for split in splits:
                self.validate_split(split) 

                data_url = f"{base_url}{data_size}.{split}.jsonl"
                response = requests.get(data_url, stream=True)
                response.raise_for_status()

                if os.path.exists(self.output_file) and os.path.getsize(self.output_file) == int(response.headers['Content-Length']):
                    print(f"Skipping download for {data_size}.{split}.jsonl. Local file matches the remote.")
                    return

                with tqdm(total=int(response.headers['Content-Length']), unit='B', unit_scale=True, desc=f'Downloading {data_size}.{split}.jsonl') as pbar, open(self.output_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):  # Use a larger chunk size for a speed boost.
                        if chunk:
                            file.write(chunk)
                            file.flush()
                            pbar.update(len(chunk))

        self.chat_data_list = self.read_chat_data()
        if self.target_examples is not None and len(self.chat_data_list) >= self.target_examples:
            self.chat_data_list = self.chat_data_list[:self.target_examples]

        self.save_chat_data()

    def read_chat_data(self) -> List[ChatData]:
        chat_data_list = []
        with open(self.output_file, "r", encoding="utf-8") as file:
            for line in file:
                data_dict = json.loads(line)
                chat_data = ChatData.from_dict(data_dict)
                chat_data_list.append(chat_data)
        return chat_data_list

    def save_chat_data(self) -> None:
        with open(self.output_file, "w", encoding="utf-8") as file:
            for chat_data in self.chat_data_list:
                file.write(json.dumps(chat_data, cls=ChatDataEncoder) + "\n")

gpt_data = GPTData(target_examples=None)  # Can be left as None, or you can provide an integer based on how much data you want.
gpt_data.download_and_save_data(data_size_fn='webtext', split_fn='train')

if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

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

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
