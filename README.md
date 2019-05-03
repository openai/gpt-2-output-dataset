# gpt-2-output-dataset

This dataset contains:
- 250K samples from the WebText test set
- For each GPT-2 model (trained on the WebText training set), 250K plain samples (temperature 1, no truncation) and 250K samples generated with top-k 40 truncation

We look forward to the research produced using this data!

### Download

For each, we have a training split of 500K total examples, as well as validation and test splits of 10K examples.

All data is located in Google Cloud Storage, at under the directory `gs://gpt-2/output-dataset/v1`.

There, you will find files:

- `webtext.${split}.jsonl`
- `small-117M.${split}.jsonl`
- `small-117M-k40.${split}.jsonl`
- `medium-345M.${split}.jsonl`
- `medium-345M-k40.${split}.jsonl`
- `large-762M.${split}.jsonl`
- `large-762M-k40.${split}.jsonl`
- `xl-1542M.${split}.jsonl`
- `xl-1542M-k40.${split}.jsonl`

where split is one of `train`, `test`, and `valid`.

We've provided a script to download all of them, in `download_dataset.py`.

### Detectability baselines

We're interested in seeing research in detectability of our model generations.

We've provided a starter baseline which trains a logistic regression detector on TF-IDF unigram and bigram features, in `baseline.py`.

| Model | Temperature 1 | Top-K 40 |
| ----- | ------ | ------ |
| 117M  | 88.29% | 96.79% |
| 345M  | 88.94% | 95.22% |
| 762M  | 77.16% | 94.43% |
| 1542M | 74.31% | 92.69% |

### Data removal requests

If you believe your work is included in our dataset and would like us to remove it, please let us know at webtextdata@openai.com.
