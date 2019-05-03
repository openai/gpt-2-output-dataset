# gpt-2-output-dataset

This dataset contains:
- 250K samples from the WebText test set
- For each GPT-2 model (trained on the WebText training set), 250K plain samples (temperature 1, no truncation) and 250K samples generated with top-k 40 truncation

We look forward to the research produced using this data!

### Download

For each, we have a training split of 250K samples, as well as validation and test splits of 5K samples.

For each model, we're releasing temperature 1 samples

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

We've provided a baseline of logistic regression on tf-idf, in `baseline.py`.

### Data removal requests

If you believe your work is included in our dataset and would like us to remove it, please let us know at webtextdata@openai.com.
