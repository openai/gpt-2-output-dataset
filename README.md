# gpt-2-output-dataset

This dataset contains:
- 250K documents from the WebText test set
- For each GPT-2 model (trained on the WebText training set), 250K random samples (temperature 1, no truncation) and 250K samples generated with Top-K 40 truncation

We look forward to the research produced using this data!

### Download

For each model, we have a training split of 250K generated examples, as well as validation and test splits of 5K examples.

All data is located in Google Cloud Storage, under the directory `gs://gpt-2/output-dataset/v1`.

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

#### Finetuned model samples

Additionally, we encourage research on detection of finetuned models.  We have released data under `gs://gpt-2/output-dataset/v1-amazonfinetune/` with samples from a GPT-2 full model finetuned to output Amazon reviews.

### Detectability baselines

We're interested in seeing research in detectability of GPT-2 model family generations.

We provide some [initial analysis](detection.md) of two baselines, as well as [code](./baseline.py) for the better baseline.

Overall, we are able to achieve accuracies in the mid-90s for Top-K 40 generations, and mid-70s to high-80s (depending on model size) for random generations.  We also find some evidence that adversaries can evade detection via finetuning from released models.

### Data removal requests

If you believe your work is included in WebText and would like us to remove it, please let us know at webtextdata@openai.com.
