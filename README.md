# gpt-2-output-dataset

This dataset contains:
- 250K documents from the WebText test set
- For each GPT-2 model (trained on the WebText training set), 250K random samples (temperature 1, no truncation) and 250K samples generated with Top-K 40 truncation

We look forward to the research produced using this data!

### Download

For each model, we have a training split of 250K generated examples, as well as validation and test splits of 5K examples.

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

We're interested in seeing research in detectability of GPT-2 model family generations.

We've provided a starter baseline which trains a logistic regression detector on TF-IDF unigram and bigram features, in `baseline.py`.

| Model | Temperature 1 | Top-K 40 |
| ----- | ------ | ------ |
| 117M  | 88.29% | 96.79% |
| 345M  | 88.94% | 95.22% |
| 762M  | 77.16% | 94.43% |
| 1542M | 74.31% | 92.69% |

### Initial Analysis

<img src="https://i.imgur.com/PZ3GOeS.png" width="475" height="335" title="Impact of Document Length">

Shorter documents are harder to detect. Accuracy of detection of a short documents of 500 characters (a long paragraph) is about 15% lower.

<img src="https://i.imgur.com/eH9Ogqo.png" width="482" height="300" title="Part of Speech Analysis">

Truncated sampling, which is commonly used for high-quality generations from the GPT-2 model family, results in a shift in the part of speech distribution of the generated text compared to real text. A clear example is the underuse of proper nouns and overuse of pronouns which are more generic. This shift contributes to the 8% to 18% higher detection rate of Top-K samples compared to random samples across models.

### Data removal requests

If you believe your work is included in WebText and would like us to remove it, please let us know at webtextdata@openai.com.
