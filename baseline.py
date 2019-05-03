import os
import json

import fire
import numpy as np
from scipy import sparse

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def load_split(data_dir, source, split, n=np.inf):
    webtext = _load_split(data_dir, 'webtext', split, n=n//2)
    gen = _load_split(data_dir, source, split, n=n//2)
    texts = webtext+gen
    labels = [0]*len(webtext)+[1]*len(gen)
    return texts, labels

def main(data_dir, log_dir, source='xl-1542M-k40', n_train=500000, n_valid=10000, n_jobs=None, verbose=False):
    train_texts, train_labels = load_split(data_dir, source, 'train', n=n_train)
    valid_texts, valid_labels = load_split(data_dir, source, 'valid', n=n_valid)
    test_texts, test_labels = load_split(data_dir, source, 'test')

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)
    train_features = vect.fit_transform(train_texts)
    valid_features = vect.transform(valid_texts)
    test_features = vect.transform(test_texts)

    model = LogisticRegression(solver='liblinear')
    params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
    split = PredefinedSplit([-1]*n_train+[0]*n_valid)
    search = GridSearchCV(model, params, cv=split, n_jobs=n_jobs, verbose=verbose, refit=False)
    search.fit(sparse.vstack([train_features, valid_features]), train_labels+valid_labels)
    model = model.set_params(**search.best_params_)
    model.fit(train_features, train_labels)
    valid_accuracy = model.score(valid_features, valid_labels)*100.
    test_accuracy = model.score(test_features, test_labels)*100.
    data = {
        'source':source,
        'n_train':n_train,
        'valid_accuracy':valid_accuracy,
        'test_accuracy':test_accuracy
    }
    print(data)
    json.dump(data, open(os.path.join(log_dir, f'{source}.json'), 'w'))

if __name__ == '__main__':
    fire.Fire(main)
