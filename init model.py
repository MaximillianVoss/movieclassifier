# -*- coding: utf-8 -*-
"""movieclassifier_updated.py

This script is based on the code provided in the Google Colab notebook.
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
import dill
import requests
import tarfile
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Download and extract dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
response = requests.get(url)
with open('aclImdb_v1.tar.gz', 'wb') as f:
    f.write(response.content)

with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()

# Process and save dataset
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
dfs = []

for s in ['test', 'train']:
    for l in ['pos', 'neg']:
        path = os.path.join(basepath, s, l)
        files = sorted(os.listdir(path))
        data = []
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            data.append({'review': txt, 'sentiment': labels[l]})
        dfs.append(pd.DataFrame(data))

df = pd.concat(dfs, ignore_index=True)

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
df = pd.read_csv('movie_data.csv', encoding='utf-8')


# Preprocessing and tokenizing
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    if not isinstance(text, str):
        return []
    return text.split()


df['review'] = df['review'].apply(preprocessor)


def tokenizer_porter(text):
    if not isinstance(text, str):
        return []
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


nltk.download('stopwords')
stop = stopwords.words('english')

# Training the model
x_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer_porter)

sgd = SGDClassifier(loss='log', n_jobs=-1, random_state=1)

clf = Pipeline([
    ('vect', tfidf),
    ('clf', sgd),
])

clf.fit(x_train, y_train)

# Saving the model
dest = 'pkl_objects'
if not os.path.exists(dest):
    os.makedirs(dest)

dill.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'))
dill.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'))
