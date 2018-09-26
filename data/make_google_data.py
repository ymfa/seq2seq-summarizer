"""
Download and tokenize the Google Sentence Compression Data. This script will create the data file
`sentences.txt` for use in the summarizer, and a directory `google-sentence-compression-data` to
hold the downloaded raw files.

One may use shell commands to randomly split `sentences.txt` into training, validation, and testing
sets. I use 80% training (sent.txt) and 20% validation (sent.val.txt).

About this dataset: https://github.com/google-research-datasets/sentence-compression
"""
import gzip
import json
import unicodedata
import re
import os
import urllib.request
from nltk import word_tokenize


splitter = re.compile(r'(-)')
contractions = {"'s", "'d", "'ve", "'ll", "'m", "'re"}


print_every = 1000  # print progress every 1000 sentences
data_path = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(data_path, 'google-sentence-compression-data')
if not os.path.isdir(corpus_path):
  os.mkdir(corpus_path)


def tokenize(text):
  # de-accent and lower
  text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
  text = unicodedata.normalize('NFC', text).lower()
  # split hyphens
  tokens = []
  for token in word_tokenize(text):
    if '-' in token and not '--' in token:
      tokens.extend(t for t in splitter.split(token) if t)
    else:
      tokens.append(token)
  # separate leading apostrophe from words e.g. "'apple"
  new_tokens = []
  for token in tokens:
    if len(token) > 1 and token.startswith("'") and "''" not in token \
            and token not in contractions:
      new_tokens.append("'")
      new_tokens.append(token[1:])
    else:
      new_tokens.append(token)
  return ' '.join(new_tokens)


count = 0
with open(os.path.join(data_path, 'sentences.txt'), 'wt') as fout:
  for volume_id in range(1, 11):
    filename = 'sent-comp.train%02d.json.gz' % volume_id
    file_path = os.path.join(corpus_path, filename)
    if not os.path.isfile(file_path):
      url = "https://github.com/google-research-datasets/sentence-compression/raw/master/data/" \
            + filename
      print("Downloading %s..." % url)
      urllib.request.urlretrieve(url, file_path)
    print("Processing %s..." % filename)
    with gzip.open(file_path, 'rt', encoding='utf-8') as fin:
      lines = []
      for line in fin:
        line = line.strip()
        if not line:
          if lines:
            obj = json.loads('\n'.join(lines))
            original = obj['source_tree']['sentence']
            summary = obj['headline']
            entry = '%s\t%s' % (tokenize(original), tokenize(summary))
            fout.write(entry + "\n")
            count += 1
            if count % print_every == 0:
              print(count)
            lines = []
        else:
          lines.append(line)
