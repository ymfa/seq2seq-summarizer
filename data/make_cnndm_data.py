"""
Pre-process the CNN/Daily Mail dataset. Before using this script, please download the following
files and put all of them under `data/cnndm`:

* cnn_stories_tokenized.zip, dm_stories_tokenized.zip -- These can be obtained from
  https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
* all_test.txt, all_train.txt, all_val.txt -- These are the indices of documents in See et al's
  training/validation/testing sets, used here to ensure the same data split. They can be found in
  https://github.com/abisee/cnn-dailymail/tree/master/url_lists

This script will generate `cnndm.gz`, `cnndm.val.gz`, and `cnndm.test.gz`. Each file is a gzipped
text file containing one example per line.
"""
import re
import os
import gzip
from zipfile import ZipFile
from hashlib import sha1


splitter = re.compile(r'(-)')
word_recognizer = re.compile(r'^\w[\w\-]+\w$')
contractions = {"s", "d", "ve", "ll", "m", "re", "em"}
ptb_unescape = {'-LRB-': '(', '-RRB-': ')', '-LCB-': '{', '-RCB-': '}'}

print_every = 1000  # print progress every 1000 documents
data_path = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(data_path, 'cnndm')


def split_example(filename: str, data: str, eop: str='<P>') -> tuple:
  text, summary = [], []
  highlight_mode = False
  for paragraph in data.split('\n\n'):
    if paragraph == '@highlight':
      highlight_mode = True
    else:
      original_tokens = paragraph.split()
      tokens, next_prefix = [], None
      for i, tok in enumerate(original_tokens):
        if tok == '¿':  # convert ¿ into '
          if i + 1 < len(original_tokens):
            if original_tokens[i+1] == 't' and len(tokens) > 0 and tokens[-1][-1] == 'n':
              tokens[-1] = tokens[-1][:-1]
              next_prefix = "n'"
            elif original_tokens[i+1] in contractions:
              next_prefix = "'"
            elif len(tokens) > 0 and tokens[-1] == 'o':  # o ' clock => o'clock
              tokens.pop()
              next_prefix = "o'"
            elif len(tokens) > 0 and tokens[-1] == 'y':  # y ' all => y' all
              tokens[-1] = "y'"
            else:
              tokens.append("'")
          else:
            tokens.append("'")
        elif tok in ptb_unescape:
          assert next_prefix is None
          tokens.append(ptb_unescape[tok])
        elif tok == '|':
          assert next_prefix is None
        else:
          tok = tok.lower()
          if next_prefix is not None:
            tok = next_prefix + tok
          if tok == '-':
            tokens.append('--')
          elif '-' in tok and not '--' in tok and word_recognizer.match(tok):
            tokens.extend(t for t in splitter.split(tok) if t)
          else:
            tokens.append(tok)
          next_prefix = None
      if not tokens:
        continue  # skip empty paragraphs
      if eop: tokens.append(eop)
      if highlight_mode is False:
        text.extend(tokens)
      else:
        if highlight_mode is True:
          summary.extend(tokens)
          highlight_mode = None
        else:
          print("A paragraph in %s is dropped because it is not text or summary." % filename)
  return text, summary


def get_story_set(filename: str) -> set:
  story_names = set()
  with open(os.path.join(corpus_path, filename), 'rb') as f:
    for line in f:
      story_names.add(sha1(line.strip()).hexdigest())
  return story_names


train_set = get_story_set('all_train.txt')
valid_set = get_story_set('all_val.txt')
test_set = get_story_set('all_test.txt')
train_out = gzip.open(os.path.join(data_path, 'cnndm.gz'), 'wt')
valid_out = gzip.open(os.path.join(data_path, 'cnndm.val.gz'), 'wt')
test_out = gzip.open(os.path.join(data_path, 'cnndm.test.gz'), 'wt')

count = 0
for download_file in ['cnn_stories_tokenized.zip', 'dm_stories_tokenized.zip']:
  with ZipFile(os.path.join(corpus_path, download_file), 'r') as archive:
    for filename in archive.namelist():
      if not filename.endswith('.story'): continue
      story_name = filename[-46:-6]
      if story_name in train_set:
        fout = train_out
      elif story_name in valid_set:
        fout = valid_out
      elif story_name in test_set:
        fout = test_out
      else:
        print("Error: filename %s is not found in train, valid, or test set." % filename)
        continue
      with archive.open(filename, 'r') as f:
        content = f.read().decode('utf-8')
        text, summary = split_example(filename, content)
        if not text:
          print("Skipped: %s has no text." % filename)
          continue
        if not summary:
          print("Skipped: %s has no summary." % filename)
          continue
        if len(text) < len(summary):
          print("Skipped: the text of %s is shorter than its summary." % filename)
          continue
        fout.write(" ".join(text) + "\t" + " ".join(summary) + "\n")
        count += 1
        if count % print_every == 0:
          print(count)
          fout.flush()

train_out.close()
valid_out.close()
test_out.close()
