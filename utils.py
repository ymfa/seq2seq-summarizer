import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import NamedTuple, List, Callable
from collections import Counter
from random import shuffle
import torch

plt.switch_backend('agg')


class Vocab(object):

  PAD = 0
  SOS = 1
  EOS = 2
  UNK = 3

  def __init__(self):
    self.word2index = {}
    self.word2count = Counter()
    self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    self.index2word = self.reserved[:]
    self.embeddings = None

  def add_words(self, words: List[str]):
    for word in words:
      if word not in self.word2index:
        self.word2index[word] = len(self.index2word)
        self.index2word.append(word)
    self.word2count.update(words)

  def trim(self, *, vocab_size: int=None, min_freq: int=1):
    if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.word2index)):
      return
    ordered_words = sorted(((c, w) for (w, c) in self.word2count.items()), reverse=True)
    if vocab_size:
      ordered_words = ordered_words[:vocab_size]
    self.word2index = {}
    self.word2count = Counter()
    self.index2word = self.reserved[:]
    for count, word in ordered_words:
      if count < min_freq: break
      self.word2index[word] = len(self.index2word)
      self.word2count[word] = count
      self.index2word.append(word)

  def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
    num_embeddings = 0
    vocab_size = len(self)
    with open(file_path, 'rb') as f:
      for line in f:
        line = line.split()
        word = line[0].decode('utf-8')
        idx = self.word2index.get(word)
        if idx is not None:
          vec = np.array(line[1:], dtype=dtype)
          if self.embeddings is None:
            n_dims = len(vec)
            self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
            self.embeddings[self.PAD] = np.zeros(n_dims)
          self.embeddings[idx] = vec
          num_embeddings += 1
    return num_embeddings

  def __getitem__(self, item):
    if type(item) is int:
      return self.index2word[item]
    return self.word2index.get(item, self.UNK)

  def __len__(self):
    return len(self.index2word)


class Example(NamedTuple):
  src: List[str]
  tgt: List[str]
  src_len: int
  tgt_len: int


def simple_tokenizer(text: str, lower: bool=False, paragraph_break: str=None) -> List[str]:
  all_tokens = []
  for p in text.split('\n'):
    if len(all_tokens) > 0 and paragraph_break:
      all_tokens.append(paragraph_break)
    tokens = p.split()
    if lower:
      tokens = [t.lower() for t in tokens]
    all_tokens.extend(tokens)
  return all_tokens


class Dataset(object):

  def __init__(self, filename: str, tokenize: Callable=simple_tokenizer, max_src_len: int=None,
               max_tgt_len: int=None):
    print("Reading dataset %s..." % filename, end=' ', flush=True)
    self.filename = filename
    self.pairs = []
    self.src_len = 0
    self.tgt_len = 0
    with open(filename, encoding='utf-8') as f:
      for i, line in enumerate(f):
        pair = line.strip().split('\t')
        if len(pair) != 2:
          print("Line %d of %s is malformed." % (i, filename))
          continue
        src = tokenize(pair[0])
        if max_src_len and len(src) > max_src_len: continue
        tgt = tokenize(pair[1])
        if max_tgt_len and len(tgt) > max_tgt_len: continue
        src_len = len(src) + 1  # EOS
        tgt_len = len(tgt) + 1  # EOS
        self.src_len = max(self.src_len, src_len)
        self.tgt_len = max(self.tgt_len, tgt_len)
        self.pairs.append(Example(src, tgt, src_len, tgt_len))
    print("%d pairs." % len(self.pairs))

  def build_vocab(self, vocab_size: int=None, src: bool=True, tgt: bool=True,
                  embed_file: str=None) -> Vocab:
    filename, _ = os.path.splitext(self.filename)
    if vocab_size:
      filename += ".%d" % vocab_size
    filename += '.vocab'
    if os.path.isfile(filename):
      vocab = torch.load(filename)
      if vocab.embeddings is None:
        print("Vocabulary loaded, %d words." % len(vocab))
      else:
        print("Vocabulary with pre-trained embeddings loaded, %d words." % len(vocab))
    else:
      print("Building vocabulary...", end=' ', flush=True)
      vocab = Vocab()
      for example in self.pairs:
        if src:
          vocab.add_words(example.src)
        if tgt:
          vocab.add_words(example.tgt)
      vocab.trim(vocab_size=vocab_size)
      if embed_file:
        count = vocab.load_embeddings(embed_file)
        print("%d words, %d pre-trained embeddings." % (len(vocab), count))
      else:
        print("%d words." % len(vocab))
      torch.save(vocab, filename)
    return vocab

  def generator(self, batch_size: int, src_vocab: Vocab=None, tgt_vocab: Vocab=None):
    ptr = len(self.pairs)  # make sure to shuffle at first run
    while True:
      if ptr + batch_size > len(self.pairs):
        shuffle(self.pairs)  # shuffle inplace to save memory
        ptr = 0
      examples = self.pairs[ptr:ptr + batch_size]
      ptr += batch_size
      if src_vocab or tgt_vocab:
        if src_vocab:
          examples.sort(key=lambda x: -x.src_len)
          lengths = [x.src_len for x in examples]
          max_src_len = lengths[0]
          src_tensor = torch.zeros(max_src_len, batch_size, dtype=torch.long)
        if tgt_vocab:
          max_tgt_len = max(x.tgt_len for x in examples)
          tgt_tensor = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)
        for i, example in enumerate(examples):
          if src_vocab:
            for j, word in enumerate(example.src):
              src_tensor[j, i] = src_vocab[word]
            src_tensor[example.src_len - 1, i] = src_vocab.EOS
          if tgt_vocab:
            for j, word in enumerate(example.tgt):
              tgt_tensor[j, i] = tgt_vocab[word]
            tgt_tensor[example.tgt_len - 1, i] = tgt_vocab.EOS
        if src_vocab and tgt_vocab:
          yield examples, src_tensor, tgt_tensor, lengths
        elif src_vocab:
          yield examples, src_tensor, lengths
        else:
          yield examples, tgt_tensor
      else:
        yield examples


def show_plot(points, step=1, file_prefix=None):
  plt.figure()
  fig, ax = plt.subplots()
  # this locator puts ticks at regular intervals
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(range(step, len(points)*step + 1, step), points)
  if file_prefix:
    plt.savefig(file_prefix + '.png')
