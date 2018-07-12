from __future__ import unicode_literals, print_function, division
from io import open
import re, unicodedata
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SOS = 0
EOS = 1


class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = ['<SOS>', '<EOS>']

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = len(self.index2word)
      self.word2count[word] = 1
      self.index2word.append(word)
    else:
      self.word2count[word] += 1


# Remove accents to obtain plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def deaccent(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
  )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
  s = deaccent(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  return s


def readLangs(lang1, lang2, reverse=False):
  print("Reading lines...")

  # Read the file and split into lines
  lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
      read().strip().split('\n')

  # Split every line into pairs and normalize
  pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

  # Reverse pairs, make Lang instances
  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    lang1, lang2 = lang2, lang1
  input_lang = Lang(lang1)
  output_lang = Lang(lang2)

  return input_lang, output_lang, pairs


MAX_LENGTH = 10


def filterPairs(pairs):
  return pairs[:500]  # for now, use the first 500 examples


def prepareData(lang1, lang2, reverse=False):
  input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
  print("Read %s sentence pairs" % len(pairs))
  pairs = filterPairs(pairs)
  print("Trimmed to %s sentence pairs" % len(pairs))
  print("Counting words...")
  for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])
  print("Counted words:")
  print(input_lang.name, len(input_lang.index2word))
  print(output_lang.name, len(output_lang.index2word))
  return input_lang, output_lang, pairs


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (ETA %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  # this locator puts ticks at regular intervals
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)
