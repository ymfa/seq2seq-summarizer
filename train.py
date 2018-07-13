import torch
import torch.nn as nn
from torch import optim
import time, random
from utils import EOS, SOS, timeSince, showPlot, prepareData
from model import DEVICE, Seq2Seq


def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(EOS)
  # "view" reshapes the vector to a matrix (any number of rows (-1) * one column (1))
  return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
  input_tensor = tensorFromSentence(input_lang, pair[0])
  target_tensor = tensorFromSentence(output_lang, pair[1])
  return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, model, optimizer, criterion):
  optimizer.zero_grad()

  loss = model(input_tensor, target_tensor, criterion)
  loss.backward()

  optimizer.step()

  target_length = target_tensor.size(0)
  return loss.item() / target_length


def trainIters(input_lang, output_lang, pairs, model, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
  start = time.time()
  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, model, optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  showPlot(plot_losses)


if __name__ == "__main__":
  orig, summ, pairs = prepareData('org', 'sht')

  hidden_size = 100
  model = Seq2Seq(len(orig.index2word), hidden_size, hidden_size, orig.max_length + 1, summ.max_length + 1)

  trainIters(orig, summ, pairs, model, 5000, print_every=100)
  torch.save(model.state_dict(), 'checkpoints/newsumm.pt')
