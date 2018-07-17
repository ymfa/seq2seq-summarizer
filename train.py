import torch
import torch.nn as nn
import math
from torch import optim
from tqdm import tqdm
from utils import Dataset, show_plot
from model import Seq2Seq, DEVICE


def train_batch(batch, model, optimizer, criterion, pack_seq=True):
  _, input_tensor, target_tensor, input_lengths = batch
  if not pack_seq:
    input_lengths = None

  optimizer.zero_grad()
  loss = model(input_tensor.to(DEVICE), target_tensor.to(DEVICE), input_lengths, criterion)
  loss.backward()
  optimizer.step()

  target_length = target_tensor.size(0)
  return loss.item() / target_length


def train(generator, vocab, model, n_batches=100, n_epochs=5, *, lr=0.001, pack_seq=True,
          auto_save_prefix=None):
  plot_points_per_epoch = max(math.log(n_batches, 1.6), 1.)
  plot_every = round(n_batches / plot_points_per_epoch)
  plot_losses, cached_losses = [], []
  total_batch_count = 0

  model.to(DEVICE)
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.NLLLoss(ignore_index=vocab.PAD)

  for epoch_count in range(1, n_epochs + 1):
    epoch_loss = 0
    prog_bar = tqdm(range(1, n_batches + 1), desc='Epoch %d' % epoch_count)

    for batch_count in prog_bar:
      batch = next(generator)
      loss = train_batch(batch, model, optimizer, criterion, pack_seq)

      epoch_loss += float(loss)
      epoch_avg_loss = epoch_loss / batch_count
      prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

      cached_losses.append(loss)
      if (total_batch_count + batch_count) % plot_every == 0:
        period_avg_loss = sum(cached_losses) / len(cached_losses)
        plot_losses.append(period_avg_loss)
        cached_losses = []

    if auto_save_prefix:
      filename = '%s.%02d.pt' % (auto_save_prefix, epoch_count)
      torch.save(model.state_dict(), filename)

    total_batch_count += n_batches

  show_plot(plot_losses, plot_every, auto_save_prefix)


if __name__ == "__main__":
  from params import *

  dataset = Dataset(data_path, max_src_len=80, max_tgt_len=25)
  vocabulary = dataset.build_vocab(vocab_size, embed_file=embed_file)
  training_data = dataset.generator(batch_size, vocabulary, vocabulary)
  m = Seq2Seq(vocabulary, embed_size, hidden_size, dataset.src_len, dataset.tgt_len,
              enc_bidi=encoder_bidi, enc_attn=encoder_attn)

  train(training_data, vocabulary, m, n_batches=num_batches, n_epochs=num_epochs, lr=learning_rate,
        pack_seq=use_packed_seq, auto_save_prefix=model_path_prefix)
