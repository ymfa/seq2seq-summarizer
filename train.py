import torch
import torch.nn as nn
import math
from torch import optim
from tqdm import tqdm
from utils import Dataset, show_plot, Vocab
from model import Seq2Seq, DEVICE
from params import Params


def train_batch(batch, model, optimizer, criterion, *, pack_seq=True, forcing_ratio=0.5,
                partial_forcing=True):
  _, input_tensor, target_tensor, input_lengths = batch
  if not pack_seq:
    input_lengths = None

  optimizer.zero_grad()
  loss = model(input_tensor.to(DEVICE), target_tensor.to(DEVICE), input_lengths, criterion,
               forcing_ratio=forcing_ratio, partial_forcing=partial_forcing)
  loss.backward()
  optimizer.step()

  target_length = target_tensor.size(0)
  return loss.item() / target_length


def train(generator, vocab: Vocab, model: Seq2Seq, params: Params):
  plot_points_per_epoch = max(math.log(params.n_batches, 1.6), 1.)
  plot_every = round(params.n_batches / plot_points_per_epoch)
  plot_losses, cached_losses = [], []
  total_batch_count = 0

  total_parameters = sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)
  print("Training %d trainable parameters..." % total_parameters)
  model.to(DEVICE)
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=params.lr)
  criterion = nn.NLLLoss(ignore_index=vocab.PAD)

  for epoch_count in range(1, params.n_epochs + 1):
    epoch_loss = 0
    prog_bar = tqdm(range(1, params.n_batches + 1), desc='Epoch %d' % epoch_count)

    for batch_count in prog_bar:
      batch = next(generator)
      loss = train_batch(batch, model, optimizer, criterion, pack_seq=params.pack_seq,
                         forcing_ratio=params.forcing_ratio,
                         partial_forcing=params.partial_forcing)

      epoch_loss += float(loss)
      epoch_avg_loss = epoch_loss / batch_count
      prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

      cached_losses.append(loss)
      if (total_batch_count + batch_count) % plot_every == 0:
        period_avg_loss = sum(cached_losses) / len(cached_losses)
        plot_losses.append(period_avg_loss)
        cached_losses = []

    if params.model_path_prefix:
      filename = '%s.%02d.pt' % (params.model_path_prefix, epoch_count)
      torch.save(model, filename)
      torch.save({
        'epoch': epoch_count,
        'epoch_avg_loss': epoch_avg_loss,
        'params': params,
        'optimizer': optimizer
      }, '%s.train.pt' % params.model_path_prefix)

    total_batch_count += params.n_batches

  show_plot(plot_losses, plot_every, params.model_path_prefix)


if __name__ == "__main__":
  p = Params()

  dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len)
  v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
  m = Seq2Seq(v, p)

  training_data = dataset.generator(p.batch_size, v, v)
  train(training_data, v, m, p)
