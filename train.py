import torch
import torch.nn as nn
import math
import os
from torch import optim
from tqdm import tqdm
from utils import Dataset, show_plot, Vocab
from model import Seq2Seq, DEVICE
from params import Params
from test import eval_batch, eval_batch_output


def train_batch(batch, model, criterion, optimizer, *, pack_seq=True, forcing_ratio=0.5,
                partial_forcing=True, rl_ratio: float=0, vocab=None):
  examples, input_tensor, target_tensor, input_lengths, oov_dict = batch
  if not pack_seq:
    input_lengths = None

  optimizer.zero_grad()
  input_tensor = input_tensor.to(DEVICE)
  target_tensor = target_tensor.to(DEVICE)
  ext_vocab_size = oov_dict['size'] if oov_dict is not None else None

  out = model(input_tensor, target_tensor, input_lengths, criterion,
              forcing_ratio=forcing_ratio, partial_forcing=partial_forcing,
              ext_vocab_size=ext_vocab_size)

  if rl_ratio > 0:
    assert vocab is not None
    sample_out = model(input_tensor, None, input_lengths, criterion, sample=True,
                       ext_vocab_size=ext_vocab_size)
    baseline_out = model(input_tensor, None, input_lengths, visualize=False,
                         ext_vocab_size=ext_vocab_size)
    scores = eval_batch_output([ex.tgt for ex in examples], vocab, oov_dict,
                               sample_out.decoded_tokens, baseline_out.decoded_tokens)
    reward = scores[0]['l_f'] - scores[1]['l_f']  # sample - baseline
    rl_loss = reward * sample_out.loss
    loss = (1 - rl_ratio) * out.loss + rl_ratio * rl_loss
  else:
    loss = out.loss

  loss.backward()
  optimizer.step()

  target_length = target_tensor.size(0)
  return loss.item() / target_length


def train(train_generator, vocab: Vocab, model: Seq2Seq, params: Params, valid_generator=None):
  # variables for plotting
  plot_points_per_epoch = max(math.log(params.n_batches, 1.6), 1.)
  plot_every = round(params.n_batches / plot_points_per_epoch)
  plot_losses, cached_losses = [], []
  total_batch_count = 0
  plot_val_losses, plot_val_metrics = [], []

  total_parameters = sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)
  print("Training %d trainable parameters..." % total_parameters)
  model.to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=params.lr)
  criterion = nn.NLLLoss(ignore_index=vocab.PAD)
  best_avg_loss, best_epoch_id = float("inf"), None

  for epoch_count in range(1, params.n_epochs + 1):
    epoch_loss = 0
    epoch_avg_loss, valid_avg_loss, valid_avg_metric = None, None, None
    prog_bar = tqdm(range(1, params.n_batches + 1), desc='Epoch %d' % epoch_count)
    model.train()

    for batch_count in prog_bar:  # training batches
      batch = next(train_generator)
      loss = train_batch(batch, model, criterion, optimizer, pack_seq=params.pack_seq,
                         forcing_ratio=params.forcing_ratio,
                         partial_forcing=params.partial_forcing,
                         rl_ratio=params.rl_ratio, vocab=vocab)

      epoch_loss += float(loss)
      epoch_avg_loss = epoch_loss / batch_count
      prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

      cached_losses.append(loss)
      if (total_batch_count + batch_count) % plot_every == 0:
        period_avg_loss = sum(cached_losses) / len(cached_losses)
        plot_losses.append(period_avg_loss)
        cached_losses = []

    if valid_generator is not None:  # validation batches
      valid_loss, valid_metric = 0, 0
      prog_bar = tqdm(range(1, params.n_val_batches + 1), desc='Valid %d' % epoch_count)
      model.eval()

      for batch_count in prog_bar:
        batch = next(valid_generator)
        metric, loss = eval_batch(batch, model, vocab, criterion, pack_seq=params.pack_seq)
        valid_loss += loss
        valid_metric += metric
        valid_avg_loss = valid_loss / batch_count
        valid_avg_metric = valid_metric / batch_count
        prog_bar.set_postfix(loss='%g' % valid_avg_loss, rouge='%.4g' % (valid_avg_metric * 100))

      plot_val_losses.append(valid_avg_loss)
      plot_val_metrics.append(valid_avg_metric)

      metric_loss = -valid_avg_metric  # choose the best model by ROUGE instead of loss
      if metric_loss < best_avg_loss:
        best_epoch_id = epoch_count
        best_avg_loss = metric_loss

    else:  # no validation, "best" is defined by training loss
      if epoch_avg_loss < best_avg_loss:
        best_epoch_id = epoch_count
        best_avg_loss = epoch_avg_loss

    if params.model_path_prefix:
      # save model
      filename = '%s.%02d.pt' % (params.model_path_prefix, epoch_count)
      torch.save(model, filename)
      if not params.keep_every_epoch:  # clear previously saved models
        for epoch_id in range(1, epoch_count):
          if epoch_id != best_epoch_id:
            try:
              prev_filename = '%s.%02d.pt' % (params.model_path_prefix, epoch_id)
              os.remove(prev_filename)
            except FileNotFoundError:
              pass
      # save training status
      torch.save({
        'epoch': epoch_count,
        'train_avg_loss': epoch_avg_loss,
        'valid_avg_loss': valid_avg_loss,
        'valid_avg_metric': valid_avg_metric,
        'best_epoch_so_far': best_epoch_id,
        'params': params,
        'optimizer': optimizer
      }, '%s.train.pt' % params.model_path_prefix)

    total_batch_count += params.n_batches
    show_plot(plot_losses, plot_every, plot_val_losses, plot_val_metrics, params.n_batches,
              params.model_path_prefix)


if __name__ == "__main__":
  p = Params()

  dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len)
  v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
  m = Seq2Seq(v, p)

  train_gen = dataset.generator(p.batch_size, v, v, True if p.pointer else False)
  if p.val_data_path:
    val_dataset = Dataset(p.val_data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len)
    val_gen = val_dataset.generator(p.val_batch_size, v, v, True if p.pointer else False)
  else:
    val_gen = None

  train(train_gen, v, m, p, val_gen)
