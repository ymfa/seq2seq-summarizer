import torch
import torch.nn as nn
import math
import os
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import Dataset, show_plot, Vocab, Batch
from model import Seq2Seq, DEVICE
from params import Params
from test import eval_batch, eval_batch_output


def train_batch(batch: Batch, model: Seq2Seq, criterion, optimizer, *,
                pack_seq=True, forcing_ratio=0.5, partial_forcing=True, sample=False,
                rl_ratio: float=0, vocab=None, grad_norm: float=0, show_cover_loss=False):
  if not pack_seq:
    input_lengths = None
  else:
    input_lengths = batch.input_lengths

  optimizer.zero_grad()
  input_tensor = batch.input_tensor.to(DEVICE)
  target_tensor = batch.target_tensor.to(DEVICE)
  ext_vocab_size = batch.ext_vocab_size

  out = model(input_tensor, target_tensor, input_lengths, criterion,
              forcing_ratio=forcing_ratio, partial_forcing=partial_forcing, sample=sample,
              ext_vocab_size=ext_vocab_size, include_cover_loss=show_cover_loss)

  if rl_ratio > 0:
    assert vocab is not None
    sample_out = model(input_tensor, saved_out=out, criterion=criterion, sample=True,
                       ext_vocab_size=ext_vocab_size)
    baseline_out = model(input_tensor, saved_out=out, visualize=False,
                         ext_vocab_size=ext_vocab_size)
    scores = eval_batch_output([ex.tgt for ex in batch.examples], vocab, batch.oov_dict,
                               sample_out.decoded_tokens, baseline_out.decoded_tokens)
    greedy_rouge = scores[1]['l_f']
    neg_reward = greedy_rouge - scores[0]['l_f']
    # if sample > baseline, the reward is positive (i.e. good exploration), rl_loss is negative
    rl_loss = neg_reward * sample_out.loss
    rl_loss_value = neg_reward * sample_out.loss_value
    loss = (1 - rl_ratio) * out.loss + rl_ratio * rl_loss
    loss_value = (1 - rl_ratio) * out.loss_value + rl_ratio * rl_loss_value
  else:
    loss = out.loss
    loss_value = out.loss_value
    greedy_rouge = None

  loss.backward()
  if grad_norm > 0:
    clip_grad_norm_(model.parameters(), grad_norm)
  optimizer.step()

  target_length = target_tensor.size(0)
  return loss_value / target_length, greedy_rouge


def train(train_generator, vocab: Vocab, model: Seq2Seq, params: Params, valid_generator=None,
          saved_state: dict=None):
  # variables for plotting
  plot_points_per_epoch = max(math.log(params.n_batches, 1.6), 1.)
  plot_every = round(params.n_batches / plot_points_per_epoch)
  plot_losses, cached_losses = [], []
  plot_val_losses, plot_val_metrics = [], []

  total_parameters = sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)
  print("Training %d trainable parameters..." % total_parameters)
  model.to(DEVICE)
  if saved_state is None:
    if params.optimizer == 'adagrad':
      optimizer = optim.Adagrad(model.parameters(), lr=params.lr,
                                initial_accumulator_value=params.adagrad_accumulator)
    else:
      optimizer = optim.Adam(model.parameters(), lr=params.lr)
    past_epochs = 0
    total_batch_count = 0
  else:
    optimizer = saved_state['optimizer']
    past_epochs = saved_state['epoch']
    total_batch_count = saved_state['total_batch_count']
  if params.lr_decay:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, params.lr_decay_step, params.lr_decay,
                                             past_epochs - 1)
  criterion = nn.NLLLoss(ignore_index=vocab.PAD)
  best_avg_loss, best_epoch_id = float("inf"), None

  for epoch_count in range(1 + past_epochs, params.n_epochs + 1):
    if params.lr_decay:
      lr_scheduler.step()
    rl_ratio = params.rl_ratio if epoch_count >= params.rl_start_epoch else 0
    epoch_loss, epoch_metric = 0, 0
    epoch_avg_loss, valid_avg_loss, valid_avg_metric = None, None, None
    prog_bar = tqdm(range(1, params.n_batches + 1), desc='Epoch %d' % epoch_count)
    model.train()

    for batch_count in prog_bar:  # training batches
      if params.forcing_decay_type:
        if params.forcing_decay_type == 'linear':
          forcing_ratio = max(0, params.forcing_ratio - params.forcing_decay * total_batch_count)
        elif params.forcing_decay_type == 'exp':
          forcing_ratio = params.forcing_ratio * (params.forcing_decay ** total_batch_count)
        elif params.forcing_decay_type == 'sigmoid':
          forcing_ratio = params.forcing_ratio * params.forcing_decay / (
                  params.forcing_decay + math.exp(total_batch_count / params.forcing_decay))
        else:
          raise ValueError('Unrecognized forcing_decay_type: ' + params.forcing_decay_type)
      else:
        forcing_ratio = params.forcing_ratio

      batch = next(train_generator)
      loss, metric = train_batch(batch, model, criterion, optimizer, pack_seq=params.pack_seq,
                                 forcing_ratio=forcing_ratio,
                                 partial_forcing=params.partial_forcing, sample=params.sample,
                                 rl_ratio=rl_ratio, vocab=vocab, grad_norm=params.grad_norm,
                                 show_cover_loss=params.show_cover_loss)

      epoch_loss += float(loss)
      epoch_avg_loss = epoch_loss / batch_count
      if metric is not None:  # print ROUGE as well if reinforcement learning is enabled
        epoch_metric += metric
        epoch_avg_metric = epoch_metric / batch_count
        prog_bar.set_postfix(loss='%g' % epoch_avg_loss, rouge='%.4g' % (epoch_avg_metric * 100))
      else:
        prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

      cached_losses.append(loss)
      total_batch_count += 1
      if total_batch_count % plot_every == 0:
        period_avg_loss = sum(cached_losses) / len(cached_losses)
        plot_losses.append(period_avg_loss)
        cached_losses = []

    if valid_generator is not None:  # validation batches
      valid_loss, valid_metric = 0, 0
      prog_bar = tqdm(range(1, params.n_val_batches + 1), desc='Valid %d' % epoch_count)
      model.eval()

      for batch_count in prog_bar:
        batch = next(valid_generator)
        loss, metric = eval_batch(batch, model, vocab, criterion, pack_seq=params.pack_seq,
                                  show_cover_loss=params.show_cover_loss)
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
        for epoch_id in range(1 + past_epochs, epoch_count):
          if epoch_id != best_epoch_id:
            try:
              prev_filename = '%s.%02d.pt' % (params.model_path_prefix, epoch_id)
              os.remove(prev_filename)
            except FileNotFoundError:
              pass
      # save training status
      torch.save({
        'epoch': epoch_count,
        'total_batch_count': total_batch_count,
        'train_avg_loss': epoch_avg_loss,
        'valid_avg_loss': valid_avg_loss,
        'valid_avg_metric': valid_avg_metric,
        'best_epoch_so_far': best_epoch_id,
        'params': params,
        'optimizer': optimizer
      }, '%s.train.pt' % params.model_path_prefix)

    if rl_ratio > 0:
      params.rl_ratio **= params.rl_ratio_power

    show_plot(plot_losses, plot_every, plot_val_losses, plot_val_metrics, params.n_batches,
              params.model_path_prefix)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Train the seq2seq abstractive summarizer.')
  parser.add_argument('--resume_from', type=str, metavar='R',
                      help='path to a saved training status (*.train.pt)')
  args, unknown_args = parser.parse_known_args()

  if args.resume_from:
    print("Resuming from %s..." % args.resume_from)
    train_status = torch.load(args.resume_from)
    m = torch.load('%s.%02d.pt' % (args.resume_from[:-9], train_status['epoch']))
    p = train_status['params']
  else:
    p = Params()
    m = None
    train_status = None

  if unknown_args:  # allow command line args to override params.py
    p.update(unknown_args)

  dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                    truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
  if m is None:
    v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
    m = Seq2Seq(v, p)
  else:
    v = dataset.build_vocab(p.vocab_size)

  train_gen = dataset.generator(p.batch_size, v, v, True if p.pointer else False)
  if p.val_data_path:
    val_dataset = Dataset(p.val_data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                          truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
    val_gen = val_dataset.generator(p.val_batch_size, v, v, True if p.pointer else False)
  else:
    val_gen = None

  train(train_gen, v, m, p, val_gen, train_status)
