import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from params import Params
from utils import Vocab, Hypothesis, word_detector
from typing import Union, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-31


class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_directions = 2 if bidi else 1
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

  def forward(self, embedded, hidden, input_lengths=None):
    """
    :param embedded: (src seq len, batch size, embed size)
    :param hidden: (num directions, batch size, encoder hidden size)
    :param input_lengths: list containing the non-padded length of each sequence in this batch;
                          if set, we use `PackedSequence` to skip the PAD inputs and leave the
                          corresponding encoder states as zeros
    :return: (src seq len, batch size, hidden size * num directions = decoder hidden size)

    Perform multi-step encoding.
    """
    if input_lengths is not None:
      embedded = pack_padded_sequence(embedded, input_lengths)

    output, hidden = self.gru(embedded, hidden)

    if input_lengths is not None:
      output, _ = pad_packed_sequence(output)

    if self.num_directions > 1:
      # hidden: (num directions, batch, hidden) => (1, batch, hidden * 2)
      batch_size = hidden.size(1)
      hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                                                        self.hidden_size * self.num_directions)
    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, *, enc_attn=True, dec_attn=True,
               enc_attn_cover=True, pointer=True, tied_embedding=None, out_embed_size=None,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None):
    super(DecoderRNN, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.combined_size = self.hidden_size
    self.enc_attn = enc_attn
    self.dec_attn = dec_attn
    self.enc_attn_cover = enc_attn_cover
    self.pointer = pointer
    self.out_embed_size = out_embed_size
    if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
      print("Warning: Output embedding size %d is overriden by its tied embedding size %d."
            % (self.out_embed_size, embed_size))
      self.out_embed_size = embed_size

    self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
    self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)

    if enc_attn:
      if not enc_hidden_size: enc_hidden_size = self.hidden_size
      self.enc_bilinear = nn.Bilinear(self.hidden_size, enc_hidden_size, 1)
      self.combined_size += enc_hidden_size
      if enc_attn_cover:
        self.cover_weight = nn.Parameter(torch.rand(1))

    if dec_attn:
      self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
      self.combined_size += self.hidden_size

    self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
    if pointer:
      self.ptr = nn.Linear(self.combined_size, 1)

    if tied_embedding is not None and embed_size != self.combined_size:
      # use pre_out layer if combined size is different from embedding size
      self.out_embed_size = embed_size

    if self.out_embed_size:  # use pre_out layer
      self.pre_out = nn.Linear(self.combined_size, self.out_embed_size)
      size_before_output = self.out_embed_size
    else:  # don't use pre_out layer
      size_before_output = self.combined_size

    self.out = nn.Linear(size_before_output, vocab_size)
    if tied_embedding is not None:
      self.out.weight = tied_embedding.weight

  def forward(self, embedded, hidden, encoder_states=None, decoder_states=None, coverage_vector=None, *,
              encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True):
    """
    :param embedded: (batch size, embed size)
    :param hidden: (1, batch size, decoder hidden size)
    :param encoder_states: (src seq len, batch size, hidden size), for attention mechanism
    :param decoder_states: (past dec steps, batch size, hidden size), for attention mechanism
    :param encoder_word_idx: (src seq len, batch size), for pointer network
    :param ext_vocab_size: the dynamic vocab size, determined by the max num of OOV words contained
                           in any src seq in this batch, for pointer network
    :param log_prob: return log probability instead of probability
    :return: tuple of four things:
             1. word prob or log word prob, (batch size, dynamic vocab size);
             2. RNN hidden state after this step, (1, batch size, decoder hidden size);
             3. attention weights over encoder states, (batch size, src seq len);
             4. prob of copying by pointing as opposed to generating, (batch size, 1)

    Perform single-step decoding.
    """
    batch_size = embedded.size(0)
    combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)

    if self.in_drop: embedded = self.in_drop(embedded)

    output, hidden = self.gru(embedded.unsqueeze(0), hidden)  # unsqueeze and squeeze are necessary
    combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)
    offset = self.hidden_size
    enc_attn, prob_ptr = None, None  # for visualization

    if self.enc_attn or self.pointer:
      # energy and attention: (num encoder states, batch size, 1)
      num_enc_steps = encoder_states.size(0)
      enc_total_size = encoder_states.size(2)
      enc_energy = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(),
                                     encoder_states)
      if self.enc_attn_cover and coverage_vector is not None:
        enc_energy += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
      # transpose => (batch size, num encoder states, 1)
      enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
      if self.enc_attn:
        # context: (batch size, encoder hidden size, 1)
        enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn)
        combined[:, offset:offset+enc_total_size] = enc_context.squeeze(2)
        offset += enc_total_size
      enc_attn = enc_attn.squeeze(2)

    if self.dec_attn:
      if decoder_states is not None and len(decoder_states) > 0:
        dec_energy = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(),
                                       decoder_states)
        dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
        dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
        combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
      offset += self.hidden_size

    if self.out_drop: combined = self.out_drop(combined)

    # generator
    if self.out_embed_size:
      out_embed = self.pre_out(combined)
    else:
      out_embed = combined
    logits = self.out(out_embed)  # (batch size, vocab size)

    # pointer
    if self.pointer:
      output = torch.zeros(batch_size, ext_vocab_size, device=DEVICE)
      # distribute probabilities between generator and pointer
      prob_ptr = F.sigmoid(self.ptr(combined))  # (batch size, 1)
      prob_gen = 1 - prob_ptr
      # add generator probabilities to output
      gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
      output[:, :self.vocab_size] = prob_gen * gen_output
      # add pointer probabilities to output
      ptr_output = enc_attn
      output.scatter_add_(1, encoder_word_idx.transpose(0, 1), prob_ptr * ptr_output)
      if log_prob: output = torch.log(output + eps)
    else:
      if log_prob: output = F.log_softmax(logits, dim=1)
      else: output = F.softmax(logits, dim=1)

    return output, hidden, enc_attn, prob_ptr


class Seq2SeqOutput(object):

  def __init__(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor,
               decoded_tokens: torch.Tensor, loss: Union[torch.Tensor, float]=0,
               loss_value: float=0, enc_attn_weights: torch.Tensor=None,
               ptr_probs: torch.Tensor=None):
    self.encoder_outputs = encoder_outputs
    self.encoder_hidden = encoder_hidden
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.loss_value = loss_value  # float value, excluding coverage loss
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)


class Seq2Seq(nn.Module):

  def __init__(self, vocab: Vocab, params: Params, max_dec_steps=None):
    """
    :param vocab: mainly for info about special tokens and vocab size
    :param params: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the seq2seq model; its encoder and decoder will be created automatically.
    """
    super(Seq2Seq, self).__init__()
    self.vocab = vocab
    self.vocab_size = len(vocab)
    if vocab.embeddings is not None:
      self.embed_size = vocab.embeddings.shape[1]
      if params.embed_size is not None and self.embed_size != params.embed_size:
        print("Warning: Model embedding size %d is overriden by pre-trained embedding size %d."
              % (params.embed_size, self.embed_size))
      embedding_weights = torch.from_numpy(vocab.embeddings)
    else:
      self.embed_size = params.embed_size
      embedding_weights = None
    self.max_dec_steps = params.max_tgt_len + 1 if max_dec_steps is None else max_dec_steps
    self.enc_attn = params.enc_attn
    self.enc_attn_cover = params.enc_attn_cover
    self.dec_attn = params.dec_attn
    self.pointer = params.pointer
    self.cover_loss = params.cover_loss
    self.cover_func = params.cover_func
    enc_total_size = params.hidden_size * 2 if params.enc_bidi else params.hidden_size
    if params.dec_hidden_size:
      dec_hidden_size = params.dec_hidden_size
      self.enc_dec_adapter = nn.Linear(enc_total_size, dec_hidden_size)
    else:
      dec_hidden_size = enc_total_size
      self.enc_dec_adapter = None

    self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                  _weight=embedding_weights)
    self.encoder = EncoderRNN(self.embed_size, params.hidden_size, params.enc_bidi,
                              rnn_drop=params.enc_rnn_dropout)
    self.decoder = DecoderRNN(self.vocab_size, self.embed_size, dec_hidden_size,
                              enc_attn=params.enc_attn, dec_attn=params.dec_attn,
                              pointer=params.pointer, out_embed_size=params.out_embed_size,
                              tied_embedding=self.embedding if params.tie_embed else None,
                              in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout,
                              out_drop=params.dec_out_dropout, enc_hidden_size=enc_total_size)

  def filter_oov(self, tensor, ext_vocab_size):
    """Replace any OOV index in `tensor` with UNK"""
    if ext_vocab_size and ext_vocab_size > self.vocab_size:
      result = tensor.clone()
      result[tensor >= self.vocab_size] = self.vocab.UNK
      return result
    return tensor

  def get_coverage_vector(self, enc_attn_weights):
    """Combine the past attention weights into one vector"""
    if self.cover_func == 'max':
      coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
    elif self.cover_func == 'sum':
      coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
    else:
      raise ValueError('Unrecognized cover_func: ' + self.cover_func)
    return coverage_vector

  def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *,
              forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False,
              saved_out: Seq2SeqOutput=None, visualize: bool=None, include_cover_loss: bool=False)\
          -> Seq2SeqOutput:
    """
    :param input_tensor: tensor of word indices, (src seq len, batch size)
    :param target_tensor: tensor of word indices, (tgt seq len, batch size)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the seq2seq model for training or testing.
    """
    input_length = input_tensor.size(0)
    batch_size = input_tensor.size(1)
    log_prob = not (sample or self.decoder.pointer)  # don't apply log too soon in these cases
    if visualize is None:
      visualize = criterion is None
    if visualize and not (self.enc_attn or self.pointer):
      visualize = False  # nothing to visualize

    if target_tensor is None:
      target_length = self.max_dec_steps
    else:
      target_length = target_tensor.size(0)

    if forcing_ratio == 1:
      # if fully teacher-forced, it may be possible to eliminate the for-loop over decoder steps
      # for generality, this optimization is not investigated
      use_teacher_forcing = True
    elif forcing_ratio > 0:
      if partial_forcing:
        use_teacher_forcing = None  # decide later individually in each step
      else:
        use_teacher_forcing = random.random() < forcing_ratio
    else:
      use_teacher_forcing = False

    if saved_out:  # reuse encoder states of a previous run
      encoder_outputs = saved_out.encoder_outputs
      encoder_hidden = saved_out.encoder_hidden
      assert input_length == encoder_outputs.size(0)
      assert batch_size == encoder_outputs.size(1)
    else:  # run the encoder
      encoder_hidden = self.encoder.init_hidden(batch_size)
      # encoder_embedded: (input len, batch size, embed size)
      encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
      encoder_outputs, encoder_hidden = \
        self.encoder(encoder_embedded, encoder_hidden, input_lengths)

    # initialize return values
    r = Seq2SeqOutput(encoder_outputs, encoder_hidden,
                      torch.zeros(target_length, batch_size, dtype=torch.long))
    if visualize:
      r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
      if self.pointer:
        r.ptr_probs = torch.zeros(target_length, batch_size)

    decoder_input = torch.tensor([self.vocab.SOS] * batch_size, device=DEVICE)
    if self.enc_dec_adapter is None:
      decoder_hidden = encoder_hidden
    else:
      decoder_hidden = self.enc_dec_adapter(encoder_hidden)
    decoder_states = []
    enc_attn_weights = []

    for di in range(target_length):
      decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
      if enc_attn_weights:
        coverage_vector = self.get_coverage_vector(enc_attn_weights)
      else:
        coverage_vector = None
      decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                     torch.cat(decoder_states) if decoder_states else None, coverage_vector,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob)
      if self.dec_attn:
        decoder_states.append(decoder_hidden)
      # save the decoded tokens
      if not sample:
        _, top_idx = decoder_output.data.topk(1)  # top_idx shape: (batch size, k=1)
      else:
        prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
        top_idx = torch.multinomial(prob_distribution, 1)
      top_idx = top_idx.squeeze(1).detach()  # detach from history as input
      r.decoded_tokens[di] = top_idx
      # compute loss
      if criterion:
        if target_tensor is None:
          gold_standard = top_idx  # for sampling
        else:
          gold_standard = target_tensor[di]
        if not log_prob:
          decoder_output = torch.log(decoder_output + eps)  # necessary for NLLLoss
        nll_loss = criterion(decoder_output, gold_standard)
        r.loss += nll_loss
        r.loss_value += nll_loss.item()
      # update attention history and compute coverage loss
      if self.enc_attn_cover or (criterion and self.cover_loss > 0):
        if coverage_vector is not None and criterion and self.cover_loss > 0:
          coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size \
                          * self.cover_loss
          r.loss += coverage_loss
          if include_cover_loss: r.loss_value += coverage_loss.item()
        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
      # save data for visualization
      if visualize:
        r.enc_attn_weights[di] = dec_enc_attn.data
        if self.pointer:
          r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data
      # decide the next input
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        decoder_input = top_idx
    
    return r

  def beam_search(self, input_tensor, input_lengths=None, ext_vocab_size=None, beam_size=4, *,
                  min_out_len=1, max_out_len=None, len_in_words=True) -> List[Hypothesis]:
    """
    :param input_tensor: tensor of word indices, (src seq len, batch size); for now, batch size has
                         to be 1
    :param input_lengths: see explanation in `EncoderRNN`
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param beam_size: the beam size
    :param min_out_len: required minimum output length
    :param max_out_len: required maximum output length (if None, use the model's own value)
    :param len_in_words: if True, count output length in words instead of tokens (i.e. do not count
                         punctuations)
    :return: list of the best decoded sequences, in descending order of probability

    Use beam search to generate summaries.
    """
    batch_size = input_tensor.size(1)
    assert batch_size == 1
    if max_out_len is None:
      max_out_len = self.max_dec_steps - 1  # max_out_len doesn't count EOS

    # encode
    encoder_hidden = self.encoder.init_hidden(batch_size)
    # encoder_embedded: (input len, batch size, embed size)
    encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
    encoder_outputs, encoder_hidden = \
      self.encoder(encoder_embedded, encoder_hidden, input_lengths)
    if self.enc_dec_adapter is None:
      decoder_hidden = encoder_hidden
    else:
      decoder_hidden = self.enc_dec_adapter(encoder_hidden)
    # turn batch size from 1 to beam size (by repeating)
    # if we want dynamic batch size, the following must be created for all possible batch sizes
    encoder_outputs = encoder_outputs.expand(-1, beam_size, -1).contiguous()
    input_tensor = input_tensor.expand(-1, beam_size).contiguous()

    # decode
    hypos = [Hypothesis([self.vocab.SOS], [], decoder_hidden, [], [], 1)]
    results, backup_results = [], []
    step = 0
    while hypos and step < 2 * max_out_len:  # prevent infinitely generating punctuations
      # make batch size equal to beam size (n_hypos <= beam size)
      n_hypos = len(hypos)
      if n_hypos < beam_size:
        hypos.extend(hypos[-1] for _ in range(beam_size - n_hypos))
      # assemble existing hypotheses into a batch
      decoder_input = torch.tensor([h.tokens[-1] for h in hypos], device=DEVICE)
      decoder_hidden = torch.cat([h.dec_hidden for h in hypos], 1)
      if self.dec_attn and step > 0:  # dim 0 is decoding step, dim 1 is beam batch
        decoder_states = torch.cat([torch.cat(h.dec_states, 0) for h in hypos], 1)
      else:
        decoder_states = None
      if self.enc_attn_cover:
        enc_attn_weights = [torch.cat([h.enc_attn_weights[i] for h in hypos], 1)
                            for i in range(step)]
      else:
        enc_attn_weights = []
      if enc_attn_weights:
        coverage_vector = self.get_coverage_vector(enc_attn_weights)  # shape: (beam size, src len)
      else:
        coverage_vector = None
      # run the decoder over the assembled batch
      decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
      decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                     decoder_states, coverage_vector,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size)
      top_v, top_i = decoder_output.data.topk(beam_size)  # shape of both: (beam size, beam size)
      # create new hypotheses
      new_hypos = []
      for in_idx in range(n_hypos):
        for out_idx in range(beam_size):
          new_tok = top_i[in_idx][out_idx].item()
          new_prob = top_v[in_idx][out_idx].item()
          if len_in_words:
            non_word = not self.vocab.is_word(new_tok)
          else:
            non_word = new_tok == self.vocab.EOS  # only SOS & EOS don't count
          new_hypo = hypos[in_idx].create_next(new_tok, new_prob,
                                               decoder_hidden[0][in_idx].unsqueeze(0).unsqueeze(0),
                                               self.dec_attn,
                                               dec_enc_attn[in_idx].unsqueeze(0).unsqueeze(0)
                                               if dec_enc_attn is not None else None, non_word)
          new_hypos.append(new_hypo)
      # process the new hypotheses
      new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)
      hypos = []
      new_complete_results, new_incomplete_results = [], []
      for nh in new_hypos:
        length = len(nh)
        if nh.tokens[-1] == self.vocab.EOS:  # a complete hypothesis
          if len(new_complete_results) < beam_size and min_out_len <= length <= max_out_len:
            new_complete_results.append(nh)
        elif len(hypos) < beam_size and length < max_out_len:  # an incomplete hypothesis
          hypos.append(nh)
        elif length == max_out_len and len(new_incomplete_results) < beam_size:
          new_incomplete_results.append(nh)
      if new_complete_results:
        results.extend(new_complete_results)
      elif new_incomplete_results:
        backup_results.extend(new_incomplete_results)
      step += 1
    if not results:  # if no sequence ends with EOS within desired length, fallback to sequences
      results = backup_results  # that are "truncated" at the end to max_out_len
    return sorted(results, key=lambda h: -h.avg_log_prob)[:beam_size]
