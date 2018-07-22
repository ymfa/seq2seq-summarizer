import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from params import Params
from utils import Vocab
from typing import Union

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
               pointer=True, tied_embedding=None, out_embed_size=None,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0):
    super(DecoderRNN, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.combined_size = self.hidden_size
    self.enc_attn = enc_attn
    self.pointer = pointer
    self.out_embed_size = out_embed_size
    if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
      print("Warning: Output embedding size %d is overriden by its tied embedding size %d."
            % (self.out_embed_size, embed_size))
      self.out_embed_size = embed_size

    self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
    self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)

    if enc_attn:
      self.enc_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
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

  def forward(self, embedded, hidden, encoder_states=None, *, encoder_word_idx=None,
              ext_vocab_size: int=None, log_prob: bool=True):
    """
    :param embedded: (batch size, embed size)
    :param hidden: (1, batch size, decoder hidden size)
    :param encoder_states: (src seq len, batch size, decoder hidden size), for attention mechanism
    :param encoder_word_idx: (src seq len, batch size), for pointer network
    :param ext_vocab_size: the dynamic vocab size, determined by the max num of OOV words contained
                           in any src seq in this batch, for pointer network
    :param log_prob: return log probability instead of probability
    :return: tuple of four things:
             1. word prob or log word prob, (batch size, dynamic vocab size);
             2. RNN hidden state after this step, (1, batch size, decoder hidden size);
             3. attention weights over encoder states, (batch size, src seq len, 1);
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
      enc_energy = self.enc_bilinear(hidden.expand_as(encoder_states).contiguous(), encoder_states)
      # transpose => (batch size, num encoder states, 1)
      enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
      if self.enc_attn:
        # context: (batch size, encoder hidden size, 1)
        enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn)
        combined[:, offset:offset+self.hidden_size] = enc_context.squeeze(2)
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
      ptr_output = enc_attn.squeeze(2)
      output.scatter_add_(1, encoder_word_idx.transpose(0, 1), prob_ptr * ptr_output)
      if log_prob: output = torch.log(output)  # necessary for NLLLoss
    else:
      if log_prob: output = F.log_softmax(logits, dim=1)
      else: output = F.softmax(logits, dim=1)

    return output, hidden, enc_attn, prob_ptr


class Seq2SeqOutput(object):

  def __init__(self, decoded_tokens: torch.Tensor, loss: Union[torch.Tensor, float]=0,
               enc_attn_weights: torch.Tensor=None, ptr_probs: torch.Tensor=None):
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)


class Seq2Seq(nn.Module):

  def __init__(self, vocab: Vocab, params: Params, max_output_length=None):
    """
    :param vocab: only for accessing vocab info (special tokens and vocab size)
    :param params: model hyper-parameters
    :param max_output_length: max num of decoder steps if not hitting SOS (only effective at test
                              time, as during training the num of steps is determined by the
                              `target_tensor`); it is safe to change `self.max_output_length` as
                              it network architecture is independent of src/tgt seq lengths
    Create the seq2seq model; its encoder and decoder will be created automatically.
    """
    super(Seq2Seq, self).__init__()
    self.SOS = vocab.SOS
    self.UNK = vocab.UNK
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
    self.max_output_length = \
      params.max_tgt_len + 1 if max_output_length is None else max_output_length
    self.enc_attn = params.enc_attn
    self.dec_attn = params.dec_attn
    self.pointer = params.pointer

    self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                  _weight=embedding_weights)
    self.encoder = EncoderRNN(self.embed_size, params.hidden_size, params.enc_bidi,
                              rnn_drop=params.enc_rnn_dropout)
    self.decoder = DecoderRNN(self.vocab_size, self.embed_size,
                              params.hidden_size * 2 if params.enc_bidi else params.hidden_size,
                              enc_attn=params.enc_attn, dec_attn=params.dec_attn,
                              pointer=params.pointer, out_embed_size=params.out_embed_size,
                              tied_embedding=self.embedding if params.tie_embed else None,
                              in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout,
                              out_drop=params.dec_out_dropout)

  def filter_oov(self, tensor, ext_vocab_size):
    """Replace any OOV index in `tensor` with UNK"""
    if ext_vocab_size and ext_vocab_size > self.vocab_size:
      result = tensor.clone()
      result[tensor >= self.vocab_size] = self.UNK
      return result
    return tensor

  def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *,
              forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False,
              visualize: bool=None) -> Seq2SeqOutput:
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
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    Run the seq2seq model for training or testing.
    """
    input_length = input_tensor.size(0)
    batch_size = input_tensor.size(1)
    log_prob = type(criterion) is nn.NLLLoss
    if visualize is None:
      visualize = criterion is None
    if visualize and not (self.enc_attn or self.pointer):
      visualize = False  # nothing to visualize

    encoder_hidden = self.encoder.init_hidden(batch_size)
    # encoder_embedded: (input len, batch size, embed size)
    encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))

    encoder_outputs, encoder_hidden = \
      self.encoder(encoder_embedded, encoder_hidden, input_lengths)

    if target_tensor is None:
      target_length = self.max_output_length
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

    r = Seq2SeqOutput(torch.zeros(target_length, batch_size)) # initialize return values
    if visualize:
      r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
      if self.pointer:
        r.ptr_probs = torch.zeros(target_length, batch_size)

    decoder_input = torch.tensor([self.SOS] * batch_size, device=DEVICE)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
      decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
      decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob)
      # save the decoded tokens
      if not sample:
        _, top_idx = decoder_output.data.topk(1)  # top_idx shape: (batch size, k=1)
      else:
        prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
        top_idx = torch.multinomial(prob_distribution, 1)
      top_idx = top_idx.squeeze(1)
      r.decoded_tokens[di] = top_idx
      # compute additional outputs
      if criterion:
        r.loss += criterion(decoder_output, target_tensor[di])
      if visualize:
        r.enc_attn_weights[di] = dec_enc_attn.squeeze(2).data
        if self.pointer:
          r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data
      # decide the next input
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        decoder_input = top_idx.detach()  # detach from history as input
    
    return r
