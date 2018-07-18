import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from params import Params
from utils import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_directions = 2 if bidi else 1
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

  def forward(self, embedded, hidden, input_lengths=None):
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
               tied_embedding=None, out_embed_size=None,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0):
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.combined_size = self.hidden_size
    self.enc_attn = enc_attn
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

  def forward(self, embedded, hidden, encoder_states=None):
    batch_size = embedded.size(0)
    combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)

    if self.in_drop: embedded = self.in_drop(embedded)
    output, hidden = self.gru(embedded.unsqueeze(0), hidden)  # unsqueeze and squeeze are necessary
    combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)
    offset = self.hidden_size
    enc_attn = None

    if self.enc_attn:
      # energy: (num encoder states, batch size, 1)
      enc_energy = self.enc_bilinear(hidden.expand_as(encoder_states).contiguous(), encoder_states)
      # attention: same as energy
      enc_attn = F.softmax(enc_energy, dim=0)
      # context: (batch size, encoder hidden size, 1)
      enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn.transpose(0, 1))
      # save context vector
      combined[:, offset:offset+self.hidden_size] = enc_context.squeeze(2)
      offset += self.hidden_size

    if self.out_drop: combined = self.out_drop(combined)
    if self.out_embed_size: combined = self.pre_out(combined)

    output = self.out(combined)  # (batch size, vocab size)
    output = F.log_softmax(output, dim=1)
    return output, hidden, enc_attn


class Seq2Seq(nn.Module):

  def __init__(self, vocab: Vocab, params: Params):
    super(Seq2Seq, self).__init__()
    self.SOS = vocab.SOS
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
    self.max_input_length = params.max_src_len
    self.max_output_length = params.max_tgt_len
    self.enc_attn = params.enc_attn
    self.dec_attn = params.dec_attn

    self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                  _weight=embedding_weights)
    self.encoder = EncoderRNN(self.embed_size, params.hidden_size, params.enc_bidi,
                              rnn_drop=params.enc_rnn_dropout)
    self.decoder = DecoderRNN(self.vocab_size, self.embed_size,
                              params.hidden_size * 2 if params.enc_bidi else params.hidden_size,
                              enc_attn=params.enc_attn, dec_attn=params.dec_attn,
                              out_embed_size=params.out_embed_size,
                              tied_embedding=self.embedding if params.tie_embed else None,
                              in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout,
                              out_drop=params.dec_out_dropout)

  def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *,
              forcing_ratio=0.5, partial_forcing=True):
    input_length = input_tensor.size(0)
    batch_size = input_tensor.size(1)

    encoder_hidden = self.encoder.init_hidden(batch_size)
    encoder_embedded = self.embedding(input_tensor)  # (input len, batch size, embed size)

    encoder_outputs, encoder_hidden = \
      self.encoder(encoder_embedded, encoder_hidden, input_lengths)

    if target_tensor is None:
      target_length = self.max_output_length
    else:
      target_length = target_tensor.size(0)

    if criterion:  # training: compute loss
      loss = 0
      if partial_forcing:
        use_teacher_forcing = None  # decide later individually in each step
      else:
        use_teacher_forcing = random.random() < forcing_ratio
    else:  # testing: decode tokens
      decoded_tokens = [[] for _ in range(batch_size)]
      use_teacher_forcing = False
      if self.enc_attn:
        enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
      else:
        enc_attn_weights = None

    decoder_input = torch.tensor([self.SOS] * batch_size, device=DEVICE)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
      decoder_embedded = self.embedding(decoder_input)
      decoder_output, decoder_hidden, dec_enc_attn = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs)
      if criterion:
        loss += criterion(decoder_output, target_tensor[di])
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        _, topi = decoder_output.data.topk(1)  # topi shape: (batch size, k=1)
        topi = topi.squeeze(1)
        if not criterion:
          for bi in range(batch_size):
            decoded_tokens[bi].append(topi[bi].item())
          if self.enc_attn:
            enc_attn_weights[di] = dec_enc_attn.squeeze(2).transpose(0, 1).data
        decoder_input = topi.detach()  # detach from history as input
        #if decoder_input.item() == EOS: break
    
    if criterion:
      return loss
    else:
      #if enc_attn_weights is not None:
      #  enc_attn_weights = enc_attn_weights[:di + 1]
      return decoded_tokens, enc_attn_weights
