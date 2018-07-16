import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.gru = nn.GRU(embed_size, hidden_size)

  def forward(self, embedded, hidden, input_lengths=None):
    if input_lengths is not None:
      embedded_packed = pack_padded_sequence(embedded, input_lengths)
      output_packed, hidden = self.gru(embedded_packed, hidden)
      output, _ = pad_packed_sequence(output_packed)
    else:
      output, hidden = self.gru(embedded, hidden)
    return output, hidden

  def initHidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):

  def __init__(self, vocab_size, hidden_size, max_length, enc_attn: bool=True, dec_attn: bool=True,
               dropout_p=0.1):
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.max_length = max_length
    self.size_before_output = self.hidden_size
    self.enc_attn = enc_attn

    self.dropout = nn.Dropout(dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    if enc_attn:
      self.enc_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
      self.size_before_output += self.hidden_size

    self.out = nn.Linear(self.size_before_output, vocab_size)

  def forward(self, embedded, hidden, encoder_states=None):
    batch_size = embedded.size(0)
    combined = torch.zeros(batch_size, self.size_before_output, device=DEVICE)

    embedded = self.dropout(embedded).unsqueeze(0)  # n_steps is always 1
    output, hidden = self.gru(embedded, hidden)
    combined[:, :self.hidden_size] = output.squeeze(0)
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

    # output: (batch size, vocab size)
    output = self.out(combined)
    output = F.log_softmax(output, dim=1)
    return output, hidden, enc_attn


class Seq2Seq(nn.Module):

  def __init__(self, vocab, embed_size, hidden_size, max_input_length, max_output_length,
               enc_attn=True, dec_attn=True):
    super(Seq2Seq, self).__init__()
    self.SOS = vocab.SOS
    self.vocab_size = len(vocab)
    self.embed_size = embed_size
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length
    self.enc_attn = enc_attn
    self.dec_attn = dec_attn

    self.embedding = nn.Embedding(self.vocab_size, embed_size)
    self.encoder = EncoderRNN(embed_size, hidden_size)
    self.decoder = DecoderRNN(self.vocab_size, hidden_size, max_input_length, enc_attn, dec_attn)

  def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None,
              teacher_forcing_ratio=0.5):
    input_length = input_tensor.size(0)
    batch_size = input_tensor.size(1)

    encoder_hidden = self.encoder.initHidden(batch_size)
    encoder_outputs = torch.zeros(self.max_input_length, batch_size, self.encoder.hidden_size,
                                  device=DEVICE)
    encoder_embedded = self.embedding(input_tensor)  # (input len, batch size, embed size)

    encoder_outputs[:input_length], encoder_hidden = \
      self.encoder(encoder_embedded, encoder_hidden, input_lengths)
    
    if criterion:  # training: compute loss
      loss = 0
      use_teacher_forcing = random.random() < teacher_forcing_ratio
      target_length = target_tensor.size(0)
    else:  # testing: decode tokens
      decoded_tokens = [[] for _ in range(batch_size)]
      use_teacher_forcing = False
      target_length = self.max_output_length
      if self.enc_attn:
        enc_attn_weights = torch.zeros(target_length, batch_size, self.max_input_length)
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
      if use_teacher_forcing:
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
      if enc_attn_weights is not None:
        enc_attn_weights = enc_attn_weights[:di + 1]
      return decoded_tokens, enc_attn_weights
