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


class AttnDecoderRNN(nn.Module):

  def __init__(self, vocab_size, hidden_size, max_length, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, vocab_size)

  def forward(self, embedded, hidden, encoder_states):
    embedded = self.dropout(embedded)
    # attn_weights shape (batch size, sentence len)
    attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
    attn_applied = (attn_weights * encoder_states.transpose(0, 2)).transpose(0, 2)

    output = torch.cat((embedded, attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights


class Seq2Seq(nn.Module):

  def __init__(self, vocab, embed_size, hidden_size, max_input_length, max_output_length):
    super(Seq2Seq, self).__init__()
    self.SOS = vocab.SOS
    self.vocab_size = len(vocab)
    self.embed_size = embed_size
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length

    self.embedding = nn.Embedding(self.vocab_size, embed_size)
    self.encoder = EncoderRNN(embed_size, hidden_size)
    self.decoder = AttnDecoderRNN(self.vocab_size, hidden_size, max_input_length)

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
      decoder_attentions = torch.zeros(target_length, batch_size, self.max_input_length)

    decoder_input = torch.tensor([self.SOS] * batch_size, device=DEVICE)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
      decoder_embedded = self.embedding(decoder_input)
      decoder_output, decoder_hidden, decoder_attention = \
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
          decoder_attentions[di] = decoder_attention.data
        decoder_input = topi.detach()  # detach from history as input
        #if decoder_input.item() == EOS: break
    
    if criterion:
      return loss
    else:
      return decoded_tokens, decoder_attentions[:di + 1]
