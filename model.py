import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import SOS, EOS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.gru = nn.GRU(embed_size, hidden_size)

  def forward(self, embedded, hidden):
    output, hidden = self.gru(embedded.unsqueeze(0), hidden)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


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

    attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_states.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class Seq2Seq(nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, max_input_length, max_output_length):
    super(Seq2Seq, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length

    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.encoder = EncoderRNN(embed_size, hidden_size)
    self.decoder = AttnDecoderRNN(vocab_size, hidden_size, max_input_length)

  def forward(self, input_tensor, target_tensor=None, criterion=None, teacher_forcing_ratio=0.5):
    input_length = input_tensor.size(0)

    encoder_hidden = self.encoder.initHidden()
    encoder_outputs = torch.zeros(self.max_input_length, self.encoder.hidden_size, device=DEVICE)
    encoder_embedded = self.embedding(input_tensor)  # (input len, batch size, embed size)

    for ei in range(input_length):
      encoder_output, encoder_hidden = self.encoder(encoder_embedded[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]
    
    if criterion:  # training: compute loss
      loss = 0
      use_teacher_forcing = random.random() < teacher_forcing_ratio
      target_length = target_tensor.size(0)
    else:  # testing: decode tokens
      decoded_tokens = []
      use_teacher_forcing = False
      target_length = self.max_output_length
      decoder_attentions = torch.zeros(target_length, self.max_input_length)

    decoder_input = torch.tensor([[SOS]], device=DEVICE)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
      decoder_embedded = self.embedding(decoder_input).view(1, 1, -1)
      decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_embedded, decoder_hidden, encoder_outputs)
      if criterion:
        loss += criterion(decoder_output, target_tensor[di])
      if use_teacher_forcing:
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        _, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        if not criterion:
          decoded_tokens.append(decoder_input.item())
          decoder_attentions[di] = decoder_attention.data
        if decoder_input.item() == EOS: break
    
    if criterion:
      return loss
    else:
      return decoded_tokens, decoder_attentions[:di + 1]
