from train import tensorFromSentence
from model import DEVICE
from utils import MAX_LENGTH, SOS, EOS
import torch

def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
  with torch.no_grad():
    input_tensor = tensorFromSentence(input_lang, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    for ei in range(input_length):
      encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
      encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=DEVICE)
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
      decoder_attentions[di] = decoder_attention.data
      topv, topi = decoder_output.data.topk(1)
      if topi.item() == EOS:
        decoded_words.append('<EOS>')
        break
      else:
        decoded_words.append(output_lang.index2word[topi.item()])
      decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]
