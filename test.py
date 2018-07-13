from train import tensorFromSentence
import torch

def evaluate(input_lang, output_lang, model, sentence):
  with torch.no_grad():
    input_tensor = tensorFromSentence(input_lang, sentence)
    decoded_tokens, decoder_attentions = model(input_tensor)
    decoded_sentence = ' '.join(output_lang.index2word[t] for t in decoded_tokens)
  return decoded_sentence, decoder_attentions
