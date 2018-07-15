import torch

def evaluate(vocab, model, input_tensor):
  with torch.no_grad():
    decoded_tokens, decoder_attentions = model(input_tensor)
    decoded_sentence = ' '.join(vocab.index2word[t] for t in decoded_tokens[0])
  return decoded_sentence, decoder_attentions[:,0,:]
