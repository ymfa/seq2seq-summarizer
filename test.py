import torch


def evaluate(vocab, model, input_tensor):
  with torch.no_grad():
    model.eval()
    decoded_tokens, decoder_attentions = model(input_tensor)
    decoded_sentence = []
    i = -1
    for i, token in enumerate(decoded_tokens[0]):
      decoded_sentence.append(vocab[token])
      if token == vocab.EOS:
        break
    decoded_sentence = ' '.join(decoded_sentence)
    decoder_attentions = decoder_attentions[:i+1, 0, :]
  return decoded_sentence, decoder_attentions
