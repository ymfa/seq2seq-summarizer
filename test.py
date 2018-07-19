import torch
from typing import Dict


def decode_one(vocab, model, input_tensor, oov_dict: Dict=None):
  idx_in_batch = 0
  ext_vocab_size = None
  if oov_dict is not None:
    oov_idx2word = {}  # type: Dict[int, str]
    for key, value in oov_dict.items():
      if key == 'size':
        ext_vocab_size = value
      else:
        i, word = key
        if i == idx_in_batch:
          oov_idx2word[value] = word
  with torch.no_grad():
    model.eval()
    decoded_tokens, attn_weights, ptr_probs = model(input_tensor, ext_vocab_size=ext_vocab_size)
    decoded_sentence = []
    i = -1
    for i, token in enumerate(decoded_tokens[0]):
      if token >= len(vocab):
        word = oov_idx2word[token]
      else:
        word = vocab[token]
      decoded_sentence.append(word)
      if token == vocab.EOS:
        break
    if attn_weights is not None:
      attn_weights = attn_weights[:i+1, idx_in_batch, :]
    if ptr_probs is not None:
      ptr_probs = ptr_probs[:i+1, idx_in_batch]
  return decoded_sentence, attn_weights, ptr_probs
