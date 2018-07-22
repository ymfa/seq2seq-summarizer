import torch
from typing import Dict, Tuple
from utils import rouge
from model import DEVICE


def decode_batch(batch, model, vocab, criterion=None, *, pack_seq=True):
  examples, input_tensor, target_tensor, input_lengths, oov_dict = batch
  if criterion is None:
    target_tensor = None  # output length not controlled by target length
  else:
    target_length = target_tensor.size(0)
  if not pack_seq:
    input_lengths = None
  # construct reverse lookup table for oov_dict
  oov_idx2word = {}  # type: Dict[Tuple[int, int], str]
  ext_vocab_size = None
  if oov_dict is not None:
    for key, value in oov_dict.items():
      if key == 'size':
        ext_vocab_size = value
      else:
        i, word = key
        oov_idx2word[(i, value)] = word
  # convert word indices to strings
  with torch.no_grad():
    word_indices, additional_info = model(input_tensor.to(DEVICE), target_tensor.to(DEVICE),
                                          input_lengths, criterion, ext_vocab_size=ext_vocab_size)
    decoded_batch = []
    for i, doc in enumerate(word_indices.numpy().astype(int)):
      decoded_doc = []
      for word_idx in doc:
        word_idx = word_idx.item()
        if word_idx >= len(vocab):
          word = oov_idx2word.get((i, word_idx), '<UNK>')
        else:
          word = vocab[word_idx]
        decoded_doc.append(word)
        if word_idx == vocab.EOS:
          break
      decoded_batch.append(decoded_doc)
  if type(additional_info) is not tuple:
    additional_info = additional_info.item() / target_length
  return decoded_batch, additional_info


def decode_one(*args, **kwargs):
  decoded_batch, additional_info = decode_batch(*args, **kwargs)
  decoded_doc = decoded_batch[0]
  if type(additional_info) is tuple:
    attn_weights, ptr_probs = additional_info
    if attn_weights is not None:
      attn_weights = attn_weights[:len(decoded_doc), 0, :]
    if ptr_probs is not None:
      ptr_probs = ptr_probs[:len(decoded_doc), 0]
    additional_info = attn_weights, ptr_probs
  return decoded_doc, additional_info


def eval_batch(batch, model, vocab, criterion=None, *, pack_seq=True):
  decoded_batch, additional_info = decode_batch(batch, model, vocab, criterion=criterion,
                                                pack_seq=pack_seq)
  if type(additional_info) is tuple:
    loss = None
  else:
    loss = additional_info
  examples = batch[0]
  gold_summaries = [ex.tgt for ex in examples]
  scores = rouge(decoded_batch, gold_summaries)
  return scores['su4_f'], loss  # use ROUGE-SU4 for now
