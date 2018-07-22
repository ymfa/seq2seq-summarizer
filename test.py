import torch
from typing import Dict, Tuple
from utils import rouge
from model import DEVICE, Seq2SeqOutput
from typing import List


def decode_batch(batch, model, vocab, criterion=None, *, pack_seq=True) \
        -> Tuple[List[List[str]], Seq2SeqOutput]:
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
    out = model(input_tensor.to(DEVICE), target_tensor.to(DEVICE),
                input_lengths, criterion, ext_vocab_size=ext_vocab_size)
    decoded_batch = []
    for i, doc in enumerate(out.decoded_tokens.transpose(0, 1).numpy().astype(int)):
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
  if isinstance(out.loss, torch.Tensor):  # iff criterion was provided
    out.loss = out.loss.item() / target_length
  return decoded_batch, out


def decode_one(*args, **kwargs):
  decoded_batch, out = decode_batch(*args, **kwargs)
  decoded_doc = decoded_batch[0]
  if out.attn_weights is not None:
    out.attn_weights = out.attn_weights[:len(decoded_doc), 0, :]
  if out.ptr_probs is not None:
    out.ptr_probs = out.ptr_probs[:len(decoded_doc), 0]
  return decoded_doc, out


def eval_batch(batch, model, vocab, criterion=None, *, pack_seq=True) -> Tuple[float, float]:
  decoded_batch, out = decode_batch(batch, model, vocab, criterion=criterion, pack_seq=pack_seq)
  examples = batch[0]
  gold_summaries = [ex.tgt for ex in examples]
  scores = rouge(decoded_batch, gold_summaries)
  return scores['su4_f'], out.loss  # use ROUGE-SU4 for now
