import torch
from typing import Dict, Tuple
from utils import rouge, Vocab
from model import DEVICE, Seq2SeqOutput
from typing import List, Union


def reverse_dict(oov_dict):
  """Construct reverse lookup table for oov_dict."""
  oov_idx2word = {}  # type: Dict[Tuple[int, int], str]
  ext_vocab_size = None
  if oov_dict is not None:
    for key, value in oov_dict.items():
      if key == 'size':
        ext_vocab_size = value
      else:
        i, word = key
        oov_idx2word[(i, value)] = word
  return oov_idx2word, ext_vocab_size


def decode_batch_output(decoded_tokens, vocab, oov_idx2word):
  """Convert word indices to strings."""
  decoded_batch = []
  for i, doc in enumerate(decoded_tokens.transpose(0, 1).numpy()):
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
  return decoded_batch


def decode_batch(batch, model, vocab, criterion=None, *, pack_seq=True) \
        -> Tuple[List[List[str]], Seq2SeqOutput]:
  """Test the `model` on the `batch`, return the decoded textual tokens and the Seq2SeqOutput."""
  examples, input_tensor, target_tensor, input_lengths, oov_dict = batch
  oov_idx2word, ext_vocab_size = reverse_dict(oov_dict)
  if criterion is None:
    target_tensor = None  # output length not controlled by target length
  else:
    target_length = target_tensor.size(0)
  if not pack_seq:
    input_lengths = None
  with torch.no_grad():
    input_tensor = input_tensor.to(DEVICE)
    if target_tensor is not None:
      target_tensor = target_tensor.to(DEVICE)
    out = model(input_tensor, target_tensor, input_lengths, criterion,
                ext_vocab_size=ext_vocab_size)
    decoded_batch = decode_batch_output(out.decoded_tokens, vocab, oov_idx2word)
  if isinstance(out.loss, torch.Tensor):  # iff criterion was provided
    out.loss = out.loss.item() / target_length
  return decoded_batch, out


def decode_one(*args, **kwargs):
  """
  Same as `decode_batch()` but because batch size is 1, the batch dim in visualization data is
  eliminated.
  """
  decoded_batch, out = decode_batch(*args, **kwargs)
  decoded_doc = decoded_batch[0]
  if out.enc_attn_weights is not None:
    out.enc_attn_weights = out.enc_attn_weights[:len(decoded_doc), 0, :]
  if out.ptr_probs is not None:
    out.ptr_probs = out.ptr_probs[:len(decoded_doc), 0]
  return decoded_doc, out


def eval_batch(batch, model, vocab, criterion=None, *, pack_seq=True) -> Tuple[float, float]:
  """Test the `model` on the `batch`, return the ROUGE score and the loss."""
  decoded_batch, out = decode_batch(batch, model, vocab, criterion=criterion, pack_seq=pack_seq)
  examples = batch[0]
  gold_summaries = [ex.tgt for ex in examples]
  scores = rouge(gold_summaries, decoded_batch)
  return out.loss, scores[0]['l_f']


def eval_batch_output(tgt_tensor_or_tokens: Union[torch.Tensor, List[List[str]]], vocab: Vocab,
                      oov_dict: dict, *pred_tensors: torch.Tensor) -> List[Dict[str, float]]:
  """
  :param tgt_tensor_or_tokens: the gold standard, either as indices or textual tokens
  :param vocab: the fixed-size vocab
  :param oov_dict: out-of-vocab dict
  :param pred_tensors: one or more systems' prediction (output tensors)
  :return: two-level score lookup (system index => ROUGE metric => value)

  Evaluate one or more systems' output.
  """
  oov_idx2word, ext_vocab_size = reverse_dict(oov_dict)
  decoded_batch = [decode_batch_output(pred_tensor, vocab, oov_idx2word)
                   for pred_tensor in pred_tensors]
  if isinstance(tgt_tensor_or_tokens, torch.Tensor):
    gold_summaries = decode_batch_output(tgt_tensor_or_tokens, vocab, oov_idx2word)
  else:
    gold_summaries = tgt_tensor_or_tokens
  scores = rouge(gold_summaries, *decoded_batch)
  return scores
