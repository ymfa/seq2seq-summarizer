import torch
import tarfile
from io import BytesIO
from typing import Dict, Tuple, List, Union, Optional
from utils import rouge, Vocab, OOVDict, Batch, format_tokens, format_rouge_scores, Dataset
from model import DEVICE, Seq2SeqOutput, Seq2Seq
from params import Params
from tqdm import tqdm


def decode_batch_output(decoded_tokens, vocab: Vocab, oov_dict: OOVDict) -> List[List[str]]:
  """Convert word indices to strings."""
  decoded_batch = []
  if not isinstance(decoded_tokens, list):
    decoded_tokens = decoded_tokens.transpose(0, 1).tolist()
  for i, doc in enumerate(decoded_tokens):
    decoded_doc = []
    for word_idx in doc:
      if word_idx >= len(vocab):
        word = oov_dict.index2word.get((i, word_idx), '<UNK>')
      else:
        word = vocab[word_idx]
      decoded_doc.append(word)
      if word_idx == vocab.EOS:
        break
    decoded_batch.append(decoded_doc)
  return decoded_batch


def decode_batch(batch: Batch, model: Seq2Seq, vocab: Vocab, criterion=None, *, pack_seq=True,
                 show_cover_loss=False) -> Tuple[List[List[str]], Seq2SeqOutput]:
  """Test the `model` on the `batch`, return the decoded textual tokens and the Seq2SeqOutput."""
  if not pack_seq:
    input_lengths = None
  else:
    input_lengths = batch.input_lengths
  with torch.no_grad():
    input_tensor = batch.input_tensor.to(DEVICE)
    if batch.target_tensor is None or criterion is None:
      target_tensor = None
    else:
      target_tensor = batch.target_tensor.to(DEVICE)
    out = model(input_tensor, target_tensor, input_lengths, criterion,
                ext_vocab_size=batch.ext_vocab_size, include_cover_loss=show_cover_loss)
    decoded_batch = decode_batch_output(out.decoded_tokens, vocab, batch.oov_dict)
  target_length = batch.target_tensor.size(0)
  out.loss_value /= target_length
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


def eval_batch(batch: Batch, model: Seq2Seq, vocab: Vocab, criterion=None, *, pack_seq=True,
               show_cover_loss=False) -> Tuple[float, float]:
  """Test the `model` on the `batch`, return the ROUGE score and the loss."""
  decoded_batch, out = decode_batch(batch, model, vocab, criterion=criterion, pack_seq=pack_seq,
                                    show_cover_loss=show_cover_loss)
  examples = batch[0]
  gold_summaries = [ex.tgt for ex in examples]
  scores = rouge(gold_summaries, decoded_batch)
  return out.loss_value, scores[0]['l_f']


def eval_batch_output(tgt_tensor_or_tokens: Union[torch.Tensor, List[List[str]]], vocab: Vocab,
                      oov_dict: OOVDict, *pred_tensors: torch.Tensor) -> List[Dict[str, float]]:
  """
  :param tgt_tensor_or_tokens: the gold standard, either as indices or textual tokens
  :param vocab: the fixed-size vocab
  :param oov_dict: out-of-vocab dict
  :param pred_tensors: one or more systems' prediction (output tensors)
  :return: two-level score lookup (system index => ROUGE metric => value)

  Evaluate one or more systems' output.
  """
  decoded_batch = [decode_batch_output(pred_tensor, vocab, oov_dict)
                   for pred_tensor in pred_tensors]
  if isinstance(tgt_tensor_or_tokens, torch.Tensor):
    gold_summaries = decode_batch_output(tgt_tensor_or_tokens, vocab, oov_dict)
  else:
    gold_summaries = tgt_tensor_or_tokens
  scores = rouge(gold_summaries, *decoded_batch)
  return scores


def eval_bs_batch(batch: Batch, model: Seq2Seq, vocab: Vocab, *, pack_seq=True, beam_size=4,
                  min_out_len=1, max_out_len=None, len_in_words=True, best_only=True,
                  details: bool=True) -> Tuple[Optional[List[Dict[str, float]]], Optional[str]]:
  """
  :param batch: a test batch of a single example
  :param model: a trained summarizer
  :param vocab: vocabulary of the trained summarizer
  :param pack_seq: currently has no effect as batch size is 1
  :param beam_size: the beam size
  :param min_out_len: required minimum output length
  :param max_out_len: required maximum output length (if None, use the model's own value)
  :param len_in_words: if True, count output length in words instead of tokens (i.e. do not count
                       punctuations)
  :param best_only: if True, run ROUGE only on the best hypothesis instead of all `beam size` many
  :param details: if True, also return a string containing the result of this document
  :return: two-level score lookup (hypothesis index => ROUGE metric => value)

  Test a trained summarizer on a document using the beam search decoder.
  """
  assert len(batch.examples) == 1
  with torch.no_grad():
    input_tensor = batch.input_tensor.to(DEVICE)
    hypotheses = model.beam_search(input_tensor, batch.input_lengths if pack_seq else None,
                                   batch.ext_vocab_size, beam_size, min_out_len=min_out_len,
                                   max_out_len=max_out_len, len_in_words=len_in_words)
  if best_only:
    to_decode = [hypotheses[0].tokens]
  else:
    to_decode = [h.tokens for h in hypotheses]
  decoded_batch = decode_batch_output(to_decode, vocab, batch.oov_dict)
  if details:
    file_content = "[System Summary]\n" + format_tokens(decoded_batch[0])
  else:
    file_content = None
  if batch.examples[0].tgt is not None:  # run ROUGE if gold standard summary exists
    gold_summaries = [batch.examples[0].tgt for _ in range(len(decoded_batch))]
    scores = rouge(gold_summaries, decoded_batch)
    if details:
      file_content += "\n\n\n[Reference Summary]\n" + format_tokens(batch.examples[0].tgt)
      file_content += "\n\n\n[ROUGE Scores]\n" + format_rouge_scores(scores[0]) + "\n"
  else:
    scores = None
  if details:
    file_content += "\n\n\n[Source Text]\n" + format_tokens(batch.examples[0].src)
  return scores, file_content


def eval_bs(test_set: Dataset, vocab: Vocab, model: Seq2Seq, params: Params):
  test_gen = test_set.generator(1, vocab, None, True if params.pointer else False)
  n_samples = int(params.test_sample_ratio * len(test_set.pairs))

  if params.test_save_results and params.model_path_prefix:
    result_file = tarfile.open(params.model_path_prefix + ".results.tgz", 'w:gz')
  else:
    result_file = None

  model.eval()
  r1, r2, rl, rsu4 = 0, 0, 0, 0
  prog_bar = tqdm(range(1, n_samples + 1))
  for i in prog_bar:
    batch = next(test_gen)
    scores, file_content = eval_bs_batch(batch, model, vocab, pack_seq=params.pack_seq,
                                         beam_size=params.beam_size,
                                         min_out_len=params.min_out_len,
                                         max_out_len=params.max_out_len,
                                         len_in_words=params.out_len_in_words,
                                         details=result_file is not None)
    if file_content:
      file_content = file_content.encode('utf-8')
      file_info = tarfile.TarInfo(name='%06d.txt' % i)
      file_info.size = len(file_content)
      result_file.addfile(file_info, fileobj=BytesIO(file_content))
    if scores:
      r1 += scores[0]['1_f']
      r2 += scores[0]['2_f']
      rl += scores[0]['l_f']
      rsu4 += scores[0]['su4_f']
      prog_bar.set_postfix(R1='%.4g' % (r1 / i * 100), R2='%.4g' % (r2 / i * 100),
                           RL='%.4g' % (rl / i * 100), RSU4='%.4g' % (rsu4 / i * 100))


if __name__ == "__main__":
  import argparse
  import os.path

  parser = argparse.ArgumentParser(description='Evaluate a summarization model.')
  parser.add_argument('--model', type=str, metavar='M', help='path to the model to be evaluated')
  args, unknown_args = parser.parse_known_args()

  p = Params()
  if unknown_args:  # allow command line args to override params.py
    p.update(unknown_args)

  if args.model:  # evaluate a specific model
    filename = args.model
  else:  # evaluate the best model
    train_status = torch.load(p.model_path_prefix + ".train.pt")
    filename = '%s.%02d.pt' % (p.model_path_prefix, train_status['best_epoch_so_far'])

  print("Evaluating %s..." % filename)
  m = torch.load(filename)  # use map_location='cpu' if you are testing a CUDA model using CPU

  m.encoder.gru.flatten_parameters()
  m.decoder.gru.flatten_parameters()

  if hasattr(m, 'vocab'):
    v = m.vocab
  else:  # fixes for models trained by a previous version of the summarizer
    filename, _ = os.path.splitext(p.data_path)
    if p.vocab_size:
      filename += ".%d" % p.vocab_size
    filename += '.vocab'
    v = torch.load(filename)
    m.vocab = v
    m.max_dec_steps = m.max_output_length

  d = Dataset(p.test_data_path)
  eval_bs(d, v, m, p)
