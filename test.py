import torch

def evaluate(vocab, model, input_tensor):
  with torch.no_grad():
    decoded_tokens, decoder_attentions = model(input_tensor)
    decoded_sentence = ' '.join(vocab.index2word[t] for t in decoded_tokens[0])
  return decoded_sentence, decoder_attentions[:,0,:]


if __name__ == "__main__":
  from utils import Dataset
  from model import Seq2Seq
  from params import *
  import matplotlib.pyplot as plt

  dataset = Dataset(data_path, max_src_len=80, max_tgt_len=25)
  vocabulary = dataset.build_vocab('english', vocab_size, True, True)
  test_data = dataset.generator(1, vocabulary)
  m = Seq2Seq(vocabulary, embed_size, hidden_size, dataset.src_len, dataset.tgt_len)

  saved_model = torch.load(model_path)
  m.load_state_dict(saved_model)

  examples, src_tensor, lengths = next(test_data)
  print(examples[0])

  pred, attention = evaluate(vocabulary, m, src_tensor)
  print(pred)
  plt.matshow(attention.numpy())
  plt.show()
