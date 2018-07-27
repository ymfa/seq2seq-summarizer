from typing import Optional


class Params:
  # Model architecture
  vocab_size: int = 15000
  hidden_size: int = 50  # of the encoder; decoder size is doubled if encoder is bidi
  embed_size: int = 100
  enc_bidi: bool = True
  enc_attn: bool = True  # decoder has attention over encoder states?
  dec_attn: bool = True  # (not yet implemented)
  pointer: bool = True  # use pointer network (copy mechanism) in addition to word generator?
  out_embed_size: Optional[int] = None  # if set, use an additional layer before decoder output
  tie_embed: bool = True  # tie the decoder output layer to the input embedding layer?

  # Regularization
  enc_rnn_dropout: float = 0
  dec_in_dropout: float = 0
  dec_rnn_dropout: float = 0
  dec_out_dropout: float = 0

  # Training
  lr: float = 0.0005  # learning rate
  batch_size: int = 64
  n_batches: int = 1000  # how many batches per epoch
  val_batch_size: int = 128
  n_val_batches: int = 50  # how many validation batches per epoch
  n_epochs: int = 15
  pack_seq: bool = True  # use packed sequence to skip PAD inputs?
  forcing_ratio: float = 0.5  # percentage of using teacher forcing
  partial_forcing: bool = True  # in a seq, can some steps be teacher forced and some not?
  rl_ratio: float = 0.9  # use mixed objective if > 0; ratio of reinforcement learning

  # Data
  embed_file: Optional[str] = 'data/.vector_cache/glove.6B.100d.txt'  # use pre-trained embeddings
  data_path: str = 'data/sent.txt'
  val_data_path: Optional[str] = 'data/sent.val.txt'
  max_src_len: int = 80
  max_tgt_len: int = 25

  # Saving model automatically during training
  model_path_prefix: Optional[str] = 'checkpoints/rl'
  keep_every_epoch: bool = False  # save all epochs, or only the best and the latest one?
