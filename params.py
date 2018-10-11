from typing import Optional


class Params:
  # Model architecture
  vocab_size: int = 15000
  hidden_size: int = 50  # of the encoder; decoder size is doubled if encoder is bidi
  embed_size: int = 100
  enc_bidi: bool = True
  enc_attn: bool = True  # decoder has attention over encoder states?
  dec_attn: bool = False  # decoder has attention over previous decoder states?
  pointer: bool = True  # use pointer network (copy mechanism) in addition to word generator?
  out_embed_size: Optional[int] = None  # if set, use an additional layer before decoder output
  tie_embed: bool = True  # tie the decoder output layer to the input embedding layer?

  # Coverage (to turn on/off, change both `enc_attn_cover` and `cover_loss`)
  enc_attn_cover: bool = True  # provide coverage as input when computing enc attn?
  cover_func: str = 'max'  # how to aggregate previous attention distributions? sum or max
  cover_loss: float = 1  # add coverage loss if > 0; weight of coverage loss as compared to NLLLoss

  # Regularization
  enc_rnn_dropout: float = 0
  dec_in_dropout: float = 0
  dec_rnn_dropout: float = 0
  dec_out_dropout: float = 0

  # Training
  lr: float = 0.001  # learning rate
  batch_size: int = 128
  n_batches: int = 500  # how many batches per epoch
  val_batch_size: int = 128
  n_val_batches: int = 50  # how many validation batches per epoch
  n_epochs: int = 15
  pack_seq: bool = True  # use packed sequence to skip PAD inputs?
  forcing_ratio: float = 0.5  # percentage of using teacher forcing
  partial_forcing: bool = True  # in a seq, can some steps be teacher forced and some not?
  # note: enabling reinforcement learning can significantly slow down training
  rl_ratio: float = 0  # use mixed objective if > 0; ratio of RL in the loss function
  rl_ratio_power: float = 1  # increase rl_ratio by **= rl_ratio_power after each epoch; (0, 1]
  rl_start_epoch: int = 1  # start RL at which epoch (later start can ensure a strong baseline)?

  # Data
  embed_file: Optional[str] = 'data/.vector_cache/glove.6B.100d.txt'  # use pre-trained embeddings
  data_path: str = 'data/sent.txt'
  val_data_path: Optional[str] = 'data/sent.val.txt'
  max_src_len: int = 80
  max_tgt_len: int = 25
  truncate_src: bool = True  # truncate to max_src_len? if false, drop example if too long
  truncate_tgt: bool = True  # truncate to max_tgt_len? if false, drop example if too long

  # Saving model automatically during training
  model_path_prefix: Optional[str] = 'checkpoints/ml'
  keep_every_epoch: bool = False  # save all epochs, or only the best and the latest one?
