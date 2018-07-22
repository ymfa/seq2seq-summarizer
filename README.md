Lightweight PyTorch implementation of a seq2seq text summarizer.

#### Advantages
* Simple code structure, easy to understand.
* Minimal dependencies (Python 3.6, `torch`, `tqdm` and `matplotlib`).

#### Implemented
* Batch training/testing on GPU/CPU.
* Teacher forcing.
* Initialization with pre-trained word embeddings.
* Embedding sharing across encoder, decoder input, and decoder output.
* Attention mechanism.
* Pointer network, which copies words (can be out-of-vocabulary) from the source.
* Visualization of attention and pointer weights:

![Visualization](https://user-images.githubusercontent.com/6981180/42974503-765b21f6-8baf-11e8-8928-9b7a88b033a8.png)

* Validation using ROUGE: Please put ROUGE-1.5.5.pl and its "data" folder under data/; `pyrouge` is NOT required.

#### To be implemented
* Repetition avoidance.
* Run on longer texts (missing modern hardware $_$).
* Reinforcement learning.