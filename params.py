# Model
vocab_size = 15000
hidden_size = 100
embed_size = 100
encoder_bidi = False
encoder_attn = True

# Training
learning_rate = 0.0005
batch_size = 32
num_batches = 6000
num_epochs = 15
use_packed_seq = True

embed_file = 'data/.vector_cache/glove.6B.100d.txt'
data_path = 'data/short.txt'
model_path_prefix = 'checkpoints/short_uni'
