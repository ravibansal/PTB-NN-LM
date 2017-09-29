class Config(object):
    batch_size = 20
    num_steps = 35  # number of unrolled time steps
    hidden_size = 450 # number of blocks in an LSTM cell
    vocab_size = 10000
    max_grad_norm = 5 # maximum gradient for clipping
    init_scale = 0.05 # scale between -0.1 and 0.1 for all random initialization
    keep_prob = 0.5 # dropout probability
    num_layers = 2 # number of LSTM layers
    learning_rate = 1.0
    lr_decay = 0.8
    lr_decay_epoch_offset = 6 # don't decay until after the Nth epoch