from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
import tensorflow as tf
import os
import shutil
import sys
from configSettings import Config
import zipfile
import ptb_reader

# Importing different files for different version of python.
if sys.version_info[0] == 2:
  import urllib2
else:
  import urllib.request as urllib2

flags = tf.flags
flags.DEFINE_string("test", None, "Path for explicit test file")
FLAGS = flags.FLAGS

class LSTM(object):
  def __init__(self, is_training, config):
    # Initialize the parameter values from the config.
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    dropout_probability = config.keep_prob
    hidden_dimension = config.hidden_size
    vocab_size = config.vocab_size
    batch_size = self.batch_size
    num_steps = self.num_steps

    self.input_data = tf.placeholder(tf.int32, shape = [batch_size, num_steps])
    
    # Convert input matrix to embedding matrix.
    embedding = tf.get_variable("embedding", [vocab_size, hidden_dimension], 
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale))
    input_vector_embedding = tf.nn.embedding_lookup(embedding, self.input_data)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dimension)
    
    # Applying dropout over the input to LSTM cell i.e. from input at time t 
    # and from hidden state at time t - 1. 
    if is_training:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = dropout_probability)
      input_vector_embedding = tf.nn.dropout(input_vector_embedding, config.keep_prob)

    # Stacking of RNN with two layers of it.
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)

    self.initial_state = stacked_lstm_cell.zero_state(batch_size, tf.float32)
    output_vector_embedding = []
    hidden_cell_states = []
    current_state = self.initial_state

    # Recurrent unit for simulating RNN.
    with tf.variable_scope("LangaugeModel",
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)):
      for iter_ in range(num_steps):
        if (iter_ > 0):
          tf.get_variable_scope().reuse_variables()
        current_cell_input = input_vector_embedding[:, iter_, :]
        (current_cell_output, current_state) = stacked_lstm_cell(current_cell_input, current_state)
        output_vector_embedding.append(current_cell_output)
        hidden_cell_states.append(current_state)

    self.final_state = hidden_cell_states[-1]
    output = tf.reshape(tf.concat(output_vector_embedding, 1), [-1, hidden_dimension])
    self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.reshape(self.targets, [-1])
    weights = tf.ones([batch_size * num_steps])
    logits = tf.matmul(output, tf.transpose(embedding))

    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights], vocab_size)
    self.cost = tf.div(tf.reduce_sum(loss), batch_size)

    if is_training:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
      optimizer = tf.train.GradientDescentOptimizer(0.9)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    else:
      self.train_op = tf.no_op();
      
def run_epoch(sess,model,data):
  epoch_size=((len(data)//model.batch_size) - 1)//model.num_steps
  saver=tf.train.Saver()
  state = sess.run(model.initial_state)
  total_cost=0
  iterations=0
  for step,(x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    cost, state, _ = sess.run([model.cost, model.final_state, model.train_op], 
      feed_dict={model.input_data: x,model.targets: y,model.initial_state: state})
    total_cost += cost
    iterations += model.num_steps
    perplexity = np.exp(total_cost / iterations)
    if step % 100 == 0:
        progress = (step *1.0/ epoch_size) * 100
        print("%.1f%% Perplexity: %.3f (Cost: %.3f) " % (progress, perplexity, cost))
  save_path=saver.save(sess,"./saved_model_rnn/lstm-model.ckpt")
  return (total_cost / iterations), perplexity

def test_epoch(sess,model,data):
  saver=tf.train.Saver()
  saver.restore(sess, "./saved_model_rnn/lstm-model.ckpt")
  state = sess.run(model.initial_state)
  total_cost=0
  iterations=0
  epoch_size=((len(data)//model.batch_size) - 1)//model.num_steps
  for step,(x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    cost, state = sess.run([model.cost, model.final_state], 
      feed_dict={model.input_data: x,model.targets: y,model.initial_state: state})
    total_cost += cost
    iterations += model.num_steps
    perplexity = np.exp(total_cost / iterations)
  return (total_cost / iterations), perplexity


def main(_):
  train_config = Config()
  eval_config = Config()
  eval_config.num_steps = 1
  num_epochs = 20

  if not FLAGS.test:
    train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data("../data")
    with tf.Graph().as_default() and tf.Session() as session:
      with tf.variable_scope("model", reuse=None):
        train_model = LSTM(is_training=True, config=train_config)
      with tf.variable_scope("model", reuse=True):
        test_model = LSTM(is_training=False, config=eval_config)

      if not os.path.exists('saved_model_rnn'):
        os.makedirs('saved_model_rnn')
      else:
        shutil.rmtree('saved_model_rnn')
        os.makedirs('saved_model_rnn')
      session.run(tf.global_variables_initializer())
      for i in range(num_epochs):
        train_cost, train_perp = run_epoch(session, train_model, train_data)
        print("Epoch: %i Training Perplexity: %.3f (Cost: %.3f)" % (i + 1, train_perp, train_cost))
  else:
    if not os.path.exists('saved_model_rnn'):
      proxy = urllib2.ProxyHandler({'https': '10.3.100.207:8080'})
      opener = urllib2.build_opener(proxy)
      urllib2.install_opener(opener)

      with open('saved_model_rnn.zip','wb') as f:
          f.write(urllib2.urlopen("https://github.com/ravibansal/saved_files/raw/master/saved_model_rnn.zip").read())
          f.close()

      zip_ref = zipfile.ZipFile('./saved_model_rnn.zip', 'r')
      zip_ref.extractall('./')
      zip_ref.close()

      with open('obj.zip','wb') as f:
          f.write(urllib2.urlopen("https://github.com/ravibansal/saved_files/raw/master/obj.zip").read())
          f.close()

      zip_ref = zipfile.ZipFile('./obj.zip', 'r')
      zip_ref.extractall('./')
      zip_ref.close()

    with tf.Graph().as_default() and tf.Session() as session:
      test_data = ptb_reader.ptb_test_data(FLAGS.test)
      with tf.variable_scope("model", reuse=None):
        train_model = LSTM(is_training=True, config=train_config)
      with tf.variable_scope("model", reuse=True):
        test_model = LSTM(is_training=False, config=eval_config)
      session.run(tf.global_variables_initializer())
      test_cost, test_perp = test_epoch(session, test_model, test_data)
      print(test_perp)

if __name__ == "__main__":
  tf.app.run()