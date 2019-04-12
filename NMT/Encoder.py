import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.layers import Bidirectional, LSTM, Masking, Dense, Dropout, Reshape, Lambda, Conv1D, Add, \
    MaxPooling1D, Layer, Concatenate, BatchNormalization
from tensorflow.keras import backend as K


# Main encoder model
class LSTMEncoder(tf.keras.Model):
    # Constructor #
    def __init__(self, batch_size, drop_out, embedding_dim, max_input_size, r_drop_out=0.0):
        super(LSTMEncoder, self).__init__()
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.r_drop_out = r_drop_out
        self.embedding_dim = embedding_dim
        self.max_input_size = max_input_size

        self.mask_l = Masking(mask_value=0,
                              input_shape=(None, self.embedding_dim))
        self.lstm_l1 = Bidirectional(LSTM(self.embedding_dim,
                                          return_sequences=True,
                                          recurrent_activation='sigmoid',
                                          dropout=self.drop_out,
                                          recurrent_dropout=self.r_drop_out),
                                     name="layer1")

        self.lstm_l2 = Bidirectional(LSTM(self.embedding_dim,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_activation='sigmoid',
                                          dropout=self.drop_out,
                                          recurrent_dropout=self.r_drop_out),
                                     name="layer2")

    def call(self, X):
        masking_layer = self.mask_l(X)
        lstm_layer1 = self.lstm_l1(masking_layer)

        H, h_t_FW, FW_cell, h_t_BW, BW_cell = self.lstm_l2(lstm_layer1)
        cell_state = K.concatenate([FW_cell, BW_cell], axis=-1)
        hidden_state = K.concatenate([h_t_FW, h_t_BW], axis=-1)

        return H, hidden_state, cell_state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.max_input_size))


class CNNblock(tf.keras.layers.Layer):
  def __init__(self, kernel_size, embedding_dim, dilation_rate, drop_out, batch_size, max_input_size):
      super(CNNblock, self).__init__()
      self.kernel_size = kernel_size
      self.dilation_rate = dilation_rate
      self.drop_out = drop_out
      self.embedding_dim = embedding_dim
      self.batch_size = batch_size
      self.max_input_size = max_input_size
      self.c1_matrix = tf.constant(2.0, shape=(self.batch_size, self.max_input_size , self.embedding_dim))
      self.c2_matrix = tf.constant(0.5, shape=(self.batch_size, self.max_input_size, self.embedding_dim))

      self.cnn1 = Conv1D(filters=self.embedding_dim,
                         kernel_size=self.kernel_size,
                         dilation_rate=self.dilation_rate[0],
                         padding='same',
                         activation='linear')

      self.cnn2 = Conv1D(filters=self.embedding_dim,
                         kernel_size=self.kernel_size,
                         dilation_rate=self.dilation_rate[1],
                         padding='same',
                         activation='linear')

      self.drop1 = Dropout(rate=self.drop_out)
      self.drop2 = Dropout(rate=self.drop_out)

      self.skip_a = Add()

      self.multiply= Multiply()

  def call(self, X, Mask):
      cnn_layer1 = self.cnn1(X)
      cnn_layer1_m = self.multiply([Mask, cnn_layer1 ])
      output1 = tf.nn.tanh(cnn_layer1_m)
      cnn_layerd1 = self.drop1(output1)

      cnn_layer2 = self.cnn2(cnn_layerd1)
      cnn_layer2_m = self.multiply([Mask, cnn_layer2])
      output2 = tf.nn.tanh(cnn_layer2_m)
      logits = self.drop2(output2)

      logits_up = self.multiply([logits, self.c1_matrix])

      skipped = self.skip_a([X, logits_up])

      normalized = self.multiply([skipped, self.c2_matrix])

      return normalized

class CNNEncoder(tf.keras.Model):
    # Constructor #
    def __init__(self, batch_size, drop_out, embedding_dim,
                 max_input_size, kernel_size):
        super(CNNEncoder, self).__init__()
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.max_input_size = max_input_size
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size

        self.pos_vec = pos_encoding(self.max_input_size, self.embedding_dim)
        self.pos_matrix = tf.convert_to_tensor(np.repeat(self.pos_vec[np.newaxis, ...], self.batch_size, axis=0),
                                               dtype=tf.float32)

        self.cnn_f1 = CNNblock(self.kernel_size, self.embedding_dim, [1, 2], self.drop_out, self.batch_size, self.max_input_size)
        self.cnn_f2 = CNNblock(self.kernel_size, self.embedding_dim,  [4, 8], self.drop_out, self.batch_size, self.max_input_size)

        self.cnn_b1 = CNNblock(self.kernel_size ,self.embedding_dim, [1, 2], self.drop_out, self.batch_size, self.max_input_size)
        self.cnn_b2 = CNNblock(self.kernel_size, self.embedding_dim, [4, 8], self.drop_out, self.batch_size, self.max_input_size)

        self.skip_c = Concatenate()
        self.skip_a = Add()
        self.multiply = Multiply()

    def call(self, X, Mask):
        # Generate contextualized embeddings
        embedding_pos = self.skip_a([X, self.pos_matrix])

        embedding_pos_m = self.multiply([embedding_pos, Mask])

        skipped_f1 = self.cnn_f1(embedding_pos_m, Mask)
        skipped_b1 = self.cnn_b1(embedding_pos_m, Mask)

        skipped_f2 = self.cnn_f2(skipped_f1, Mask)
        skipped_b2 = self.cnn_b2(skipped_b1, Mask)

        H= self.skip_c([skipped_f2, skipped_b2])

        return H, None, None, None, None

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.max_input_size))



# Positional encoding layer
def pos_encoding(max_len, d_emb):
  pos_enc = np.zeros((max_len, d_emb))
  for i in range(max_len):
    for j in range(0, d_emb, 2):
      pos_enc[i, j] = (np.sin(i/10000**(2*j//d_emb)))
      pos_enc[i, j+1] = (np.cos(i/10000**(2*(j+1)//d_emb)))
  return pos_enc


# Self attention layer
class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_k, batch_size, max_input_size):
    super(SelfAttention, self).__init__()
    self.d_k = d_k
    self.batch_size = batch_size
    self.max_input_size = max_input_size
    self.d_k_matrix = tf.constant(1/math.sqrt(d_k), shape=(self.batch_size, self.max_input_size, self.max_input_size))
    self.Q = Dense(units=self.d_k)
    self.K = Dense(units=self.d_k)
    self.V = Dense(units=self.d_k)

  def call(self, X):
    # X, Q, K, V = [batch, sentence_length, embedding_size]
    Q = self.Q(X)
    K = self.K(X)
    V = self.V(X)
    prod = tf.linalg.matmul(Q, K, transpose_b=True)
    # prod, scaled, softmaxed = [batch, sentence_length, sentence_length]
    scaled = tf.math.multiply(prod, self.d_k_matrix)
    softmaxed = tf.nn.softmax(scaled, axis=1)
    output = tf.linalg.matmul(softmaxed, V)
    # output = [batch, sentence_length, embedding_size]
    return output


# Multi-headed self attention layer
class MultiHeaded(tf.keras.layers.Layer):
    def __init__(self, d_k, embedding_dim, max_input_size, batch_size):
        super(MultiHeaded, self).__init__()
        self.d_k = d_k  # 100
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim  # 300
        self.max_input_size = max_input_size
        self.h1 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.h2 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.h3 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.h4 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.h5 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.h6 = SelfAttention(self.d_k, self.batch_size, self.max_input_size)
        self.dense = Dense(units=2 * self.embedding_dim)
        self.reshape = Reshape((max_input_size, 2*self.embedding_dim))

    def call(self, X):
        self_attn_1 = self.h1(X)
        self_attn_2 = self.h2(X)
        self_attn_3 = self.h3(X)
        self_attn_4 = self.h4(X)
        self_attn_5 = self.h5(X)
        self_attn_6 = self.h6(X)

        Z = K.concatenate([self_attn_1,
                           self_attn_2,
                           self_attn_3,
                           self_attn_4,
                           self_attn_5,
                           self_attn_6], axis=2)
        output = self.dense(Z)

        return output


# Main encoder model
class ATTNEncoder(tf.keras.Model):
    # Constructor #
    def __init__(self, batch_size, drop_out,
                 max_input_size, embedding_dim, d_k):
        super(ATTNEncoder, self).__init__()
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.max_input_size = max_input_size
        self.embedding_dim = embedding_dim
        self.pos_vec = pos_encoding(self.max_input_size, self.embedding_dim)
        self.pos_matrix = tf.convert_to_tensor(np.repeat(self.pos_vec[np.newaxis, ...], self.batch_size, axis=0),
                                               dtype=tf.float32)
        self.add = Add()
        self.dropout_l1 = Dropout(rate=self.drop_out)
        self.dropout_l2 = Dropout(rate=self.drop_out)
        self.dropout_l3 = Dropout(rate=self.drop_out)
        self.dropout_l4 = Dropout(rate=self.drop_out)
        self.dropout_l5 = Dropout(rate=self.drop_out)
        self.d_k = d_k
        self.multi_head_l1 = MultiHeaded(self.d_k, self.embedding_dim, self.max_input_size, self.batch_size)
        self.multi_head_l2 = MultiHeaded(self.d_k, self.embedding_dim, self.max_input_size, self.batch_size)
        self.concat = Concatenate()
        self.dense_l1 = Dense(units=2 * self.embedding_dim, activation='tanh')
        self.dense_l2 = Dense(units=2 * self.embedding_dim, activation='tanh')
        self.normalize = BatchNormalization(axis=2)
        self.max_l1 = MaxPooling1D(data_format='channels_last',
                                   pool_size=self.max_input_size)

        self.reshape_l1 = Reshape((2 * self.embedding_dim,))

    def call(self, X):
        # X = [batch_size, sentence_length]
        # embedding_layer, embedding_pos, dropped_embed_pos = [batch]

        embedding_pos = self.add([X, self.pos_matrix])
        dropped_embed_pos = self.dropout_l1(embedding_pos)
        embed_cat = self.concat([embedding_pos, embedding_pos])
        attn = self.multi_head_l1(dropped_embed_pos)
        attn_dropped = self.dropout_l2(attn)

        skipped = self.add([attn_dropped, embed_cat])
        # skipped = [batch_size, sentence_length, 600]

        mean, variance = tf.nn.moments(skipped, axes=[0], keepdims=False)
        normalized = tf.nn.batch_normalization(skipped, mean=mean,
                                               variance=variance, offset=None,
                                               scale=None,
                                               variance_epsilon=0.001)  # from Keras layer

        ff = self.dense_l1(normalized)
        ff_dropped = self.dropout_l3(ff)
        skipped_2 = self.add([ff_dropped, normalized])

        mean, variance = tf.nn.moments(skipped_2, axes=[0], keepdims=False)
        normalized_2 = tf.nn.batch_normalization(skipped_2, mean=mean,
                                               variance=variance, offset=None,
                                               scale=None,
                                               variance_epsilon=0.001)  # from Keras layer

        attn2 = self.multi_head_l2(normalized_2)
        attn2_dropped = self.dropout_l4(attn2)
        skipped_3 = self.add([attn2_dropped, normalized_2])

        mean, variance = tf.nn.moments(skipped_3, axes=[0], keepdims=False)
        normalized_3 = tf.nn.batch_normalization(skipped_3, mean=mean,
                                               variance=variance, offset=None,
                                               scale=None,
                                               variance_epsilon=0.001)  # from Keras layer

        ff_2 = self.dense_l2(normalized_3)
        ff_2_dropped = self.dropout_l5(ff_2)
        skipped_4 = self.add([ff_2_dropped, normalized_3])

        mean, variance = tf.nn.moments(skipped_4, axes=[0], keepdims=False)
        H = tf.nn.batch_normalization(skipped_4, mean=mean,
                                               variance=variance, offset=None,
                                               scale=None,
                                               variance_epsilon=0.001)  # from Keras layer

        return H, None, None, None, None

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.max_input_size))

