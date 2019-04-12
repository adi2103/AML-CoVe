import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.initializers import RandomUniform


class LSTMDecoder(tf.keras.Model):
    # Constructor #
    def __init__(self, embedding_dim, batch_size,
                 drop_out, max_input_size,
                 vocab_size, r_drop_out=0.0):
        super(LSTMDecoder, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.drop_out = drop_out
        self.r_drop_out = r_drop_out
        self.max_input_size = max_input_size
        self.vocab_size = vocab_size

        self.embed_l = Embedding(input_dim=self.vocab_size, output_dim=embedding_dim,
                                 embeddings_initializer=RandomUniform(minval=-1, maxval=1, seed=1234),
                                 input_length=1,
                                 trainable=False)

        self.lstm_l1 = LSTM(embedding_dim * 2,
                            return_sequences=True,
                            return_state=True,
                            recurrent_activation='sigmoid',
                            dropout=self.drop_out,
                            recurrent_dropout=self.r_drop_out,
                            name="uni-lstm1")

        self.lstm_l2 = LSTM(embedding_dim * 2,
                            return_sequences=True,
                            return_state=True,
                            recurrent_activation='sigmoid',
                            dropout=self.drop_out,
                            recurrent_dropout=self.r_drop_out,
                            name="uni-lstm2")

        self.dense_l1 = Dense(self.embedding_dim * 2)

        self.dropout_l1 = Dropout(rate=self.drop_out)

        self.dense_l2 = Dense(self.embedding_dim * 2, activation='tanh')

        self.dropout_l2 = Dropout(rate=self.drop_out)

        self.dense_l3 = Dense(self.vocab_size)

        self.dropout_l3 = Dropout(rate=self.drop_out)

    def call(self, z_t, h_t_1, c_t_1, h_t_2, c_t_2, H, context):
        # Input dims:
        # z_t = [batch, 1] (input must be 2D)
        # h_t, c_t = [batch, 600]
        # H = [batch, 35, 600]
        # context = [batch, 600]

        ## Equation 2 ##
        embed_target = self.embed_l(z_t)
        # embed_target is [batch, 1, 300]
        # Reshape context to match embed_target (input must be 3D)
        context = tf.expand_dims(context, 1)
        hidden_input = K.concatenate([embed_target, context], axis=-1)
        # hidden_input is [batch, 1, 900]
        ht_dec1, h_t_new, c_t_new = self.lstm_l1(hidden_input, initial_state=[h_t_1, c_t_1])
        _, h_t_dec2, c_t_2_new = self.lstm_l2(ht_dec1, initial_state=[h_t_2, c_t_2])
        # h_t_new, c_t_new are [batch, 600]

        ## Equation 3 ##
        alpha = self.dense_l1(h_t_dec2)
        # alpha is [batch, 600]
        alpha = self.dropout_l1(alpha)
        alpha = tf.expand_dims(alpha, 2)
        # alpha is [batch, 600, 1]
        alpha = tf.linalg.matmul(H, alpha)
        # alpha is [batch, 35, 1]
        alpha = activations.softmax(alpha, axis=1)
        # alpha is [batch, 35, 1]

        ## Equation 4 ##
        # Transpose the H and multiply by attention
        context = tf.linalg.matmul(H, alpha,
                                   transpose_a=True)
        # context: [batch, 600, 1] -> [batch,600]
        context = tf.squeeze(context, axis=2)
        context = tf.concat([h_t_dec2, context], axis=1)
        # context is [batch, 1200]
        context = self.dense_l2(context)
        context = self.dropout_l2(context)
        # context is [batch, 600]

        ## Output ##
        output = self.dense_l3(context)
        output = self.dropout_l3(output)
        # output is [batch, vocab_size] and is logits

        return output, h_t_new, c_t_new, h_t_dec2, c_t_2_new, context

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, 2 * self.embedding_dim))
