import numpy as np
import tensorflow as tf


def positional_encoding(seq_len, embedding_dim, maximum_encoding=10e3):
    ''' Positional Encoding
        Formular:
            sin(pos/[maximum_encoding^(2i/embedding_dim)]), i = even
            cos(pos/[maximum_encoding^(2i/embedding_dim)]), i = odd
    '''
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embedding_dim)[np.newaxis, :]
    radians = pos / np.power(10000, 2 * (i // 2) / embedding_dim)
    # pos encoded below
    radians[:, 0::2] = np.sin(radians[:, 0::2])
    radians[:, 1::2] = np.cos(radians[:, 1::2])
    pos_enc = radians[np.newaxis, ...]
    return tf.cast(pos_enc, dtype=tf.float32)


def QKV_op(q, k, v):
    ''' [q, k, v] = [query, key, value]
        q shape: [batch_size, seq_len, dim]
        k shape: [batch_size, seq_len, dim]
        v shape: [batch_size, seq_len, dim]
        attention_weights shape: [batch_size, seq_len, seq_len]
        z shape: [batch_size, seq_len, dim]
    '''
    dk = tf.math.sqrt(tf.cast(tf.shape(k), tf.float32))
    attention_weights = tf.nn.softmax( \
        tf.matmul(q, k, transpose_b=True),
        axis=-1)
    z = tf.matmul(attention_weights, v)
    return z, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, units, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.units = units
        self.num_heads = num_heads
        # check if the depth for each head is an integer
        assert self.units % self.num_heads == 0
        self.depth = self.units // self.num_heads
        # dense layers
        self.q_dense = tf.keras.layers.Dense(self.units)
        self.k_dense = tf.keras.layers.Dense(self.units)
        self.v_dense = tf.keras.layers.Dense(self.units)
        self.out_dense = tf.keras.layers.Dense(self.units)

    def split(self, x):
        '''
            Return ---> x: [batch_size, num_heads, seq_len, depth]
        '''
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, x):
        ''' Input ---> x: [batch_size, seq_len, units from point wise dense]
            Return --> z: [batch_size, seq_len, self.units]
                  |--> attention_weights: [batch_size, seq_len, seq_len]
        '''
        batch_size = tf.shape(x)[0]

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        z, attention_weights = QKV_op(q, k, v)
        z = tf.transpose(z, perm=[0, 2, 1, 3])
        z = tf.reshape(z, (batch_size, -1, self.units))

        return z, attention_weights


class PointWiseFeedforwardLayer(tf.keras.layers.Layer):

    def __init__(self, units, output_units):
        super(PointWiseFeedforwardLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units)
        self.dense2 = tf.keras.layers.Dense(output_units)

    def call(self, x):
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, mha_units, num_heads, ffl_units, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        ''' 1. The shape of mha input/ffl output and
                mha output must be the same,
                since there is point wise addition.
                Thus ---> mha_units = tf.shape(x)[-1]

            2. The shape of mha output and ffl output
                must be the same,
                since there is point wise addition.
        '''

        self.mha = MultiHeadAttention(mha_units, num_heads)
        self.ffl = PointWiseFeedforwardLayer(ffl_units, mha_units)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=True):
        mha_output, mha_weights = self.mha(x)
        mha_output = self.dropout1(mha_output, training=training)
        out1 = self.layernorm1(x + mha_output)

        ffl_output = self.ffl(out1)
        ffl_output = self.dropout2(ffl_output, training=training)
        out2 = self.layernorm2(out1 + ffl_output)

        return out2


class EncoderLayers(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim,
                 max_len, num_heads, num_layers,
                 ffl_units, rate=0.1):
        super(EncoderLayers, self).__init__()

        ''' Note: (embedding_dim % num_heads) should be an integer!!!
            1. embedding_dim = mha_units, since point wise addition.
            2. ffl_out_units = embedding_dim as well, same reason.
            3. ffl has 2 layers, the 1st layer is defined by ffl_units,
               the 2nd layer units = mha_units = embedding_dim.
               Note: you can customize it so all units can be differ.
        '''
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_enc = positional_encoding(max_len, embedding_dim)
        self.embedding_dropout = tf.keras.layers.Dropout(rate)
        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, ffl_units, rate)
                           for _ in range(num_layers)]

    def call(self, x, training=True):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        x += self.pos_enc[:, :seq_len, :]
        x = self.embedding_dropout(x, training=training)
        for ith in range(self.num_layers):
            x = self.enc_layers[ith](x, training=training)
        return x


class EncoderModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim,
                 max_len, num_heads, num_layers,
                 ffl_units, rate=0.1):
        super(EncoderModel, self).__init__()

        ''' Note: (embedding_dim % num_heads) should be an integer!!!
            1. embedding_dim = mha_units, since point wise addition.
            2. ffl_out_units = embedding_dim as well, same reason.
            3. ffl has 2 layers, the 1st layer is defined by ffl_units,
               the 2nd layer units = mha_units = embedding_dim.
               Note: you can customize it so all units can be differ.
        '''
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_enc = positional_encoding(max_len, embedding_dim)
        self.embedding_dropout = tf.keras.layers.Dropout(rate)
        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, ffl_units, rate)
                           for _ in range(num_layers)]

    def call(self, x, training=True):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        x += self.pos_enc[:, :seq_len, :]
        x = self.embedding_dropout(x, training=training)
        for ith in range(self.num_layers):
            x = self.enc_layers[ith](x, training=training)
        return x