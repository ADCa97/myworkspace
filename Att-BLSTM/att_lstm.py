from numpy.core.fromnumeric import shape
import tensorflow as tf
from attention import attention

class AttBLSTM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_size, l2_reg_lambda=0.0):
        # Placeholders
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        
