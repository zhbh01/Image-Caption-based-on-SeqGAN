import tensorflow as tf

from processor.image_processor import ImageProcessor
from processor.text_processor import TextProcessor


class Discriminator:
    lstm_size = 128

    def __init__(self, batch_size, dictionary_size):
        self.batch_size = batch_size
        self.dictionary_size = dictionary_size

    def input_node(self):
        with tf.variable_scope("input_node"):
            self.embedding = tf.placeholder(tf.float32, [self.dictionary_size, TextProcessor.WORD_VEC_DIM], "embedding")
            self.fake_input = tf.placeholder(tf.int32, [self.batch_size / 2, None], "fake-input")
            self.label_input = tf.placeholder(tf.int32, [self.batch_size / 2, None], "label-input")
            self.image = tf.placeholder(tf.float32,
                                        [self.batch_size / 2, ImageProcessor.IMG_MODEL_OUTPUT_DIM], "image-input")
            self.sentence_input = tf.concat([self.fake_input, self.label_input], axis=0)
            self.image_input = tf.tile(self.image, [2, 1])

    def lstm_encoder(self):
        with tf.variable_scope("lstm_encoder"):
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(Discriminator.lstm_size)
            self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

            sequence_length, _ = self._sequence_length_op(self.sentence_input)

            lstm_input = tf.nn.embedding_lookup(self.embedding, self.sentence_input)

            _, self.state = tf.nn.dynamic_rnn(self.lstm_cell,
                                              lstm_input,
                                              initial_state=self.state,
                                              sequence_length=sequence_length)

    def dense_layer(self):
        with tf.variable_scope("output-net"):
            fc1_input = tf.concat([self.state[0], self.image_input], axis=1)
            fc1_input_dim = Discriminator.lstm_size + ImageProcessor.IMG_MODEL_OUTPUT_DIM
            fc1_output = self.fc_layer(fc1_input, fc1_input_dim, 100, "fc-1")
            fc2_output = self.fc_layer(fc1_output, 100, 1, "fc-2")
            self.output = fc2_output

    def fc_layer(self, input, input_dim, output_dim, scope_name):
        with tf.variable_scope(scope_name):
            weight = tf.get_variable("weight", [input_dim, output_dim], tf.float32, tf.truncated_normal_initializer)
            bias = tf.get_variable("bias", [output_dim], tf.float32, tf.truncated_normal_initializer)
            output = tf.matmul(input, weight) + bias
        return tf.sigmoid(output)

    def loss_op(self):
        with tf.variable_scope("loss_op"):
            self.D_fake, self.D_real = tf.split(self.output, 2, axis=0)
            self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake))
            tf.summary.scalar("discriminator_ad_batch_loss", self.D_loss)

    def _sequence_length_op(self, sequence):
        with tf.variable_scope("sequence_length_op"):
            mask = tf.sign(sequence)
            length = tf.reduce_sum(mask, 1)
            length = tf.cast(length, tf.int32)
        return length, mask

    def build_discriminator(self):
        with tf.variable_scope("Discriminator"):
            self.input_node()
            self.lstm_encoder()
            self.dense_layer()
            self.loss_op()
