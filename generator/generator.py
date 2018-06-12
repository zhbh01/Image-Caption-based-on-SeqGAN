import tensorflow as tf

from processor.image_processor import ImageProcessor
from processor.text_processor import TextProcessor


class Generator:
    lstm_size = 512

    def __init__(self, batch_size, dictionary_size):
        self.batch_size = batch_size
        self.dictionary_size = dictionary_size

    def input_node(self):
        with tf.variable_scope("input-node"):
            self.image_input = tf.placeholder(tf.float32, [self.batch_size, ImageProcessor.IMG_MODEL_OUTPUT_DIM],
                                              "image-input")
            self.sentence_input = tf.placeholder(tf.int32, [self.batch_size, None], "sentence-input")
            self.label_input = tf.placeholder(tf.int32, [self.batch_size, None], "label-input")
            self.embedding = tf.placeholder(tf.float32, [self.dictionary_size, TextProcessor.WORD_VEC_DIM],
                                            "embedding-input")

    def lstm_decoder(self):
        with tf.variable_scope("rnn-cell"):
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(Generator.lstm_size)
            self.zero_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

    def sample_op(self):
        with tf.variable_scope("sample-op"):
            samples = tf.multinomial(tf.log(tf.reshape(self.probabilities, [self.batch_size, self.dictionary_size])), 1)
            samples = tf.cast(tf.reshape(samples, [self.batch_size]), tf.int32)
            self.samples = samples

    def loss_op(self):
        with tf.variable_scope("decoder"):
            # initialize the model with the image feature
            _, self.initial_state = self.lstm_cell.call(self.image_input, self.zero_state)
            self.sentence_length, self.mask = self._sequence_length_op(self.sentence_input)
            # construct the input of the lstm
            sentence_input = tf.nn.embedding_lookup(self.embedding,
                                                    self.sentence_input)  # batch_size x time_step x word dim
            # roll out the LSTM cell
            self.outputs, _ = tf.nn.dynamic_rnn(self.lstm_cell,
                                                sentence_input,
                                                initial_state=self.initial_state,
                                                sequence_length=self.sentence_length,
                                                dtype=tf.float32)
            # project the output into the vocabulary space
            self.outputs = tf.layers.dense(self.outputs, self.dictionary_size, use_bias=True, activation=None)

        with tf.variable_scope("mle-loss-op"):
            logits = self.outputs
            labels = self.label_input
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            self.pre_loss = self._mask_cost_op(cross_entropy, self.mask)
            tf.summary.scalar("generator_train_batch_loss", self.pre_loss)

    def policy_gradient_loss(self):
        with tf.variable_scope("policy-gradient"):
            self.sample_input = tf.placeholder(tf.int32, [self.batch_size, None], "sample-input")
            self.rewards = tf.placeholder(tf.float32, [self.batch_size, None], "reward-input")
            self.ad_loss = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.sample_input, [-1])), self.dictionary_size, 1.0,
                               0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.outputs, [-1, self.dictionary_size]), 1e-20, 1.0)
                    ), 1) * tf.reshape(self.rewards, [-1])
            )

    def inference_op(self):
        with tf.variable_scope("inference-input"):
            # build inference step op
            self.inference_word_input = tf.placeholder(tf.int32, [self.batch_size], name="inference-step-word-input")
            self.inference_image_input = tf.placeholder(tf.float32,
                                                        [self.batch_size, ImageProcessor.IMG_MODEL_OUTPUT_DIM],
                                                        name="inference-step-image-input")
            word_tensor = tf.nn.embedding_lookup(self.embedding, self.inference_word_input)
            word_tensor = tf.reshape(word_tensor, [self.batch_size, 1, TextProcessor.WORD_VEC_DIM])

        with tf.variable_scope("decoder", reuse=True):
            _, self.inference_initial_state = self.lstm_cell.call(self.inference_image_input, self.zero_state)
            output, self.next_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                        word_tensor,
                                                        initial_state=self.inference_initial_state,
                                                        dtype=tf.float32)
            output = tf.layers.dense(output, self.dictionary_size, activation=None, use_bias=True, reuse=True)
            self.probabilities = tf.nn.softmax(output)

    def _sequence_length_op(self, sequence):
        with tf.variable_scope("sequence_length_op"):
            mask = tf.sign(sequence)
            length = tf.reduce_sum(mask, 1)
            length = tf.cast(length, tf.int32)
        return length, mask

    def _mask_cost_op(self, cross_entropy, mask):
        with tf.variable_scope("mask_cost_op"):
            mask = tf.cast(mask, tf.float32)
            cross_entropy *= mask
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, 1)
            cross_entropy /= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(cross_entropy)

    def build_generator(self):
        with tf.variable_scope("Generator"):
            self.input_node()
            self.lstm_decoder()
            self.loss_op()
            self.inference_op()
            self.sample_op()
            self.policy_gradient_loss()
