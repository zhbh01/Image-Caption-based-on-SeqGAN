import os
import sys

import numpy as np
import tensorflow as tf

import config
from dataloader.MSCOCODataLoader import MSCOCODataLoader
from discriminator.discriminator import Discriminator
from generator.generator import Generator


class AdversarialTrainer:
    batch_size = 64
    gen_pre_train_epoch = 0
    dis_pre_train_epoch = 0
    adversarial_train_epoch = 1000
    log_dir = os.path.join(config.DATA_DIR, "log")
    checkpoint_path = os.path.join(config.DATA_DIR, "checkpoint", "gan_model.ckpt")
    checkpoint_file = os.path.join(os.path.dirname(checkpoint_path), "gan_model.ckpt-35100")

    def __init__(self):
        print("initialize the model")
        self.data_loader = MSCOCODataLoader()
        self.dictionary = self.data_loader.dictionary
        self.embedding = self.data_loader.embedding
        self.global_step = tf.Variable(0, False, name="global-step")
        self.gen_learning_rate = 0.0005
        self.dis_learning_rate = 0.0005
        self.generator = Generator(AdversarialTrainer.batch_size, len(self.dictionary))
        self.discriminator = Discriminator(2 * AdversarialTrainer.batch_size, len(self.dictionary))

    def train(self):
        print("build the computation graph")
        self.build_computation_graph()
        # get trainable collection
        self.generator_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        self.discriminator_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        # build the optimizer
        self.build_optimizer()
        # initialize the summary writer
        summary_writer = tf.summary.FileWriter(AdversarialTrainer.log_dir, graph=tf.get_default_graph())
        merged = tf.summary.merge_all()
        # initialize the session and saver
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        # begin the training process
        epoch_size = self.data_loader.get_data_length("train") // AdversarialTrainer.batch_size

        if os.path.exists(os.path.join(os.path.dirname(AdversarialTrainer.checkpoint_path), "checkpoint")):
            saver.restore(sess, AdversarialTrainer.checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())

        # pre-train generator
        print("Pre-train the generator")
        self.pre_train(sess, saver, epoch_size, AdversarialTrainer.gen_pre_train_epoch, summary_writer, merged,
                       self.gen_pre_train_op)
        # pre-train discriminator
        print("Pre-train the discriminator")
        self.pre_train(sess, saver, epoch_size, AdversarialTrainer.dis_pre_train_epoch, summary_writer, merged,
                       self.dis_train_op)

        # adversarial training
        print("Adversarial training")
        for _ in range(AdversarialTrainer.adversarial_train_epoch // 2):
            for i in range(1 * epoch_size):
                step, summary = self.update_generator(merged, sess)
                if step % epoch_size == 0:
                    self.save_checkpoint(epoch_size, saver, sess, step, summary, summary_writer)

            for i in range(1 * epoch_size):
                step, summary = self.update_discriminator(merged, sess)
                if step % epoch_size == 0:
                    self.save_checkpoint(epoch_size, saver, sess, step, summary, summary_writer)

    def save_checkpoint(self, epoch_size, saver, sess, step, summary, summary_writer):
        summary_writer.add_summary(summary, global_step=step)
        saver.save(sess, AdversarialTrainer.checkpoint_path, global_step=step)
        print("Global epoch {}".format(step / epoch_size))
        sys.stdout.flush()

    def update_discriminator(self, merged, sess):
        img_vec, label, samples, sentence_input = self.prepare_data(sess)
        feed_dict = {
            self.discriminator.image: img_vec,
            self.discriminator.label_input: label,
            self.discriminator.embedding: self.embedding,
            self.discriminator.fake_input: samples,
            self.generator.image_input: img_vec,
            self.generator.sentence_input: sentence_input,
            self.generator.embedding: self.embedding,
            self.generator.label_input: label
        }

        step, summary, _ = sess.run([self.global_step, merged, self.dis_train_op], feed_dict)
        return step, summary

    def update_generator(self, merged, sess):
        img_vec, label, samples, sentence_input = self.prepare_data(sess)

        # # test code
        def test(sentences):
            for sentence in sentences:
                symbol = [self.dictionary[word] for word in sentence]
                print(symbol)

        test(samples)
        # # end test

        # obtain reward from discriminator
        rewards = self.estimate_reward(samples, img_vec, sess, 5)
        # filtrate the reward for <pad>
        mask = np.sign(samples)
        rewards = mask * rewards

        feed_dict = {
            self.discriminator.image: img_vec,
            self.discriminator.label_input: label,
            self.discriminator.embedding: self.embedding,
            self.discriminator.fake_input: samples,
            self.generator.sentence_input: samples,
            self.generator.label_input: label,
            self.generator.image_input: img_vec,
            self.generator.embedding: self.embedding,
            self.generator.sample_input: samples,
            self.generator.rewards: rewards
        }
        step, summary, _ = sess.run([self.global_step, merged, self.gen_ad_train_op], feed_dict)
        return step, summary

    def prepare_data(self, sess):
        img_vec, label, sentence_input = self.get_batch_data("train")
        label_length = np.shape(label)[1]
        # obtain sample from generator
        samples = self.get_sample(img_vec, None, sess)
        sample_length = np.shape(samples)[1]
        # pad the sample and label to the same length
        if label_length > sample_length:
            samples = self.pad_sentence(samples, label_length)
        else:
            label = self.pad_sentence(label, sample_length)
            sentence_input = self.pad_sentence(sentence_input, sample_length)
        return img_vec, label, samples, sentence_input

    def estimate_reward(self, target_sentences, image_vec, sess, roll_out_num):
        reward = []
        shape = np.shape(target_sentences)
        length = shape[1]
        for i in range(roll_out_num):
            for index in range(length):
                fix_part = np.split(target_sentences, [index + 1, length], axis=1)[0]
                simulation_result = self.get_sample(image_vec, fix_part, sess)
                temp_reward = self.get_reward_of_full_sentence(image_vec, simulation_result, sess)

                if i == 0:
                    reward.append(temp_reward)
                else:
                    reward[index] += temp_reward

        reward = np.array(reward) / (1.0 * roll_out_num)  # time step x batch size
        reward = np.transpose(np.squeeze(reward), [1, 0])  # batch size x time step
        return reward

    def get_reward_of_full_sentence(self, img_vec, samples, sess):
        feed_dict = {
            self.discriminator.image: img_vec,
            self.discriminator.embedding: self.embedding,
            self.discriminator.fake_input: samples,
            # when we estimate reward, the label input is unused
            self.discriminator.label_input: np.zeros(np.shape(samples))
        }
        reward = sess.run(self.discriminator.D_fake, feed_dict)
        return reward

    def get_sample(self, img_vec, given_sentence_input, sess, max_length=30):
        """
        This method continue to sample sentence based on given_sentence_input until reach <eos> or max length
        :param img_vec: input image vector of shape [batch_size, image_vec_dim]
        :param given_sentence_input: a list contains sentences, if this is None, the model will inference from scratch
        :param sess:
        :param max_length: the maximum length of sampled sentence
        :return: sampled sentence [batch size, sentence_length]
        """

        sentences_buffer = list()  # sentence length x batch size
        sentences_buffer.append(np.ones(self.batch_size) * self.dictionary.index("<sos>"))
        if given_sentence_input is not None:
            sentences_buffer.extend(np.transpose(given_sentence_input, [1, 0]))

        # initialize the state
        initial_state = sess.run(self.generator.inference_initial_state, {
            self.generator.inference_image_input: img_vec,
            self.generator.embedding: self.embedding
        })

        state = initial_state
        for index, step in enumerate(sentences_buffer):
            feed_dict = {
                self.generator.inference_word_input: step,
                self.generator.inference_image_input: img_vec,
                self.generator.inference_initial_state: state,
                self.generator.embedding: self.embedding
            }
            state, next_token = sess.run([self.generator.next_state, self.generator.samples], feed_dict)
            if index == len(sentences_buffer) - 1:
                sentences_buffer.append(np.array(next_token))
                break

        # inference process
        while len(sentences_buffer) < max_length:
            feed_dict = {
                self.generator.inference_word_input: sentences_buffer[-1],
                self.generator.inference_image_input: img_vec,
                self.generator.inference_initial_state: state,
                self.generator.embedding: self.embedding
            }
            state, next_token = sess.run([self.generator.next_state, self.generator.samples], feed_dict)
            sentences_buffer.append(np.array(next_token))

        sentences_buffer = np.transpose(np.array(sentences_buffer, np.int32), [1, 0])
        # clean the samples
        clean_buffer = []
        for sentence in sentences_buffer:
            eos_index = np.where(sentence == self.dictionary.index("<eos>"))[0]
            if len(eos_index) > 0:
                clean_buffer.append(sentence[1:eos_index[0] + 1])
            else:
                clean_buffer.append(sentence[1:])

        sentences_buffer = self.pad_sentence(clean_buffer)
        return sentences_buffer

    def build_optimizer(self):
        with tf.variable_scope("optimizers"):
            with tf.variable_scope("gen-pre-train"):
                self.gen_pre_train_op = tf.train.AdamOptimizer(self.gen_learning_rate).minimize(
                    self.generator.pre_loss,
                    var_list=self.generator_parameters,
                    global_step=self.global_step)
            with tf.variable_scope("dis-train"):
                self.dis_train_op = tf.train.AdamOptimizer(self.dis_learning_rate).minimize(
                    self.discriminator.D_loss,
                    var_list=self.discriminator_parameters,
                    global_step=self.global_step)
            with tf.variable_scope("gen-ad-train"):
                self.gen_ad_train_op = tf.train.AdamOptimizer(self.gen_learning_rate).minimize(
                    self.generator.ad_loss,
                    var_list=self.generator_parameters,
                    global_step=self.global_step
                )

    def pre_train(self, sess, saver, epoch_size, epoch_num, summary_writer, merged_summery, optimizer):
        for _ in range(epoch_size * epoch_num):
            img_vec, label, samples, sentence_input = self.prepare_data(sess)

            feed_dict = {
                self.generator.image_input: img_vec,
                self.generator.label_input: label,
                self.generator.sentence_input: sentence_input,
                self.generator.embedding: self.embedding,
                self.discriminator.label_input: label,
                self.discriminator.fake_input: samples,
                self.discriminator.image: img_vec,
                self.discriminator.embedding: self.embedding
            }

            _, step, summary = sess.run([optimizer, self.global_step, merged_summery],
                                        feed_dict=feed_dict)
            if step % epoch_size == 0:
                self.save_checkpoint(epoch_size, saver, sess, step, summary, summary_writer)

    def build_computation_graph(self):
        self.generator.build_generator()
        self.discriminator.build_discriminator()

    def get_batch_data(self, split_name):
        img_name, img_vec, sentence_word_index = self.data_loader.next_batch(split_name,
                                                                             AdversarialTrainer.batch_size)
        sentence_input = self.pad_sentence([np.insert(sentence, 0, self.dictionary.index("<sos>")) for sentence in
                                            sentence_word_index])
        label = self.pad_sentence(
            [np.append(sentence, self.dictionary.index("<eos>")) for sentence in sentence_word_index])

        return img_vec, label, sentence_input

    def pad_sentence(self, sentences, length=None):
        pad_buf = []
        lengths = [len(sentence) for sentence in sentences]
        if length is None:
            max_length = max(lengths)
        else:
            max_length = length
        for sentence in sentences:
            pad_length = max_length - len(sentence)
            for _ in range(pad_length):
                sentence = np.append(sentence, self.dictionary.index("<pad>"))
            pad_buf.append(sentence)

        return np.array(pad_buf)


if __name__ == '__main__':
    trainer = AdversarialTrainer()
    trainer.train()
