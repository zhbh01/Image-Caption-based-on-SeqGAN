"""Model wrapper class for performing inference."""


class InferenceWrapper:
    """Model wrapper class for performing inference with a ShowAndTellModel."""

    def __init__(self, generator, word_embedding):
        # initialize variables

        self.generator = generator
        self.word_embedding = word_embedding

    def feed_image(self, sess, encoded_images):
        feed_dict = {
            self.generator.embedding: self.word_embedding,
            self.generator.inference_image_input: encoded_images,
        }
        fetches = self.generator.inference_initial_state
        initial_state = sess.run(fetches, feed_dict)

        return initial_state

    def inference_step(self, sess, input_feeds, state_feeds, encoded_image):
        batch_size = len(input_feeds)
        softmax_outputs = []
        state_outputs = []
        for i in range(batch_size):
            softmax_output, state_output = sess.run(
                fetches=[self.generator.probabilities, self.generator.next_state],
                feed_dict={
                    self.generator.inference_word_input: [input_feeds[i]],
                    self.generator.inference_image_input: encoded_image,
                    self.generator.inference_initial_state: state_feeds[i],
                    self.generator.embedding: self.word_embedding,
                })
            softmax_outputs.append(softmax_output)
            state_outputs.append(state_output)

        return softmax_outputs, state_outputs, None
