import os
import pickle
import numpy as np
from processor.image_processor import ImageProcessor


class DataLoader:
    def __init__(self, text_processor, cache_path=None):
        # this attribute is used to track the position of the current data
        self.data_pointer = {"train": 0,
                             "test": 0,
                             "eval": 0
                             }
        self.dictionary = text_processor.dictionary
        self.embedding = text_processor.embedding

        if not os.path.exists(cache_path):
            train_set, test_set, eval_set = self.load_records()
        else:
            # otherwise load from the cache
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            train_set = cache_data["train"]
            test_set = cache_data["test"]
            eval_set = cache_data["eval"]

        self.data_set = {
            "train": train_set,
            "test": test_set,
            "eval": eval_set
        }

    def load_records(self):
        return list()

    def next_batch(self, split_name, batch_size):
        img_vec = []
        sentence_word_index = []
        img_name = []
        data_set = self.data_set[split_name]
        current_position = self.data_pointer[split_name]

        if current_position + batch_size < len(data_set):
            new_position = current_position + batch_size
            img_name.extend([x[0] for x in data_set[current_position: new_position]])
            img_vec.extend([x[1] for x in data_set[current_position: new_position]])
            sentence_word_index.extend([x[2] for x in data_set[current_position: new_position]])

            self.data_pointer[split_name] += batch_size
        else:
            new_position = batch_size - (len(data_set) - current_position)
            img_name.extend([x[0] for x in data_set[current_position:]])
            img_name.extend([x[0] for x in data_set[0: new_position]])
            img_vec.extend([x[1] for x in data_set[current_position:]])
            img_vec.extend([x[1] for x in data_set[0: new_position]])
            sentence_word_index.extend([x[2] for x in data_set[current_position:]])
            sentence_word_index.extend([x[2] for x in data_set[0: new_position]])

            self.data_pointer[split_name] = new_position

        # flatten the data
        img_vec = np.array(img_vec).flatten()
        # reshape the data
        img_vec = np.reshape(img_vec, [batch_size, ImageProcessor.IMG_MODEL_OUTPUT_DIM])

        return img_name, img_vec, sentence_word_index

    def get_data_length(self, split_name):
        return len(self.data_set[split_name])
