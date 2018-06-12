import os
import pickle
import random
import re

import numpy as np

from dataloader.DataLoader import DataLoader
from processor.image_processor import Filckr8kImageProcessor
from processor.text_processor import Flickr8kTextProcessor


class Flickr8KDataLoader(DataLoader):
    Flickr8k_text_processor = Flickr8kTextProcessor()

    def __init__(self):
        self.cache_path = os.path.join(Filckr8kImageProcessor.data_dir, "cache", "cache.pkl")
        DataLoader.__init__(self, Flickr8KDataLoader.Flickr8k_text_processor, self.cache_path)

    def load_records(self):
        # load the img feature from pickle file
        features = {}
        img_names = {}

        for name in self.data_pointer.keys():
            feature_store_path = os.path.join(Filckr8kImageProcessor.data_dir, "image_feature",
                                              name + "_img_feature.pkl")
            with open(feature_store_path, "rb") as f:
                features[name] = pickle.load(f)
            img_names[name] = list(features[name].keys())

        img_sentence_pairs = []
        with open(Flickr8kTextProcessor.caption_file, "r") as f:
            pattern = re.compile(r"(.+)#\d\t(.*)")
            for line in f.readlines():
                matcher = re.match(pattern, line)
                img_sentence_pairs.append((matcher.group(1), matcher.group(2).split()))

        # load the training data into three different sets, the format (image name, image vector, sentence word index)
        train_set = []
        test_set = []
        eval_set = []
        for img_sentence in img_sentence_pairs:
            # obtain the sentence index
            img_name = img_sentence[0]
            sentence = img_sentence[1]
            sentence_word_index = []
            for word in sentence:
                if word in self.dictionary:
                    index = self.dictionary.index(word)
                    sentence_word_index.append(index)
            sentence_word_index = np.array(sentence_word_index)
            # allocate the record in correct set
            if img_name in img_names["train"]:
                train_set.append([img_name, features["train"][img_name], sentence_word_index])
            elif img_name in img_names["test"]:
                test_set.append([img_name, features["test"][img_name], sentence_word_index])
            elif img_name in img_names["eval"]:
                eval_set.append([img_name, features["eval"][img_name], sentence_word_index])
        random.shuffle(train_set)
        random.shuffle(test_set)
        random.shuffle(eval_set)

        # store as the cache file
        cache = {
            "train": train_set,
            "test": test_set,
            "eval": eval_set
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache, f)

        return train_set, test_set, eval_set


if __name__ == '__main__':
    loader = Flickr8KDataLoader()
    img_name, img_vec, sentence_word_index = loader.next_batch("train", 50)
    pass
