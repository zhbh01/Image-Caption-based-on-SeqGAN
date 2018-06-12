import config
import gensim
import os
import pickle
import numpy as np
import json


class TextProcessor:
    WORD_VEC_DIM = 128

    def __init__(self):
        self._embedding = None
        self._dictionary = None

    def build_embedding(self):
        pass

    @property
    def embedding(self):
        return self._embedding

    @property
    def dictionary(self):
        return self._dictionary

    @staticmethod
    def build_op(sentences, embedding_path, dictionary_path):
        model = gensim.models.Word2Vec(sentences, size=TextProcessor.WORD_VEC_DIM, min_count=5)
        # adding special character
        embedding = model.wv.syn0
        dictionary = model.wv.index2word
        special_character = ["<pad>", "<sos>", "<unk>", "<eos>"]
        dictionary = special_character + dictionary
        print("The size of dictionary:{}".format(len(dictionary)))
        rand_vec = np.random.rand(len(special_character), TextProcessor.WORD_VEC_DIM) * 2 - 1
        embedding = np.append(6 * rand_vec, embedding, 0)
        with open(embedding_path, "wb") as f:
            pickle.dump(embedding, f)
        with open(dictionary_path, "wb") as f:
            pickle.dump(dictionary, f)


class Flickr8kTextProcessor(TextProcessor):
    word_embedding_dir = os.path.join(config.DATA_DIR, "Flickr8k", "word_embedding")
    embedding_file = os.path.join(word_embedding_dir, "embedding.pkl")
    dictionary_file = os.path.join(word_embedding_dir, "dictionary.pkl")
    caption_file = os.path.join(config.DATA_DIR, "Flickr8k", "Flickr8k_text", "Flickr8k.token.txt")

    def __init__(self):
        TextProcessor.__init__(self)

    def build_embedding(self):
        # prepare the training data
        sentences = []
        with open(Flickr8kTextProcessor.caption_file, "r") as f:
            captions = f.readlines()
            for caption in captions:
                temp = caption.split("\t")[1]
                sentences.append(temp.split())

        TextProcessor.build_op(sentences, Flickr8kTextProcessor.embedding_file, Flickr8kTextProcessor.dictionary_file)

    @property
    def dictionary(self):
        if self._dictionary is None:
            with open(Flickr8kTextProcessor.dictionary_file, "rb") as f:
                self._dictionary = pickle.load(f)
        return self._dictionary

    @property
    def embedding(self):
        if self._embedding is None:
            with open(Flickr8kTextProcessor.embedding_file, "rb") as f:
                self._embedding = pickle.load(f)
        return self._embedding


class MSCOCOTextProcessor(TextProcessor):
    word_embedding_dir = os.path.join(config.DATA_DIR, "MSCOCO", "word_embedding")
    embedding_file = os.path.join(word_embedding_dir, "embedding.pkl")
    dictionary_file = os.path.join(word_embedding_dir, "dictionary.pkl")
    train_caption_file = os.path.join(config.DATA_DIR, "MSCOCO", "annotation", "captions_train2014.json")
    eval_caption_file = os.path.join(config.DATA_DIR, "MSCOCO", "annotation", "captions_val2014.json")

    def __init__(self):
        TextProcessor.__init__(self)

    def build_embedding(self):
        # prepare the training data
        caption_file = (MSCOCOTextProcessor.train_caption_file, MSCOCOTextProcessor.eval_caption_file)
        sentences = []
        for file in caption_file:
            with open(file, "r") as f:
                annotation_info = json.load(f)
                temp = [caption["caption"].split() for caption in annotation_info["annotations"]]
                sentences.extend(temp)
        TextProcessor.build_op(sentences, MSCOCOTextProcessor.embedding_file, MSCOCOTextProcessor.dictionary_file)

    @property
    def dictionary(self):
        if self._dictionary is None:
            with open(MSCOCOTextProcessor.dictionary_file, "rb") as f:
                self._dictionary = pickle.load(f)
        return self._dictionary

    @property
    def embedding(self):
        if self._embedding is None:
            with open(MSCOCOTextProcessor.embedding_file, "rb") as f:
                self._embedding = pickle.load(f)
        return self._embedding
