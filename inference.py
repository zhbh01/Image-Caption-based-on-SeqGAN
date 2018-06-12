import json
import os

import cv2
import tensorflow as tf
from tqdm import tqdm

from generator.beam_searcher import CaptionGenerator
from generator.generator import Generator
from generator.inference_wrapper import InferenceWrapper
from processor.image_processor import ImageProcessor
from processor.text_processor import Flickr8kTextProcessor
from processor.text_processor import MSCOCOTextProcessor
from training import AdversarialTrainer


class GenInferencer:
    checkpoint_dir = os.path.dirname(AdversarialTrainer.checkpoint_path)
    checkpoint_file_name = "gan_model.ckpt-388200"

    def __init__(self, batch_size, text_processor):
        self.batch_size = batch_size
        # initialize variables
        self.text_processor = text_processor

        self.image_processor = ImageProcessor()
        self.embedding = self.text_processor.embedding
        self.dictionary = self.text_processor.dictionary

        self.generator = Generator(self.batch_size, len(self.dictionary))
        self.generator.build_generator()

        infer_wrapper = InferenceWrapper(self.generator, self.embedding)
        self.caption_generator = CaptionGenerator(infer_wrapper, self.dictionary, beam_size=1,
                                                  length_normalization_factor=0)
        # restore model from checkpoint
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, os.path.join(GenInferencer.checkpoint_dir, GenInferencer.checkpoint_file_name))

    def inference(self, image_path):
        img_data = cv2.imread(image_path)
        img_feature = self.image_processor.feature_extraction(img_data, self.sess)
        captions = self.caption_generator.beam_search(self.sess, img_feature)
        out_buffer = []
        for caption in captions:
            symbol = [self.dictionary[word] for word in caption.sentence]
            out_buffer.append(symbol)
        return out_buffer

    @staticmethod
    def remove_special_token(sentence):
        tokens = ["<sos>", "<eos>"]
        for token in tokens:
            if token in sentence:
                sentence.remove(token)

        return sentence


class MSCOCOInferencer(GenInferencer):

    def __init__(self, batch_size):
        text_processor = MSCOCOTextProcessor()
        GenInferencer.__init__(self, batch_size, text_processor)

    def inference_mscoco_val(self):
        # read meta information
        meta_data_buf = list()
        image_dir = "data/MSCOCO/image/val2014"
        with open("data/MSCOCO/annotation/captions_val2014.json", "r", encoding="UTF") as f:
            val_info = json.load(f)
        for image_info in val_info["images"]:
            image_name = image_info["file_name"]
            image_id = image_info["id"]
            image_path = os.path.join(image_dir, image_name)
            meta_data_buf.append((image_id, image_path))

        # do the inference for every image
        predictions = list()
        for image_id, image_path in tqdm(meta_data_buf):
            sentence = self.inference(image_path)[0]
            res = {
                "image_id": image_id,
                "caption": GenInferencer.remove_special_token(sentence)
            }
            predictions.append(res)

        # store the result
        with open("evaluation/prediction.json", "w", encoding="UTF") as f:
            json.dump(predictions, f, ensure_ascii=False)


class Flickr8kInferencer(GenInferencer):

    def __init__(self, batch_size):
        text_processor = Flickr8kTextProcessor()
        GenInferencer.__init__(self, batch_size, text_processor)


if __name__ == '__main__':
    inferencer = MSCOCOInferencer(1)
    sentences = inferencer.inference("data/MSCOCO/image/test2014/COCO_test2014_000000001194.jpg")
    for sentence in sentences:
        print(" ".join(sentence))
