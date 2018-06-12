import json
import os
import pickle
import random

import config
from dataloader.DataLoader import DataLoader
from processor.text_processor import MSCOCOTextProcessor


class MSCOCODataLoader(DataLoader):
    MSCOCO_text_processor = MSCOCOTextProcessor()
    annotation_dir = os.path.join(config.DATA_DIR, "MSCOCO", "annotation")
    image_feature_dir = os.path.join(config.DATA_DIR, "MSCOCO", "image_feature")

    def __init__(self):
        self.cache_path = os.path.join(config.DATA_DIR, "MSCOCO", "cache", "cache.pkl")
        self.caption_file_names = {"eval": "captions_val2014.json", "train": "captions_train2014.json"}
        self.image_feature_names = {
            "eval": "eval_img_feature.pkl",
            "test": "test_img_feature.pkl",
            "train": "train_img_feature.pkl"
        }
        DataLoader.__init__(self, MSCOCODataLoader.MSCOCO_text_processor, self.cache_path)

    def load_records(self):
        """
        :return: three separated set with the format as (image name, image vector, sentence word index)
        """

        # load the annotation file
        caption_info = dict()
        for split, caption_file in self.caption_file_names.items():
            with open(os.path.join(MSCOCODataLoader.annotation_dir, caption_file), "r") as f:
                caption_info[split] = json.load(f)
        # construct mapping between image id and file name
        image_id2name = dict()
        for split, info in caption_info.items():
            for image in info["images"]:
                image_id2name[image["id"]] = image["file_name"]
        # load the image vectors
        feature_info = dict()
        for split, feature_file in self.image_feature_names.items():
            with open(os.path.join(MSCOCODataLoader.image_feature_dir, feature_file), "rb") as f:
                feature_info[split] = pickle.load(f)
        # construct return tuple
        res = dict()
        for split, info in caption_info.items():
            tuples = list()
            for annotation in info["annotations"]:
                image_file_name = image_id2name[annotation["image_id"]]
                image_vec = feature_info[split][image_file_name]
                caption_word_index = list()
                for word in annotation["caption"].split():
                    if word in self.dictionary:
                        caption_word_index.append(self.dictionary.index(word))
                res_tuple = (image_file_name, image_vec, caption_word_index)
                tuples.append(res_tuple)
            random.shuffle(tuples)
            res[split] = tuples
        # for the test set
        test_tuple = list()
        for file_name, image_vec in feature_info["test"].items():
            test_tuple.append((file_name, None, image_vec))

        # store as the cache file
        cache = {
            "train": res["train"],
            "test": test_tuple,
            "eval": res["eval"]
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache, f)

        return res["train"], test_tuple, res["eval"]


if __name__ == '__main__':
    loader = MSCOCODataLoader()
    img_name, img_vec, sentence_word_index = loader.next_batch("train", 50)
    pass
