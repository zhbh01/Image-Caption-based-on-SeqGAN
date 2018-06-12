import tensorflow as tf
import os
import cv2
import pickle
import config
from tqdm import tqdm
from tensorflow.python.platform import gfile


class ImageProcessor:
    IMG_MODEL_FILE = os.path.join(config.DATA_DIR, "models", "tensorflow_inception_graph.pb")
    IMG_MODEL_INPUT_TENSOR_NAME = "DecodeJpeg:0"
    IMG_MODEL_OUTPUT_TENSOR_NAME = "softmax:0"
    IMG_MODEL_OUTPUT_DIM = 1008

    def __init__(self):
        # load the computation graph of the pre-trained model
        with gfile.FastGFile(ImageProcessor.IMG_MODEL_FILE, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.input_node, self.output_node = tf.import_graph_def(graph_def,
                                                                return_elements=[
                                                                    ImageProcessor.IMG_MODEL_INPUT_TENSOR_NAME,
                                                                    ImageProcessor.IMG_MODEL_OUTPUT_TENSOR_NAME])

    def feature_extraction(self, img_data, sess):
        bottleneck_value = sess.run(self.output_node, feed_dict={self.input_node: img_data})
        return bottleneck_value

    def extract(self):
        pass


class Filckr8kImageProcessor(ImageProcessor):
    data_dir = os.path.join(config.DATA_DIR, "Flickr8k")
    image_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    text_dir = os.path.join(data_dir, "Flickr8k_text")
    SPLIT_FILES = {"test": os.path.join(text_dir, "Flickr_8k.testImages.txt"),
                   "train": os.path.join(text_dir, "Flickr_8k.trainImages.txt"),
                   "eval": os.path.join(text_dir, "Flickr_8k.devImages.txt")
                   }

    def __init__(self):
        ImageProcessor.__init__(self)

    def extract(self):
        sess = tf.Session()
        # obtain the split
        data_set = {}
        for split_name, split_file_path in Filckr8kImageProcessor.SPLIT_FILES.items():
            with open(split_file_path, "r") as f:
                temp_set = f.readlines()
                data_set[split_name] = temp_set

        for name, set in data_set.items():
            features = {}
            print("Processing images in {} set.".format(name))
            for img in tqdm(set):
                img = img.strip()
                img_path = os.path.join(Filckr8kImageProcessor.image_dir, img)
                img_data = cv2.imread(img_path)
                features[img] = ImageProcessor.feature_extraction(self, img_data, sess)

                stored_file = os.path.join(Filckr8kImageProcessor.data_dir, "image_feature",
                                           name + "_img_feature.pkl")
                with open(stored_file, "wb") as f:
                    pickle.dump(features, f)


class MSCOCOImageProcessor(ImageProcessor):
    data_dir = os.path.join(config.DATA_DIR, "MSCOCO")
    image_dir = os.path.join(data_dir, "image")
    image_split_dirs = {
        "test": os.path.join(image_dir, "test2014"),
        "train": os.path.join(image_dir, "train2014"),
        "eval": os.path.join(image_dir, "val2014")
    }

    def __init__(self):
        ImageProcessor.__init__(self)

    def extract(self):
        sess = tf.Session()
        for split_name, split_dir in MSCOCOImageProcessor.image_split_dirs.items():
            image_names = os.listdir(split_dir)
            features = {}
            print("Processing images in {} set.".format(split_name))
            for name in tqdm(image_names):
                image_path = os.path.join(split_dir, name)
                img_data = cv2.imread(image_path)
                features[name] = ImageProcessor.feature_extraction(self, img_data, sess)

            stored_file = os.path.join(MSCOCOImageProcessor.data_dir, "image_feature",
                                       split_name + "_img_feature.pkl")
            with open(stored_file, "wb") as f:
                pickle.dump(features, f)
