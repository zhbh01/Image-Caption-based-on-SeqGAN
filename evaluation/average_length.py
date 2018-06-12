import re
import json


def get_average_length(sentences):
    total_num = len(sentences)
    lengths = [len(sentence) for sentence in sentences]
    average = sum(lengths) / total_num
    return average


if __name__ == '__main__':
    path = r"C:\Users\zbh\Desktop\working_dir\毕设模型参数\eval_coco\eval_pre\captions_val2014_fakecap_results.json"

    with open(path, "r", encoding="UTF") as f:
        data = json.load(f)

    sentences = list()
    for sentence in data:
        caption = sentence["caption"].split()
        sentences.append(caption)

    average_len = get_average_length(sentences)
    print(average_len)
