import os
import sys
import json
import codecs
import pandas as pd
from tqdm import tqdm

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))


def get_data_dailydialogue(file_dir):

    def data_process(data):

        def length_requirement(text):
            if len(text.split()) > 8 and len(text.split()) < 96:
                return True
            return False

        samples = []
        for d in tqdm(data, desc='  - (data processing) -  '):
            content = d['content']
            for i, utterance in enumerate(content[:-1]):
                sample = ['Forward', utterance['text'], content[i + 1]['text']]
                if length_requirement(sample[1]) and length_requirement(sample[2]):
                    samples.append(sample)
        
        return samples

    train_data = data_process(json_load(os.path.join(file_dir, 'train.json')))
    dev_data = data_process(json_load(os.path.join(file_dir, 'validation.json')))
    test_data = data_process(json_load(os.path.join(file_dir, 'test.json')))

    return train_data, dev_data, test_data


if __name__ == '__main__':
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    train_data, dev_data, test_data = get_data_dailydialogue(input_dir)

    df_train = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False, sep=',')

    df_dev = pd.DataFrame(dev_data, columns=["prefix", "input_text", "target_text"])
    df_dev.to_csv(os.path.join(output_dir, 'dev.csv'), index=False, sep=',')

    df_test = pd.DataFrame(test_data, columns=["prefix", "input_text", "target_text"])
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False, sep=',')
