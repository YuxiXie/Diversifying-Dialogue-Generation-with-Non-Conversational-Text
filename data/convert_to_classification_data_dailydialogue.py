import sys
import random
from tqdm import tqdm
from data.utils import json_load, json_dump

'''
cmd args

data\dailydialog\dialogue_json.json data\dailydialog\dialogue_json_classified.json
'''


def convert_samples_with_filter(filename, filter_list=[], max_len=100):
    samples = json_load(filename)
    output = []
    for sample in tqdm(samples):
        topic = sample['topic']
        if topic not in filter_list:
            continue
        elif topic == 'Relationship':
            topic = 'Attitude & Emotion'
        content = sample['content']
        text = []
        for turn in content:
            text.extend(turn['text'].split(' '))
        # limit number of tokens to 100
        text = ' '.join(text[:max_len])
        output.append({'text': text, 'topic': topic})

    return output


def convert_samples(filename):
    samples = json_load(filename)
    output = []
    output2 = []
    random.seed(9)
    for sample in tqdm(samples):
        topic = sample['topic']
        content = sample['content']
        for turn in content:
            if random.random() >= 0.2:
                output.append({'text': turn['text'], 'topic': topic})
            else:
                output2.append({'text': turn['text'], 'topic': topic})
                
    result = [output, output2]
    return result


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    filter_list = ['Politics', 'Attitude & Emotion', 'Relationship', 'Health']

    json_dump(convert_samples_with_filter(input_file, filter_list), output_file)
    # json_dump(convert_samples(input_file), output_file)


