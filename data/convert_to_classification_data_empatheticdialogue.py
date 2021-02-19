import sys
from tqdm import tqdm
from data.utils import json_load, json_dump

'''
cmd args

data\empatheticdialogue\train_json.json data\empatheticdialogue\train_json_classification.json
data\empatheticdialogue\test_json.json data\empatheticdialogue\test_json_classification.json
data\empatheticdialogue\valid_json.json data\empatheticdialogue\valid_json_classification.json
'''

def convert_samples(filename):
    samples = json_load(filename)
    output = []
    for sample in tqdm(samples):
        topic = sample['context']
        content = sample['content']
        for turn in content:
            output.append({'text': turn, 'topic': topic})

    return output


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    json_dump(convert_samples(input_file), output_file)
