import sys
from tqdm import tqdm
from data.utils import json_load, json_dump


def convert_samples(filename):
    samples = json_load(filename)
    output = []
    for sample in tqdm(samples):
        topic = sample['topic']
        content = sample['content']
        for turn in content:
            output.append({'text': turn['text'], 'topic': topic})

    return output


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    json_dump(convert_samples(input_file), output_file)
