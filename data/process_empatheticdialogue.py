import csv
import sys
from tqdm import tqdm

from data.utils import json_dump

'''
cmd args

data\empatheticdialogue\train.csv data\empatheticdialogue\train_json.json
data\empatheticdialogue\test.csv data\empatheticdialogue\test_json.json
data\empatheticdialogue\valid.csv data\empatheticdialogue\valid_json.json
'''

def get_samples(filename):
    samples, avg_length = {}, 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        header = next(data)

        for row in tqdm(data):
            conv_id = row[0]
            if conv_id not in samples:
                samples[conv_id] = {'context': row[2], 'length': 0, 'prompt': row[3], 'content': []}
            samples[conv_id]['content'].append([row[1], row[4], row[5]])
    
    data = []
    for k, v in tqdm(samples.items()):
        v['content'].sort(key=lambda x: x[0])
        raw_id, is_available = v['content'][0][1], True
        for x in v['content'][1:]:
            if x[1] == raw_id:
                is_available = False
                break
            raw_id = x[1]
        
        if is_available:
            v['content'] = [x[2] for x in v['content']]
            v['length'] = len(v['content'])
            if v['length'] > 1:
                data.append(v)
    
    samples = data
    samples.sort(key=lambda x: x['length'])
    return samples


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    json_dump(get_samples(input_file), output_file)