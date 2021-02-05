import csv
import sys
from tqdm import tqdm
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

def get_samples(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        header = next(data)

    samples, avg_length = {}, 0
    for row in tqdm(data):
        conv_id = row[0]
        if conv_id not in samples:
            samples[conv_id] = {'context': row[2], 'length': 0, 'prompt': row[3], 'utterances': []}
        samples[conv_id]['utterances'].append([row[1], row[4], row[5], row[6]])
    
    for k, v in tqdm(samples.items()):
        v['utterances'].sort(key=lambda x: x[0])
        raw_id, is_available = v['utterances'][0][1], True
        for x in v['utterances']:
            if x[1] == raw_id:
                flag = False
                break
            raw_id = x[1]

with open(sys.argv[1]) as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        import ipdb; ipdb.set_trace()