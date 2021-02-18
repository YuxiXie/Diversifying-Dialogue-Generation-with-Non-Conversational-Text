import csv
import sys
from tqdm import tqdm

from data.utils import json_dump

def get_samples(filename):
    samples, avg_length = [], 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        header = next(data)

        for row in tqdm(data):
            ans, qu = row[2], row[3]
            ans_text = [{'content': a.strip(), 'label': 'declarative'} for a in ans.split('\n\n') if len(a.split()) > 8 and len(a.split()) < 96]
            samples += ans_text + [{'content': qu.strip(), 'label': 'question'}]
    
    return samples


if __name__ == '__main__':
    input_file1, input_file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    samples = get_samples(input_file1) + get_samples(input_file2)

    json_dump(samples, output_file)
