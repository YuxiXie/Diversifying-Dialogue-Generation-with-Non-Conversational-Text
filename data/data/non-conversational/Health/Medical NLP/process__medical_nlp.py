import csv
from tqdm import tqdm

import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def process_data(filename):
    output = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        next(data)
        total = 0
        for row in tqdm(data):
            processed_text = ' '.join(row[1].split())
            if processed_text == "": continue
            output.append({"text": processed_text, "topic": "Health"})
            total += 1
            
    return output, total


if __name__ == '__main__':
    input_file = "mtsamples-utf8.csv"
    output_data = process_data(input_file)
    output_file = "medical_data - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    