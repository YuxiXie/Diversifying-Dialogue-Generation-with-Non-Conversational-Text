import csv
from tqdm import tqdm

from utils import json_dump

def process_data(filename):
    output = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        next(data)
        total = 0
        for row in tqdm(data):
            output.append({"text": row[1], "topic": "Health"})
            total += 1
            
    return output, total


if __name__ == '__main__':
    input_file = "mtsamples-utf8.csv"
    output_data = process_data(input_file)
    output_file = "medical_data - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    