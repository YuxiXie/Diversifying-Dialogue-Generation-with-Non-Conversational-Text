import csv
from tqdm import tqdm

from utils import json_dump

def process_data(filename):
    output = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        total = 0
        for row in tqdm(data):
            output.append({"text": row[5], "topic": "Attitude & Emotion"})
            total += 1
            if total == 100000: break
        
    return output, total


if __name__ == '__main__':
    input_file = "training.1600000.processed.noemoticon.csv"    
    output_data = process_data(input_file)
    output_file = "sentiment_140 - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    