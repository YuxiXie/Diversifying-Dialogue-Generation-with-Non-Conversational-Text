
import sys
from tqdm import tqdm
import pandas as pd

from utils import json_dump

def process_data(filename):    
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
        output = []
        
        total = 1
        for text in data:
            tab_position = text.find("\t")
            text = text[tab_position + 1 :]
            output.append({"text": text, "topic": "Attitude & Emotion"})
            total += 1

    return output, total


if __name__ == '__main__':
    input_file = "datasetSentences.txt"
    output_data = process_data(input_file)
    output_file = "attitude&emotion - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    