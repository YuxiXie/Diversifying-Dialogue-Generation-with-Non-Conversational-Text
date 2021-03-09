import csv
import sys
from tqdm import tqdm
import pandas as pd

from utils import json_dump

def process_data(filename):
    output = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        header = next(data)
        total = 1
        for row in tqdm(data):
            output.append({'text': row[2], 'topic': "Politics"})
            total += 1
        
    return output, total


if __name__ == '__main__':
    # input_file = "2016_12_05-TrumpTwitterAll.csv"
    input_file = "2017_01_28 -Trump Tweets.csv"    
    output_data = process_data(input_file)
    output_file = "trump_twitter_2017 - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    