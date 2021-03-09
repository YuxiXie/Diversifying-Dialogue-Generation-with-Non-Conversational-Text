import csv
import sys
from tqdm import tqdm
import pandas as pd

from utils import json_dump

def get_samples(filename):
    samples = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        header = next(data)
        ind = 1
        for row in tqdm(data):
            samples[ind] = {'date': row[0], 'time': row[1], 'tweet': row[2]}
            ind += 1
            
    return samples


if __name__ == '__main__':
    input_file = "2017_01_28 -Trump Tweets.csv"
    output_file = "trump_twitter.json"

    json_dump(get_samples(input_file), output_file)
    
    