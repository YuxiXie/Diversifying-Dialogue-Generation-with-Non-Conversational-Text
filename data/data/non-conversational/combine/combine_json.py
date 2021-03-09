import sys
import random
from tqdm import tqdm
from utils import json_load, json_dump

'''
cmd args

data\dailydialog\dialogue_json.json data\dailydialog\dialogue_json_classified.json
'''


def combine_jsons(filenames):
    
    output = []
    total = 0
    
    for filename in filenames:
        input_file = json_load(filename)
        for item in tqdm(input_file):
            output.append(item)
            total += 1
    
    print("total = " + str(total))
            
    return output, total

if __name__ == '__main__':
    input_files = {"attitude&emotion - 11856.json", 
                   "medical_data - 5000.json", 
                   "trump_twitter_2016 - 30079.json", 
                   "trump_twitter_2017 - 30386.json"}
    output_data = combine_jsons(input_files)
    output_file = "combined_jsons - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)


