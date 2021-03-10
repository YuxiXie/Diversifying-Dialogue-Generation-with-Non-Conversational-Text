
import random
from tqdm import tqdm
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def combine_jsons(filenames, amounts):
    
    output = []
    total = 0
    
    for filename, max_amount in tqdm(zip(filenames, amounts)):
        input_file = json_load(filename)
        random.shuffle(input_file)  #randomly select the input text
        amount = 0
        for item in tqdm(input_file):
            output.append(item)
            amount += 1
            if amount == max_amount: break
        total += amount
            
    return output, total

if __name__ == '__main__':
    input_files = ["medical_data - 4993.json", 
                    # "sentiment_140 - 100000.json", 
                    "sentiment_140 - 1048523.json", 
                   "stanford_sentiment_treebank - 11855.json", 
                   "inaugural - 5153.json", 
                   "trump_twitter_2016 - 29973.json", 
                   "trump_twitter_2017 - 30279.json"]
    input_amounts = [4993, #-1 means all
                    67999, 
                    11855, 
                    5153, 
                    5000, 
                    5000]
    input_amounts = [-1, #-1 means all
                    -1, 
                    -1, 
                    -1, 
                    -1, 
                    -1]
    output_data = combine_jsons(input_files, input_amounts)
    output_file = "combined_jsons - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)


