
from tqdm import tqdm
import os
import re

import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def process_data():
    output = []
    total = 0
    for filename in os.listdir(os.getcwd()):
        if filename[-3:] == "txt":
            with open(os.path.join(os.getcwd(), filename), 'r', encoding='utf-8') as f: # open in readonly mode
                for line in tqdm(f.read().strip().split('\n')):
                    line = line.strip()
                    if len(line) == 0: continue
                    for sentence in re.split("\.|!|\?", line):
                        sentence = sentence.strip()
                        if len(sentence) == 0: continue
                        output.append({"text":sentence, "topic":"Politics"})
                        total += 1
                    
        
    return output, total


if __name__ == '__main__':
    
    output_data = process_data()
    output_file = "inaugural - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    
    