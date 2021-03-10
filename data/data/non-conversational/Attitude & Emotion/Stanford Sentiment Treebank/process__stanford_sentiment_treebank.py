
from tqdm import tqdm

import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def process_data(filename):    
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
        output = []
        
        total = 0
        for text in tqdm(data):
            tab_position = text.find("\t")
            text = text[tab_position + 1 :]
            text = text.strip()
            if text == "": continue
            output.append({"text": text, "topic": "Attitude & Emotion"})
            total += 1

    return output, total


if __name__ == '__main__':
    input_file = "datasetSentences.txt"
    output_data = process_data(input_file)
    output_file = "stanford_sentiment_treebank - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    