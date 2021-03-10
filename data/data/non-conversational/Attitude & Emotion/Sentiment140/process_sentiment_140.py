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
        total = 0
        for row in tqdm(data):
            processed_row = remove_unnecessary_parts(row[5])
            if processed_row == "": continue
            output.append({"text": processed_row, "topic": "Attitude & Emotion"})
            total += 1
            # if total == 100000: break
        
    return output, total

def remove_unnecessary_parts(line):
    line = line.replace("&lt;3", "")
    line = line.replace("&lt; 3", "")
    line = line.replace("@", "")
    pieces = line.split()
    new_str = ""
    for piece in pieces:
        if piece[:4] == "http": continue    #removes http addresses
        piece = remove_hashtag(piece)       #removes hashtags
        new_str += piece + " "
    new_str = ' '.join(new_str.split())
    return new_str

def remove_hashtag(piece):
    hashtag_pos = piece.find("#")
    if hashtag_pos == -1: return piece
    last_pos = hashtag_pos
    for index in range(hashtag_pos + 1, len(piece)):
        if not (piece[index].isalpha() or piece[index].isdigit()): 
            last_pos = index - 1
            break
        else:
            if index == len(piece) - 1:
                last_pos = index
    
    piece = piece[:hashtag_pos] + piece[last_pos + 1 :]
    
    return piece
    

if __name__ == '__main__':
    input_file = "training.1600000.processed.noemoticon.csv"    
    output_data = process_data(input_file)
    output_file = "sentiment_140 - " + str(output_data[1]) + ".json"

    json_dump(output_data[0], output_file)
    
    