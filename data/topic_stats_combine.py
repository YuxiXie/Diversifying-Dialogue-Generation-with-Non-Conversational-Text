import sys
from tqdm import tqdm
from data.utils import json_load, json_dump

'''
cmd args

dailydialog
data\dailydialog\dialogue_json_classified.json data\dailydialog\dialogue_json_classified_stats.json

empatheticdialogue
3 data\empatheticdialogue\train_json_classification.json data\empatheticdialogue\test_json_classification.json data\empatheticdialogue\valid_json_classification.json data\empatheticdialogue\json_classification_stats_combined.json
'''

stats_dict = {}

def convert_samples(filenames):
    for filename in filenames:
        samples = json_load(filename)
        output = []
        for sample in tqdm(samples):
            topic = sample['topic']
            if(topic not in stats_dict):
                stats_dict[topic] = 1
            else:
                stats_dict[topic] += 1
    
    stats_dict_sorted = {k: v for k, v in sorted(stats_dict.items(), key=lambda item: item[1], reverse=True)}
    for topic in stats_dict_sorted:
        output.append({"topic": topic, "amount": stats_dict_sorted[topic]})
    
    return output


if __name__ == '__main__':
    input_num = int(sys.argv[1])
    filenames = []
    for i in range(1, input_num + 1):
        print(sys.argv[i + 1])
        filenames.append(sys.argv[i + 1])
    output_file = sys.argv[input_num + 2]

    json_dump(convert_samples(filenames), output_file)
