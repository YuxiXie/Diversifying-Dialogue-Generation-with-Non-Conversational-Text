import sys
from tqdm import tqdm
import csv
from data.utils import json_dump, json_load

sentiment_dict = {'0': 'negative', '2': 'neutral', '4': 'positive'}


def filter_content(content):
    words = content.split()
    # remove @xx and #xx
    words = filter(lambda x: x[0] != '#', words)
    cleaned_words = list(filter(lambda x: x[0] != '@', words))
    length = len(cleaned_words)
    content = " ".join(cleaned_words)
    return content, length


def get_samples(filename, word_threshold=100):
    samples = []

    with open(filename, 'r', encoding='latin-1') as f:
        data = csv.reader(f)

        for i, row in enumerate(tqdm(data)):
            sentiment = sentiment_dict[row[0]]
            content = row[5].strip()
            content, length = filter_content(content)
            if length < word_threshold:
                sample = {'emotion': sentiment, 'content': content}
                samples.append(sample)

    return samples


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    json_dump(get_samples(input_file), output_file)

