import re
import sys
import csv
from tqdm import tqdm
from data.utils import json_dump


def clean(sent):
    # should we make it uncased?
    # sent = sent.lower()
    s = re.sub("<s>", r"", sent)
    s = re.sub(r'[.]+[\n]+[,]', ".\n", s)
    s = s.split()
    return ' '.join(s), len(s)


def get_samples(filename, word_threshold=100):
    samples = []

    with open(filename, 'r', encoding="utf8") as f:
        data = csv.reader(f)
        # overview,headline,text,sectionLabel,title
        header = next(data)

        for i, row in enumerate(tqdm(data)):
            # currently using overview (not sure which is the best choice)
            # some seems to have repetition
            content = row[0].strip()
            content, length = clean(content)
            if length < word_threshold:
                sample = {'emotion': 'neutral', 'content': content}
                samples.append(sample)

    return samples


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    json_dump(get_samples(input_file), output_file)
