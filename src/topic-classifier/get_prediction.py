import sys
import math
import json
import codecs
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    WEIGHTS_NAME,
    AutoModelWithLMHead, 
    AutoModelForSequenceClassification
)


LABEL_DICT = {
    'Ordinary Life': 0, 'School Life': 1, 'Culture & Education': 2, 'Attitude & Emotion': 3, 
    'Relationship': 4, 'Tourism': 5, 'Health': 6, 'Work': 7, 'Politics': 8, 'Finance': 9,
}


json_load = lambda x : json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p : json.dump(d, codecs.open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


class SeqTextDataset(Dataset):
    def __init__(self, dataset):
        self.inputs = dataset[0]
        self.indexes = dataset[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {
            'inputs': torch.tensor(self.inputs[i], dtype=torch.long),
            'label': torch.tensor(0, dtype=torch.long),
            'indexes': self.indexes[i]
        }


def classify(model_dir, samples, device, tokenizer):
    config = AutoConfig.from_pretrained(model_dir)
    config.num_labels = 3
    config.token_vocab_size = 2
    model = AutoModelForSequenceClassification.from_config(config)
    model.classifier = nn.Linear(768, 3)
    # model.bert.embeddings.token_type_embeddings = nn.Linear(768, 2, bias=False)
    model.num_labels = 3

    state_dict = torch.load(model_dir + '/pytorch_model.bin')
    # import ipdb; ipdb.set_trace()
    model.load_state_dict(state_dict)
    # model.bert.embeddings.token_type_embeddings = nn.Linear(2, 768, bias=False)

    model.to(device)
    model.eval()

    inputs = tokenizer.batch_encode_plus(samples, add_special_tokens=True, max_length=512)
    data = [inputs['input_ids'], range(len(inputs['input_ids']))]
    dataset = SeqTextDataset(data)

    def collate(examples: List[Dict]):
        inputs, indexes = [], []
        for sample in examples:
            inputs.append(sample['inputs'])
            indexes.append(sample['indexes'])
        indexes = torch.LongTensor(indexes)

        if tokenizer._pad_token is None:
            return {
                'inputs': pad_sequence(inputs, batch_first=True),
                'indexes': indexes
            }
        return {
            'inputs': pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id),
            'indexes': indexes
        }

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=32, collate_fn=collate)

    nb_eval_steps = 0
    scores, labels, indexes = [], [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, index = batch['inputs'], batch['indexes']
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs[0].contiguous()
            
            score, prediction = torch.softmax(logits, dim=0).max(dim=-1)
            scores.append(score)
            labels.append(prediction)
            indexes.append(index)
            
            batch_size = inputs.size(0)
            nb_eval_steps += batch_size
    
    scores, labels = torch.cat(scores, dim=0).tolist(), torch.cat(labels, dim=0).tolist()
    indexes = torch.cat(indexes, dim=0).tolist()
    return scores, labels, indexes


if __name__ == '__main__':
    data_dir = sys.argv[1]
    gpu = torch.device('cuda:0')
    model_dir = sys.argv[2]
    predict_output = sys.argv[3]
    threshold = float(sys.argv[4])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    data = json_load(data_dir)
    samples = []
    for idx, sample in enumerate(data):
        samples.append(sample['content'])
    
    scores, labels, indexes = classify(model_dir, samples, gpu, tokenizer)

    results = []
    LABEL_DICT = list(LABEL_DICT.keys())
    for idx, score, label in zip(indexes, scores, labels):
        if score >= threshold:
            sample = data[idx]
            sample['predict'] = {
                'label': LABEL_DICT[label],
                'score': score
            }
            results.append(sample)
    json_dump(results, predict_output)