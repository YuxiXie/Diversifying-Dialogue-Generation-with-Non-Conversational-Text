import os
import sys
from datetime import datetime
import logging

import pandas as pd
import json
from tqdm import tqdm

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces, data_process


def predict(model_args, test_path, output_path, model_path):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    # Data Loading
    eval_df = pd.read_csv(test_path)
    eval_df = data_process(eval_df)

    # Model Loading
    model = Seq2SeqModel(
        encoder_decoder_type="bart", 
        encoder_decoder_name=model_path,
        args=model_args
    )

    # Model Prediction
    to_predict = [
        str(input_text) for input_text in eval_df["input_text"].tolist()
    ]

    GTs = [gt for gt in eval_df["target_text"].tolist()]

    num_batches = int(len(to_predict) / model_args.eval_batch_size)

    outputs = []
    for ind in tqdm(range(num_batches + 1)): 
        data_batch, GT_batch = None, None
        if (ind + 1) * model_args.eval_batch_size > len(to_predict):
            data_batch = to_predict[ind * model_args.eval_batch_size:]
            GT_batch = GTs[ind * model_args.eval_batch_size:]
        else:
            data_batch = to_predict[ind * model_args.eval_batch_size:(ind + 1) * model_args.eval_batch_size]
            GT_batch = GTs[ind * model_args.eval_batch_size:(ind + 1) * model_args.eval_batch_size]
        
        preds = model.predict(data_batch)
        for i in range(len(preds)):
            outputs.append({'input': data_batch[i], 'target': GT_batch[i], 'predict': preds[i]})

    with open(output_path, 'w') as f:
        f.write(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    # predict
    model_args = Seq2SeqArgs()
    model_args.max_length = 128
    model_args.eval_batch_size = 4
    model_args.do_sample = True
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.num_return_sequences = 3
    model_args.num_beams = None

    test_path = '/home/yuxi/Projects/DiversifyDialogue/data/DailyDialogue/processed/test.csv'
    output_path = '/home/yuxi/Projects/DiversifyDialogue/outputs/forward/dailydialogue_innitialize.json'
    model_path = '/home/yuxi/Projects/DiversifyDialogue/models/forward/initialization/DailyDialogue/best_model'

    predict(model_args, test_path, output_path, model_path)