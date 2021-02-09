import os
import sys

from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces, data_process


def train(model_args, train_path, dev_path):
    # Data Loading
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(dev_path)
    train_df = data_process(train_df)
    eval_df = data_process(eval_df)

    # Model Initialization
    model = Seq2SeqModel(
        encoder_decoder_type="bart", 
        encoder_decoder_name="facebook/bart-large", 
        args=model_args,
    )

    # Model Training
    model.train_model(train_df, eval_data=eval_df)

    # Model Evaluating
    results = model.eval_model(eval_df)
    print(results)


if __name__ == '__main__':
    model_args = Seq2SeqArgs()
    model_args.eval_batch_size = 8
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 2500
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_seq_length = 128
    model_args.num_train_epochs = 5
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.train_batch_size = 8
    model_args.use_multiprocessing = False
    model_args.n_gpu = 1

    model_args.do_sample = True
    model_args.num_beams = None
    model_args.num_return_sequences = 3
    model_args.max_length = 128
    model_args.top_k = 50
    model_args.top_p = 0.95

    model_args.output_dir = '/home/yuxi/Projects/DiversifyDialogue/models/forward/initialization/DailyDialogue'
    model_args.best_model_dir = '/home/yuxi/Projects/DiversifyDialogue/models/forward/initialization/DailyDialogue/best_model'
    model_args.wandb_project = 'DailyDialogue forward initialization'

    train_path = '/home/yuxi/Projects/DiversifyDialogue/data/DailyDialogue/processed/train.csv'
    test_path = '/home/yuxi/Projects/DiversifyDialogue/data/DailyDialogue/processed/dev.csv'

    train(model_args, train_path, test_path)