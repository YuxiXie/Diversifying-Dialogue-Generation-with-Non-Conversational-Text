#!/bin/bash

set -x

DATAHOME=${HOME}/data/DailyDialogue/processed
EXEHOME=${HOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/seq2seq

cd ${EXEHOME}

python dialogue_preprocess.py \
       -copy \
       -train_file ${DATAHOME}/train.csv -valid_file ${DATAHOME}/dev.csv \
       -save_data ${DATAHOME}/basic_uncased_data_128.pt \
       -src_seq_length 128 -tgt_seq_length 128 \
       -bert_tokenizer bert-base-uncased \
       -share_vocab