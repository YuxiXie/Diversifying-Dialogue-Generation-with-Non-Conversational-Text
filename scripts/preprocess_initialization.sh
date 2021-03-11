#!/bin/bash

set -x

PROJECTHOME=/home/yuxi/Projects/DiversifyDialogue

DATAHOME=${PROJECTHOME}/data/ELI5/processed
EXEHOME=${PROJECTHOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/seq2seq

cd ${EXEHOME}

python dialogue_preprocess.py \
       -copy \
       -train_file ${DATAHOME}/train.csv -valid_file ${DATAHOME}/dev.csv \
       -save_data ${DATAHOME}/eli5_uncased_data_128.pt \
       -src_seq_length 128 -tgt_seq_length 128 \
       -bert_tokenizer bert-base-uncased \
       -share_vocab