# !/bin/bash

set -x

DATAHOME=${HOME}/data/DailyDialogue/processed-for-classification
EXEHOME=${HOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/topic-classifier
MODELHOME=${HOME}/models/classifier

mkdir -p ${MODELHOME}

cd ${EXEHOME}

export CUDA_VISIBLE_DEVICES=2,3

python train_classifier.py \
       --model_name_or_path bert-base-uncased \
       --model_type bert \
       --output_dir ${MODELHOME} \
       --overwrite_output_dir \
       --tokenizer_name bert-base-uncased \
       --train_data_file ${DATAHOME}/train.json \
       --eval_data_file ${DATAHOME}/dev.json \
       --line_by_line \
       --learning_rate 2e-5 \
       --block_size 128 \
       --per_gpu_train_batch_size 2 \
       --per_gpu_eval_batch_size 2 \
       --do_train \
       --evaluate_during_training \
       --num_train_epochs 32