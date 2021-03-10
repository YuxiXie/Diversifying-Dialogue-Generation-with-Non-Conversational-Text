# !/bin/bash

set -x

PROJECTHOME=/home/yuxi/Projects/DiversifyDialogue

DATAHOME=${PROJECTHOME}/data/DailyDialogue/processed
EXEHOME=${PROJECTHOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/seq2seq
MODELHOME=${PROJECTHOME}/models/seq2seq/initialization
LOGHOME=${PROJECTHOME}/models/seq2seq/initialization/logs

mkdir -p ${MODELHOME}
mkdir -p ${LOGHOME}

cd ${EXEHOME}

export CUDA_VISIBLE_DEVICES=1

python train.py \
       -gpus 0 \
       -data ${DATAHOME}/basic_uncased_data_128.pt \
       -epoch 50 -batch_size 32 -eval_batch_size 32 \
       -max_token_src_len 128 -max_token_tgt_len 64 \
       -copy -coverage -coverage_weight 0.4 \
       -d_word_vec 300 \
       -d_enc_model 512 -n_enc_layer 1 -brnn -enc_rnn gru \
       -d_dec_model 512 -n_dec_layer 1 -dec_rnn gru -d_k 64 \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.3 -attn_dropout 0.1 \
       -save_mode best -save_model ${MODELHOME}/initialization \
       -logfile_train ${LOGHOME}/initialization.train \
       -logfile_dev ${LOGHOME}/initialization.dev \
       -log_home ${LOGHOME} \
       -translate_ppl 20 -translate_steps 500 \
       -curriculum 0  -extra_shuffle -optim adam -learning_rate 0.001 -learning_rate_decay 0.75 \
       -valid_steps 250 -decay_steps 500 -start_decay_steps 1000 -decay_bad_cnt 5 -max_grad_norm 5 -max_weight_value 32 