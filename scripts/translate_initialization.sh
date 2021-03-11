  # !/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=1

PROJECTHOME=/home/yuxi/Projects/DiversifyDialogue

DATAHOME=${PROJECTHOME}/data/DailyDialogue/processed
EXEHOME=${PROJECTHOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/seq2seq
MODELHOME=${PROJECTHOME}/models/seq2seq/initialization

mkdir -p ${DATAHOME}/predictions

cd ${EXEHOME}

python translate.py \
       -mode initialization \
       -data ${DATAHOME}/basic_uncased_data_128.pt \
       -model ${MODELHOME}/initialization.chkpt \
       -output ${DATAHOME}/predictions/basic-initialization.txt \
       -gpus 0 \
       -batch_size 32