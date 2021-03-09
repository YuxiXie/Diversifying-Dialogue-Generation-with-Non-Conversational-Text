# !/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0

DATAHOME=${PROJECTHOME}/data/ELI5
EXEHOME=${PROJECTHOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/topic-classifier
MODELHOME=${PROJECTHOME}/models/classifier/best

cd ${EXEHOME}

python get_prediction.py \
    # input .json data
    ${DATAHOME}/processed.json \
    # model directory
    ${MODELHOME} \
    # prediction output .json file
    ${DATAHOME}/classified_data.json \
    # threshold to filter the data
    0.25
