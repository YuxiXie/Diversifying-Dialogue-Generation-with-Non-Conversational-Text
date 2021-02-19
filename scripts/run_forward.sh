export CUDA_VISIBLE_DEVICES=2

EXEHOME=${HOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/forward

cd ${EXEHOME}

##=== training ===##
# python train.py

##=== predicting ===##
python predict.py