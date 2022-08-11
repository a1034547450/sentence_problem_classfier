#!/bin/bash

# for train

python train_bert_cls.py \
    --epochs=10 \
    --is_cv \
    --cv_number=5 \
    --batch_size=3 \
    --train_path='./dataset/train.csv' \
    --pooling='first-last-avg' \
    --log_interval=3 \
    --label_number=2 \
    --cache_dir='./cache/' \
    --save_dir='./model_save/' \
    --out_dir='./result/' \
    --pretrain_model='./pretrained_model/'

#for inference
#python train_bert_cls.py \
#    --inference \
#    --batch_size=12 \
#    --train_path='./dataset/train.csv' \
#    --pooling='first-last-avg' \
#    --log_interval=10 \
#    --label_number=2 \
#    --cache_dir='./cache/' \
#    --save_dir='./model_save/' \
#    --out_dir='./result/' \
#    --pretrain_model='./pretrained_model/' \
#    --inference_model='./model_save/best_model_for_cv0' \
#    --test_path='./dataset/test1.csv'