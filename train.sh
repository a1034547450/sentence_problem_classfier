#!/bin/bash

python train_bert_cls.py \
    --epochs=10 \
    --is_cv \
    --cv_number=5 \
    --batch_size=32 \
    --train_path='./dataset/train.csv' \
    --pooling='first-last-avg' \
    --log_interval=10 \
    --label_number=2 \
    --cache_dir='./cache/' \
    --save_dir='./model_save/' \
    --out_dir='./result/' \
    --pretrain_model='./pretrained_model/'