#!/bin/sh

model_id_name=traffic
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 96
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id $model_id_name'_'$pred_len \
  --model KoopBlock \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --dynamic_dim $dynamic_dim \
  --hidden_dim $hidden_dim \
  --hidden_layers 3 \
  --dropout 0.05 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --l2 5e-3 \
  --patience 10 \
  --num_blocks 2 \
  --CI \
  --mask_type 'global' \
  --random_seed 2022 \
  --itr 3 \
  --gpu 7
done