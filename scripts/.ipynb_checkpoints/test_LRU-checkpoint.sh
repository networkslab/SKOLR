#!/bin/sh


model_id_name=LotkaVolterra
seq_len=1000
seg_len=100

# for pred_len in 96 #192 336 720
pred_len=500
for num_blocks in 1 #4 8
do
    dynamic_dim=$((512/num_blocks))
    hidden_dim=$((dynamic_dim*2))
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path 'lotka_volterra.csv' \
      --model_id $model_id_name'_'$num_blocks'_'$dynamic_dim \
      --model LRU \
      --data $model_id_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len $seg_len \
      --dynamic_dim $dynamic_dim \
      --hidden_dim $hidden_dim \
      --dropout 0.2 \
      --hidden_layers 1 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --learning_rate 0.0001 \
      --patience 8 \
      --l2 5e-3 \
      --num_blocks $num_blocks \
      --CI \
      --mask_type 'channel' \
      --itr 1 \
      --gpu 0
done
