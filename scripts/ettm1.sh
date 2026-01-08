#!/bin/sh

dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for model_id_name in ETTm1
do
  for pred_len in 48 #96 144 192
  do
    seq_len=$((pred_len*2))
    seg_len=$((pred_len/3))
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $model_id_name'.csv' \
      --model_id $model_id_name'_'$pred_len \
      --model KoopBlock_P \
      --data $model_id_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len $seg_len \
      --dynamic_dim $dynamic_dim \
      --hidden_dim $hidden_dim \
      --dropout 0.2 \
      --hidden_layers 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --learning_rate 0.0001 \
      --patience 20 \
      --l2 5e-3 \
      --num_blocks 2 \
      --CI \
      --mask_type 'channel' \
      --itr 1 \
      --gpu 1
done
done