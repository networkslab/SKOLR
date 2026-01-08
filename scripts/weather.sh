#!/bin/sh

model_id_name=weather
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 48 96 144 192
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
# seg_len=48
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
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
  --dropout 0.2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --l2 5e-4 \
  --patience 10 \
  --num_blocks 2 \
  --CI \
  --mask_type 'channel'\
  --itr 3 \
  --gpu 3
done

  # --random_seed 2022 \
