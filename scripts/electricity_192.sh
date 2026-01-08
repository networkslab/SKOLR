#!/bin/sh

model_id_name=electricity
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 96 #144 192
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --patience 10 \
  --l2 5e-3 \
  --num_blocks 2 \
  --CI \
  --mask_type 'channel'\
  --random_seed 2022 \
  --itr 3 \
  --gpu 3
done

#  --use_mlflow \
#  --mlflow_project 'ele_DKoop'
