#!/bin/sh

model_id_name=electricity
dynamic_dim=256
hidden_dim=256
#hidden_dim=$((dynamic_dim*2))
seq_len=96
for seg_len in 16 24 32 48
do
for pred_len in 96 #192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id $model_id_name'_'$pred_len'_global' \
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
  --mask_type 'channel'\
  --itr 1 \
  --gpu 6
done
done

#  --use_mlflow \
#  --mlflow_project 'ele_DKoop'
