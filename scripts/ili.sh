#!/bin/sh

model_id_name=illness
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 24 36 48 60
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id $model_id_name'_'$pred_len \
  --model KoopBlock \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --dynamic_dim $dynamic_dim \
  --hidden_dim $hidden_dim \
  --hidden_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --l2 5e-4 \
  --patience 20 \
  --num_blocks 2 \
  --CI \
  --mask_type 'channel'\
  --itr 3 \
  --gpu 7
done

model_id_name=illness
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 48
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id $model_id_name'_'$pred_len \
  --model KoopBlock \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --dynamic_dim $dynamic_dim \
  --hidden_dim $hidden_dim \
  --hidden_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --l2 5e-4 \
  --patience 20 \
  --num_blocks 2 \
  --CI \
  --mask_type 'channel'\
  --itr 3 \
  --gpu 7
done

model_id_name=illness
dynamic_dim=256
hidden_dim=$((dynamic_dim*2))
for pred_len in 60
do
seq_len=$((pred_len*2))
seg_len=$((pred_len/3))
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id $model_id_name'_'$pred_len \
  --model KoopBlock \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --dynamic_dim $dynamic_dim \
  --hidden_dim $hidden_dim \
  --hidden_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --l2 5e-4 \
  --patience 20 \
  --num_blocks 1 \
  --CI \
  --mask_type 'channel'\
  --itr 3 \
  --gpu 7
done
