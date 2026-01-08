#!/bin/sh

seq_len=96
seg_len=16
model_id_name=ETTh2

for pred_len in 96 192 336 720
do
for seg_len in 16 32 48
do
for num_blocks in 1 2 3
do
for dynamic_dim in 64 128
do
  hidden_dim=$((dynamic_dim*2))
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $model_id_name'.csv' \
      --model_id $model_id_name'_'$pred_len \
      --model KoopBlock \
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
      --num_blocks $num_blocks \
      --l2 5e-3 \
      --CI \
      --mask_type 'channel' \
      --itr 1 \
      --gpu 7
done
done
done
done

#for pred_len in 336 720
#  do
#  dynamic_dim=128
#  hidden_dim=$((dynamic_dim*2))
#    python -u run_longExp.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path $model_id_name'.csv' \
#      --model_id $model_id_name'_'$pred_len'_no_inverse' \
#      --model KoopMambaFFT \
#      --data $model_id_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --seg_len $seg_len \
#      --dynamic_dim $dynamic_dim \
#      --hidden_dim $hidden_dim \
#      --dropout 0.2 \
#      --hidden_layers 1 \
#      --enc_in 7 \
#      --dec_in 7 \
#      --c_out 7 \
#      --des 'Exp' \
#      --learning_rate 0.0001 \
#      --patience 20 \
#      --CI \
#      --itr 1 \
#      --gpu 0
#  done
