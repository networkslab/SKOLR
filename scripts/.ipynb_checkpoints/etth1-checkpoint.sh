#!/bin/sh

model_id_name=ETTh1
seq_len=96
seg_len=16

for pred_len in 96 #192 336 720
  do
  dynamic_dim=128
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
      --l2 5e-3 \
      --num_blocks 2 \
      --CI \
      --mask_type 'channel' \
      --itr 1 \
      --gpu 0
done

#      --mask_type 'channel' \


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
#      --l2 5e-3 \
#      --patience 20 \
#      --CI \
#      --itr 1 \
#      --gpu 0
##done
