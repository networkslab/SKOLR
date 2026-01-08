#!/bin/sh

model_id_name=Duffing
seq_len=1000
seg_len=100

# for pred_len in 96 #192 336 720
pred_len=500
for num_blocks in 2 # 4 8
do
    dynamic_dim=$((512/num_blocks))
    hidden_dim=$((dynamic_dim*2))
    python -u run_longExp.py \
      --is_training 0 \
      --root_path ./dataset/ \
      --data_path 'duffing.csv' \
      --model_id $model_id_name'_'$num_blocks'_'$dynamic_dim \
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
      --gpu 6
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
