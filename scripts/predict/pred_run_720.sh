python -u run.py \
  --is_training 0 \
  --do_predict \
  --use_gpu 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --freq 't' \
  --itr 1
#python -u run.py \
#  --checkpoints ./checkpoints_nonstat/ \
#  --is_training 0 \
#  --do_predict \
#  --root_path ./dataset/self_made/ \
#  --data_path sin_on_line_10k.csv \
#  --model_id auto_10k_96_96 \
#  --model Autoformer \
#  --data ETTm2 \
#  --features S \
#  --seq_len 96 \
#  --label_len 96 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --des 'Exp' \
#  --freq 't' \
#  --itr 1