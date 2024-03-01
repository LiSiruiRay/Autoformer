source myenv/bin/activate
pip install -r requirements.txt

python -u run.py \
  --is_training 1 \
  --train_epochs 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id testing \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --freq 't' \
  --itr 1