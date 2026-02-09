export CUDA_VISIBLE_DEVICES=0

model_name=MA_Mamba
#state*1 state*1
python -u run.py --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt --model_id solar_96_96 --model MA_Mamba --data Solar --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --d_model 512 --d_ff 512 --itr 1
#state*2 state*4
python -u run.py --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt --model_id solar_96_192 --model MA_Mamba --data Solar --features M --seq_len 96 --pred_len 192 --e_layers 2 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --d_model 512 --d_ff 512 --itr 1

#专家2
python -u run.py --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt --model_id solar_96_336 --model MA_Mamba --data Solar --features M --seq_len 96 --pred_len 336 --e_layers 2 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --d_model 512 --d_ff 512 --itr 1

#专家2
python -u run.py --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt --model_id solar_96_720 --model MA_Mamba --data Solar --features M --seq_len 96 --pred_len 720 --e_layers 2 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --d_model 512 --d_ff 512 --itr 1
