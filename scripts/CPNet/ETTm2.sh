if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_96_96 --model CPNet --data ETTm2 --features M --seq_len 96 --pred_len 96 --des 'Exp' --enc_in 7 --c_out 7 >logs/LongForecasting/CPNet_ETTm2_96_96.log

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_96_192 --model CPNet --data ETTm2 --features M --seq_len 96 --pred_len 192 --des 'Exp' --enc_in 7 --c_out 7 >logs/LongForecasting/CPNet_ETTm2_96_192.log

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_96_336 --model CPNet --data ETTm2 --features M --seq_len 96 --pred_len 336 --des 'Exp' --enc_in 7 --c_out 7 >logs/LongForecasting/CPNet_ETTm2_96_336.log

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_96_720 --model CPNet --data ETTm2 --features M --seq_len 96 --pred_len 720 --des 'Exp' --enc_in 7 --c_out 7 >logs/LongForecasting/CPNet_ETTm2_96_720.log
