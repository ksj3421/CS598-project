python modeling/training_clinical_bert.py \
  --qtype discharge \
  --do_train True \
  --do_test True \
  --data_file_path ./data/discharge/ \
  --bert_model ./pretraining/ \
  --max_seq_len 512 \
  --output_file_path ../final_model_discharge \
  --cuda_num 1 \
  --train_batch_size 32 \
  --learning_rate 0.00005 \
  --gradient_accumulation_steps 1 \
  --max_grad_norm 3 \
  --num_train_epochs 3


python modeling/training_clinical_bert.py \
  --qtype readmission \
  --do_train True \
  --do_test True \
  --data_file_path ./data/3days/ \
  --bert_model ./pretraining/ \
  --max_seq_len 512 \
  --output_file_path ./clinicalbert_model_readmission \
  --cuda_num 0 \
  --train_batch_size 32 \
  --learning_rate 0.00002 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 3

python modeling/training_clinical_bert.py \
  --qtype readmission \
  --do_train False \
  --do_test True \
  --data_file_path ./data/3days/ \
  --bert_model ./pretraining/ \
  --max_seq_len 512 \
  --output_file_path ./final_model_readmission \
  --cuda_num 1 \
  --train_batch_size 32 \
  --learning_rate 0.00002 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 3
