for train_batch_size in 16 32;  do
    for ACCUMULATION_STEP in 1 2;  do
        for max_grad_norm in 1 3;  do
            for learning_rate in 0.00005 0.00002; do
                python ./modeling/training_clinical_bert.py \
                  --qtype discharge \
                  --do_train True \
                  --do_test True \
                  --data_file_path ./data/discharge/ \
                  --bert_model ./pretraining/ \
                  --max_seq_len 512 \
                  --output_file_path ./experiment/discharge_model_experiment \
                  --cuda_num 1 \
                  --train_batch_size ${train_batch_size} \
                  --learning_rate ${learning_rate} \
                  --gradient_accumulation_steps ${ACCUMULATION_STEP} \
                  --max_grad_norm ${max_grad_norm} \
                  --num_train_epochs 3
            done
        done
    done
done