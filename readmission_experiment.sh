for train_batch_size in 16 32;  do
    for ACCUMULATION_STEP in 1 2 4;  do
        for max_grad_norm in 1 3 5;  do
            for learning_rate in 0.00005 0.00002; do
                python ./modeling/training_clinical_bert.py \
                  --qtype discharge \
                  --do_train True \
                  --do_test True \
                  --data_file_path ./data/3days/ \
                  --bert_model ./pretraining/ \
                  --max_seq_len 512 \
                  --output_file_path ./readmission_model_experiment \
                  --cuda_num 0 \
                  --train_batch_size ${train_batch_size} \
                  --learning_rate ${learning_rate} \
                  --gradient_accumulation_steps ${ACCUMULATION_STEP} \
                  --max_grad_norm ${max_grad_norm} \
                  --num_train_epochs 1
            done
        done
    done
done