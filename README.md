## CS598-project  
This repo  mainly tries to re-implement Repository for [Publicly Available Clinical BERT Embeddings]( for Publicly Available Clinical BERT Embeddings (NAACL Clinical NLP Workshop 2019)) (NAACL Clinical NLP Workshop 2019)

## Installation and Requirements

```
pip install -r requirements.txt
```
## Datasets
We use [MIMIC-III](https://mimic.physionet.org/about/mimic/). As MIMIC-III requires the CITI training program in order to use it, we refer users to the link. However, as clinical notes share commonality, users can test any clinical notes using the ClinicalBERT weight, although further fine-tuning from our checkpoint is recommended. 

File system expected:

```
-data
  -discharge
    -train.csv
    -val.csv
    -test.csv
  -3days
    -train.csv
    -val.csv
    -test.csv
  -2days
    -test.csv
```
Data file is expected to have column "TEXT", "ID" and "Label" (Note chunks, Admission ID, Label of readmission).

### Early Notes Prediction
If you want to fine-tuning with original bert, use `training_original_bert.py` instead of `training_clinical_bert.py`
```
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
```
when you want to do only inference
```
python modeling/training_clinical_bert.py \
  --qtype readmission \
  --do_train False \
  --do_test True \
  --data_file_path ./data/3days/ \
  --bert_model ./pretraining/ \
  --max_seq_len 512 \
  --output_file_path ./model_inference_result \
  --cuda_num 1 \
  --train_batch_size 32 \
  --learning_rate 0.00002 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 3 \
  --best_model_path ./experiment/readmission_model_experiment/readmission_clinical_bert_BATCH_SIZE_16_LEARNING_RATE_2e-05_gradient_accu_2_MAX_GRAD_NORM_1_0.pt
```
```
python modeling/training_original_bert.py \
  --qtype discharge \
  --do_train True \
  --do_test True \
  --data_file_path ./data/discharge/ \
  --bert_model bert-base-uncased \
  --max_seq_len 512 \
  --output_file_path ./final_model_discharge_bert-base \
  --cuda_num 2 \
  --train_batch_size 32 \
  --learning_rate 0.00002 \
  --gradient_accumulation_steps 1 \
  --max_grad_norm 3 \
  --num_train_epochs 3
 ```

### Discharge Summary Prediction
```
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
```
### Notebook executed files
- bi-lstm: https://github.com/ksj3421/CS598-project/blob/main/modeling/training_clinical_bert_readmission.ipynb
  - word2vec model: https://github.com/ksj3421/CS598-project/blob/main/modeling/clinical_word2vec_final.model (300 dimension)
- bag-of-word: https://github.com/ksj3421/CS598-project/blob/main/modeling/bag_of_words.ipynb

## ClinicalBERT Weights
Use [this google link](https://drive.google.com/open?id=1t8L9w-r88Q5-sfC993x2Tjt1pu--A900) or [this oneDrive link](https://hu-my.sharepoint.com/:u:/g/personal/kexinhuang_hsph_harvard_edu/ERw4LamJD4xNkkONXI7jsiYBUk6QwDv4t3y_jJcrsjkt9A?e=orU3C3) for users in mainland China to download pretrained ClinicalBERT along with the readmission task fine-tuned model weights.

The following scripts presume a model folder that has following structure:
```
-pretraining
    -bert_config.json
    -pytorch_model.bin
    -vocab.txt
```

## Citation

Please cite [arxiv](https://arxiv.org/abs/1904.05342):
```
@article{clinicalbert,
author = {Kexin Huang and Jaan Altosaar and Rajesh Ranganath},
title = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
year = {2019},
journal = {arXiv:1904.05342},
}
```
code reference
https://github.com/kexinhuang12345/clinicalBERT
