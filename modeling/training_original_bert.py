from transformers import AutoTokenizer, AutoModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import os
from modeling_utils import *
from transformers.optimization import get_linear_schedule_with_warmup, AdamW, get_cosine_schedule_with_warmup
import json
import torch

import logging
import json
import matplotlib.pyplot as plt
from csv import DictWriter

def get_data_loader(features, max_seq_length, batch_size, shuffle=True, add_sampler=False): 

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if add_sampler == True:
            logger.info('add sampler')
            if local_rank == -1:
                sampler = SequentialSampler(data)
            else:
                sampler = DistributedSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=EVAL_BATCH)
        else:
            dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader

class AbstractTrainer():
    def __init__(self, args, model, train_dataloader, dev_dataloader, n_gpu):
        self.args = args
        self.classifier = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.n_gpu = n_gpu
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.device = args.device
        self.output_file_path = args.output_file_path
        self.num_train_epochs = args.num_train_epochs
        self.best_model_path = ''
        self.qtype = args.qtype
        self.best_score = -999
        self.warmup_ratio = args.warmup_ratio
        self.tr_loss_epoch = 0
        self.max_seq_len = args.max_seq_len

    def train_one_epoch(self, epoch):
        global_step = 0 
        global_step_check = 0
        no_improvement = 0
        train_loss_history = []
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0
        self.classifier.zero_grad()
        for step, batch in enumerate(self.train_dataloader):
            self.classifier.to(self.device)
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            #with torch.set_grad_enabled(True):
            out = self.classifier(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            loss = out[0]
            logits = out[1]
            if self.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward() 
            train_loss_history.append(loss.item())
            tr_loss += loss.item()
            if ((step + 1) % self.gradient_accumulation_steps == 0  or (step+1) == len(self.train_dataloader)):
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)  
                self.optimizer.step()
                self.scheduler.step()
                self.classifier.zero_grad()
                global_step += 1
            if (step+1) % 200 == 0:
                logging_str =  "***** epoch [{}]".format(epoch)
                logging_str += " global_step [{}]".format(global_step) 
                logging_str += " train loss [{}]".format(loss.item())  
                logging.info(logging_str)
            nb_tr_steps = nb_tr_steps + 1
        tr_loss_epoch = tr_loss / nb_tr_steps
        learning_rate_list = self.scheduler.get_lr()
        return tr_loss_epoch, train_loss_history, learning_rate_list

    def create_optimizer_scheduler(self):
        num_train_steps = int(len(self.train_dataloader.dataset) / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)   
        num_warmup_steps = int(self.warmup_ratio * num_train_steps)
        param_optimizer = list(self.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.learning_rate, correct_bias=False)

        self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    def train(self):
        total_train_loss_history = []
        total_val_loss_history = []
        total_learning_rate = []
        df_raw = pd.read_csv(f'{self.args.data_file_path}val.csv')
        self.create_optimizer_scheduler()
        for real_epoch in trange(int(self.num_train_epochs)):
            self.classifier.train()
            tr_loss_epoch, train_loss_history, learning_rate_list = self.train_one_epoch(real_epoch)
            self.tr_loss_epoch = tr_loss_epoch
            total_train_loss_history.extend(train_loss_history)
            total_learning_rate.extend(learning_rate_list)
            result, loss_history = self.evaluate(real_epoch, self.dev_dataloader, df_raw, eval_type='val')
            total_val_loss_history.extend(loss_history)
            # plt.plot(total_train_loss_history)
            # plt.plot(total_val_loss_history)
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'val'], loc='upper left')
            # plt.show()
            # string = 'train_test_plot_'+f'{self.args.qtype}'+ f'{str(real_epoch)}' + '.png'
            # plt.savefig(os.path.join(args.output_file_path, string))
            model_name = f'{qtype}_clinical_bert_BATCH_SIZE_{str(self.train_batch_size)}_LEARNING_RATE_{str(self.learning_rate)}_gradient_accu_{str(self.gradient_accumulation_steps)}_MAX_GRAD_NORM_{str(self.max_grad_norm)}_{str(real_epoch)}.pt'
            model_save_check_score = (result['AUROC'] + result['AUPRC']) / 2
            if self.best_score < model_save_check_score:
                self.best_score = model_save_check_score
                torch.save(self.classifier.state_dict(), f'{self.output_file_path}/{model_name}')
                self.best_model_path = f'{self.output_file_path}/{model_name}'
        plt.plot(total_train_loss_history)
        plt.plot(total_val_loss_history)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        string = 'train_test_plot_'+f'{self.args.qtype}'+ f'{str(real_epoch)}' + '.png'
        plt.savefig(os.path.join(self.output_file_path, string))
        
    def evaluate(self, epoch, dataloader, df_test, eval_type='val'):
        nb_eval_examples = 0
        nb_eval_steps = 0
        m = nn.Sigmoid()
        self.classifier.eval()
        eval_loss = 0
        eval_accuracy = 0
        nb_eval_steps = 0
        pred_labels, true_labels, logits_history, pred_scores = [], [], [], []   
        loss_history = []

        for step, batch in enumerate(dataloader): 
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                outs = self.classifier(input_ids, attention_mask=input_mask,
                                        token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = outs[0]
                temp_logits = outs[1]
                logits = self.classifier(input_ids,segment_ids,input_mask) # we don't need twice

            logits = torch.squeeze(m(logits['logits'])).detach().cpu().numpy() 
            label_ids = np.array(np.array(label_ids.to('cpu')))
            try:
                outputs = np.asarray([1 if i else 0 for i in (logits[:,1] >=0.5)])
            except:
                outputs = np.asarray([1 if logits[1] >= 0.5 else 0])
            tmp_eval_accuracy=np.sum(outputs == label_ids)    

            true_labels += list(label_ids)
            pred_labels += list(outputs)
            try:
                logits_history = logits_history + logits[:,1].tolist()
            except:
                logits_history = logits_history +  [logits[1]]

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            loss_history.append(tmp_eval_loss.item())
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        logging_str =  "***** epoch [{}]".format(epoch)
        logging_str += " {} [{:.4}]".format('val_loss', eval_loss)

        fpr, tpr, df_out, roc_auc = vote_score(df_test, logits_history, self.output_file_path, self.qtype, epoch)
        pr_auc = pr_curve_plot(df_test['Label'].values, logits_history, self.output_file_path, self.qtype, epoch)
        rp80 =  vote_pr_curve(df_test, logits_history, self.output_file_path, self.qtype, epoch)
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,  
                  'training loss': self.tr_loss_epoch,
                  'AUROC': roc_auc,
                  'AUPRC' : pr_auc,
                  'RP80': rp80}    
        result['model'] = self.qtype
        result['epoch'] = epoch
        result['batch_size'] = self.train_batch_size
        result['MAX_GRAD_NORM'] = self.max_grad_norm
        result['GRADIENT_ACCUMULATION_STEPS'] = self.gradient_accumulation_steps
        result['MAX_SEQ_LENGTH'] = self.max_seq_len
        result['LEARNING_RATE'] = self.learning_rate
        result['eval_type'] = eval_type
        headers = {'eval_loss': None, 'eval_accuracy': None, 'training loss': None, 'AUROC': None, 'AUPRC': None, 'RP80': None, 'model': None, 'epoch': None, 'batch_size': None, 'MAX_GRAD_NORM': None, 'GRADIENT_ACCUMULATION_STEPS': None, 'MAX_SEQ_LENGTH': None, 'LEARNING_RATE': None, 'eval_type': None}
        try:
            df = pd.read_csv(f'{self.output_file_path}/result.csv')
            with open(f'{self.output_file_path}/result.csv', 'a', newline='') as f:
                dictwriter_object = DictWriter(f, delimiter=',', fieldnames=headers)
                dictwriter_object.writerow(result)
        except:
            with open(f'{self.output_file_path}/result.csv', 'a', newline='') as f:
                dictwriter_object = DictWriter(f, delimiter=',', fieldnames=headers)
                dictwriter_object.writeheader()
                dictwriter_object.writerow(result)
        return result, loss_history
 
def main():
    ROOT_DIR = '/'.join(os.getcwd().split('/'))
    print(ROOT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                            type=str,
                            default=f'{ROOT_DIR}',
                            help='ROOT_PATH which has data and models folders')
    parser.add_argument('--local_rank',
                        type=str,
                        default=0,
                        help='0 if single gpu not support multi-gpu due to clinicalbert structure itself')
    parser.add_argument('--need_proxy',
                        type=str,
                        default=True,
                        help='only for specific machine')
    parser.add_argument('--bert_model',
                        type=str,
                        default='',
                        help='bert model path')
    parser.add_argument('--data_file_path',
                        type=str,
                        default=f'{ROOT_DIR}/data/3days/',
                        help='data file path, needs to be end with slash')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=32,
                        help='train_batch_size')
    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=32,
                        help='eval_batch_size')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=512,
                        help='max seq len')
    parser.add_argument('--learning_rate',
                        type=int,
                        default=2e-5,
                        help='learning_rate')
    parser.add_argument('--max_grad_norm',
                        type=int,
                        default=1,
                        help='max_grad_norm')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help='gradient_accumulation_steps')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='gradient_accumulation_steps')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=1,
                        help='num_train_epochs')
    parser.add_argument('--output_file_path',
                        type=str,
                        default=f'{ROOT_DIR}/final_model_readmission/',
                        help='output file path')
    parser.add_argument('--qtype',
                        type=str,
                        default='readmission',
                        help='readmission or discharge')
    parser.add_argument('--no_cuda',
                        type=str,
                        default=False,
                        help='running on cpu')
    parser.add_argument('--cuda_num',
                        type=int,
                        default=0,
                        help='cuda number')
    parser.add_argument('--do_train',
                        type=str,
                        default=True,
                        help='do train')
    parser.add_argument('--do_test',
                        type=str,
                        default=True,
                        help='do test')
    parser.add_argument('--best_model_path',
                        type=str,
                        default='',
                        help='running on cpu')
    args = parser.parse_args()
    
    ## load arguments
    os.chdir(args.root_path)
    local_rank = args.local_rank
    seed = 2022
    if args.gradient_accumulation_steps > 1 :
        BATCH_SIZE = int(args.train_batch_size / args.gradient_accumulation_steps)
    else:
        BATCH_SIZE = args.train_batch_size
    
    os.makedirs(args.output_file_path, exist_ok=True)
    need_proxy = args.need_proxy
    if need_proxy:
        local_test = False
        proxy_file_path = f"{args.root_path}/secrets.json"
        
    if local_test:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        with open(proxy_file_path, "r") as json_file:
            json_data = json.load(json_file)
            proxies = json_data['proxies']
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', proxies=proxies) 
            
    ## preprocess dataset 
    processor = clinicalNoteProcessor()
    label_list = processor.get_labels()
    ## model folder named 'pretraining' should be in the root folder
    if local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda", args.cuda_num)
        n_gpu = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if args.do_train == 'True':
        # load train dataset
        train_examples = processor.get_train_examples(f'{args.data_file_path}')
        dev_example = processor.get_dev_examples(f'{args.data_file_path}')
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_len, tokenizer)
        dev_features = convert_examples_to_features(
                dev_example, label_list, args.max_seq_len, tokenizer)
        train_dataloader = get_data_loader(train_features, args.max_seq_len, BATCH_SIZE, shuffle=True)
        dev_dataloader = get_data_loader(dev_features, args.max_seq_len, args.eval_batch_size, shuffle=False)
        
        # initialize trainer
        Classifier = BertForSequenceClassification.from_pretrained(f'{args.bert_model}', 1)
        trainer = AbstractTrainer(args=args, model=Classifier, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, n_gpu=n_gpu)
        trainer.train() # train and eval at the same time
        if args.do_test == 'True':
            #load best model
            trainer.classifier.load_state_dict(torch.load(trainer.best_model_path))
            Classifier.to(args.device)
            ## load test data set
            test_example = processor.get_test_examples(f'{args.data_file_path}')
            test_features = convert_examples_to_features(
                    test_example, label_list, args.max_seq_len, tokenizer)
            test_dataloader = get_data_loader(test_features, args.max_seq_len, args.eval_batch_size, shuffle=False)
            if args.qtype == 'readmission':
                test_2_example = processor.get_test_examples(f'{args.root_path}/data/2days/')
                test_2days_features = convert_examples_to_features(test_2_example, label_list, args.max_seq_len, tokenizer)
                test_2days_dataloader = get_data_loader(test_2days_features, args.max_seq_len, args.eval_batch_size, shuffle=False)
            ## evaluation
            if args.qtype == 'readmission':
                df_raw = pd.read_csv(f'{args.data_file_path}test.csv')
                result, loss_history = trainer.evaluate(0, test_dataloader, df_raw, eval_type='test_3days')
                df_raw = pd.read_csv(f'{args.root_path}/data/2days/test.csv')
                result, loss_history = trainer.evaluate(0, test_2days_dataloader, df_raw, eval_type='test_2days')
            if args.qtype == 'discharge':
                df_raw = pd.read_csv(f'{args.data_file_path}test.csv')
                result, loss_history = trainer.evaluate(0, test_dataloader, df_raw, eval_type='discharge')
        else:
            pass

    if args.do_train == 'False':
        if args.do_test == 'True':
            assert self.model_check_point == '', 'model_check_point is None'
            ## load check point model
            Classifier = BertForSequenceClassification.from_pretrained(os.path.join(f'{args.root_path}/', 'pretraining'), 1)
            classifier.load_state_dict(torch.load(self.model_check_point))
            trainer = AbstractTrainer(args=args, model=Classifier, train_dataloader=None, dev_dataloader=None, n_gpu=n_gpu)
            ## load test data set
            test_example = processor.get_test_examples(f'{args.data_file_path}')
            test_features = convert_examples_to_features(
                    test_example, label_list, args.max_seq_len, tokenizer)
            test_dataloader = get_data_loader(test_features, args.max_seq_len, args.eval_batch_size, shuffle=False)
            if args.qtype == 'readmission':
                test_2_example = processor.get_test_examples(f'{args.root_path}/data/2days/')
                test_2days_features = convert_examples_to_features(test_2_example, label_list, args.max_seq_len, tokenizer)
                test_2days_dataloader = get_data_loader(test_2days_features, args.max_seq_len, args.eval_batch_size, shuffle=False)
            ## evaluation
            if args.qtype == 'readmission':
                df_raw = pd.read_csv(f'{args.data_file_path}test.csv')
                result, loss_history = trainer.evaluate(0, test_dataloader, df_raw, eval_type='test_3days')
                df_raw = pd.read_csv(f'{args.root_path}/data/2days/test.csv')
                result, loss_history = trainer.evaluate(0, test_2days_dataloader, df_raw, eval_type='test_2days')
            if args.qtype == 'discharge':
                df_raw = pd.read_csv(f'{args.data_file_path}test.csv')
                result, loss_history = trainer.evaluate(0, test_dataloader, df_raw, eval_type='discharge')
        else:
            pass
            print('nothing to do, check do_test True or False')
        
if __name__ == "__main__":   
    torch.cuda.empty_cache()
    main()
    