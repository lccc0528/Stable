import random
import itertools
# import wandb
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.dataset import T
from transformers import RobertaTokenizer, RobertaConfig, BertTokenizer, BertConfig,AutoTokenizer
import argparse
from dataloader import AllDataset, TokenizedForFakeNews, TokenizedForSentiment,TokenizedForNLI,FakeNewsNetDataset, SentimentDataset,PHEMEDataset,QNLIDataset
from utils import load_config, set_seed, train_val_split, print_measures, get_label_blance
from model import GeneratedPrompt, BertFineTune
import pandas as pd
from ray import tune
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_template(task):

    if task == 'fakenews':
        positive_words = ['real']#['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']  
        negative_words = ['fake']#['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
    elif task == 'sentiment':     
        positive_words = ['positive']  
        negative_words = ['negative']
    
    elif task == 'nli':
        positive_words = ['Clearly']#['yes']#['same','similar']
        negative_words = ['Yet']#['no']#['unlike','different']
    return positive_words , negative_words


def get_dataset(args,data_path):
    
    train_data_path = data_path + "/" + args.dataset + "/" + "train_data.csv"
    val_data_path = data_path + "/" + args.dataset + "/" + "val_data.csv"
    test_data_path = data_path + "/" + args.dataset + "/" + "test_data.csv"

    train_data = AllDataset(train_data_path, args.task) 
    val_data = AllDataset(val_data_path, args.task) 
    test_data = AllDataset(test_data_path, args.task) 
    
    ids = [i for i in range(len(train_data))]  # type: ignore
    random.shuffle(ids)

    return train_data, val_data, test_data


def get_data(args,data_path):
    data_path = data_path + "/" + args.dataset + "/" + "tweet.tsv"
    if args.dataset == 'gossipop' or args.dataset == 'politifact':
        data = FakeNewsNetDataset(data_path)
    elif args.dataset == 'sst-2' or args.dataset == 'mr':
        data = SentimentDataset(data_path)
    elif args.dataset == 'chnsenticorp':
        data = ChnsentiDataset(data_path)
    elif args.dataset == 'LCQMC':
        data = LCQMCDataset(data_path)
    elif args.dataset == 'gpt2':
        data = GPT2Dataset(data_path)
    elif args.dataset == 'qnli' or args.dataset == 'rte':
        data = QNLIDataset(data_path)
    elif args.dataset == 'PHEME':
        data = PHEMEDataset(data_path)
    
    return data 
 
def split_data(args,data):
    ids = [i for i in range(len(data))]  # type: ignore
    random.shuffle(ids)  

    train_ids_pool, val_ids_pool = get_label_blance(data, ids, args.shot_num)
    train_ids = train_ids_pool
    val_ids = val_ids_pool
    test_ids = ids.copy()
    
    for i in itertools.chain(train_ids, val_ids):
        test_ids.remove(i)
    
    train_data = Subset(data, train_ids)
    val_data = Subset(data, val_ids)
    test_data = Subset(data, test_ids)
    return train_data, val_data , test_data

def get_collate(task,tokenizer,args):

    if task == 'fakenews' :
        tokenized_collator = TokenizedForFakeNews(args,
                                                  tokenizer,
                                                  token_idx=0,
                                                  sort_key=lambda x:x[2],
                                                  using_prompt=args.using_hard_prompt,
                                                  need_mask_token = args.using_mask_token
                                                    )
    
    if task == 'sentiment':
        tokenized_collator = TokenizedForSentiment(args,
                                                   tokenizer,
                                                  token_idx=0,
                                                  sort_key=lambda x:x[2],
                                                  using_prompt=args.using_hard_prompt,
                                                  need_mask_token = args.using_mask_token
                                                    )
    if task == 'nli':
        tokenized_collator = TokenizedForNLI(args,
                                                 tokenizer,
                                                  token_idx=0,
                                                  sort_key=lambda x:x[2],
                                                  using_prompt=args.using_hard_prompt,
                                                  need_mask_token = args.using_mask_token
                                                    )
    return tokenized_collator
 
def get_model(method, tokenizer, args, pos_tokens, neg_tokens):
    mask_token = tokenizer.mask_token_id
    if method == 'our' :
        model = GeneratedPrompt(args,
                                mask_token_id=mask_token, 
                                positive_token_ids = pos_tokens, 
                                negative_token_ids = neg_tokens,
                                with_answer_weights = args.with_answer_weights,
                                fine_tune_all=args.fine_tune_all
                                )
                            
    if method == 'ft':
        model = BertFineTune(args)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_len", default=10, type=int)
    parser.add_argument("--eval_every", default=5, type=int)
    parser.add_argument("--print_every", default=5, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--shot_num", default=64, type=int)
    parser.add_argument("--n_prompts", default=8, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--seed", default=200, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--clip", default=15, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.33, type=float)
    parser.add_argument("--backbone", default='roberta', type=str)
    parser.add_argument("--task", default='sentiment', type=str)
    parser.add_argument("--data_path", default='process', type=str)
    parser.add_argument("--dataset", default='sst-2', type=str)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--optimizer", default='AdamW', type=str)
    parser.add_argument("--method", default='our', type=str)
    parser.add_argument("--pretrain_prompt_type", default='hard', type=str)
    parser.add_argument("--fine_tune_all", default= False, type=bool)
    parser.add_argument("--using_hard_prompt", default = True, type=bool)
    parser.add_argument("--using_cuda", default= True, type=bool)
    parser.add_argument("--with_answer_weights", default= True, type=bool)
    parser.add_argument("--using_verbalizer", default= True, type=bool)
    parser.add_argument("--using_mask_token", default= True, type=bool)

    args = parser.parse_args()

    if args.using_verbalizer:
        positive_words , negative_words = get_template(args.task)
    
    if args.backbone == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    
    pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1] 
    neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1] 
    res = []


    set_seed(args.seed)
    formatted_string =f"{args.method},{args.shot_num},roberta-bsz{args.batch_size}lr{args.learning_rate},seed-{args.seed},data-{args.dataset}"
    print(formatted_string)

    if args.using_cuda and torch.cuda.is_available:
        torch.cuda.empty_cache()

    train_data, val_data, test_data = get_dataset(args, args.data_path)
    # data= get_data(args, args.data_path)
    # train_data,val_data, test_data = split_data(args,data)

    tokenized_collator = get_collate(args.task,tokenizer,args)

    model = get_model(args.method, tokenizer, args, pos_tokens, neg_tokens)
    
    train_iter = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size, 
                            collate_fn=tokenized_collator)
    val_iter = DataLoader(dataset = val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=tokenized_collator)
    test_iter = DataLoader(dataset=test_data,
                            batch_size=args.batch_size, 
                            collate_fn=tokenized_collator)

    trainer = Trainer(args)

    best_res, best_model = trainer.train(args,
                                        train_iter = train_iter, 
                                        val_iter = val_iter,
                                        model = model, 
                                        batch_size = args.batch_size)
    model.load_state_dict(best_model) 
        
    test_loss, test_metrics = trainer.evaluate(args,
                                                model= model,
                                                eval_iter= test_iter,
                                                step=1000) 

    print("------------------------------------------")
    print("-Test: ")
    

    print_measures(test_loss, test_metrics)
    r = [test_loss.item()]
    r.extend([x for x in test_metrics.values()])
    res.append(r)

    with open("new_all/all_result_mr.csv", 'a+') as f:
        save_str = ",".join([str(x) for x in test_metrics.values()])
        f.write(f"{args.method},{args.shot_num},roberta-bsz{args.batch_size}lr{args.learning_rate},seed-{args.seed},data-{args.dataset}," + save_str +"\n")

# res = np.array(res).T.tolist()  # type: ignore
# for r in res:
#     r.remove(max(r))
#     r.remove(min(r))

# merge_res = [np.mean(x) for x in res]
# with open("average_10_reslit.csv", 'a+') as f:
#     save_str = ",".join([str(x) for x in merge_res])
#     f.write(f"{args.method},{args.shot_num},roberta-bsz{args.batch_size}lr{args.learning_rate},seed-{args.seed},data-{args.dataset}," + save_str +"\n")


if __name__ == "__main__":
    main()
