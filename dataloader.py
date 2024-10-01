import os
import json
import operator
import csv
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
import re

class FakeNewsNetDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path

        self.texts_tweet, self.labels, self.seq_lens_tweet = self.read_data(data_path)

    def read_data(self, data_path):

        labels_tweet, texts_tweet, seq_lens  = [], [], []
        texts_temp  = []

        with open(data_path,'r',encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            tweet = " "
            for row in reader: 
                if row['id'] == row['reply_id']:
                    text = " ".join(texts_temp)
                    if len(text) > 512:
                        text = text[:512]
                    texts_temp =  []
                    if row['label'] == 'real':
                        labels_tweet.append(1)
                    else:
                        labels_tweet.append(0)
                    texts_tweet.append((tweet,text))
                    seq_lens.append(len(tweet + text))
                    tweet = row['text']
                else:
                    texts_temp.append(row['text'])
                # text = row['texts']
                # label = row['labels']
                # seq_len = row['seq_lens']
                # result_tuple = eval(text)
                # texts_tweet.append(result_tuple)
                # labels_tweet.append(int(label))
                # seq_lens.append(int(seq_len))
            
            texts_tweet.append((tweet,text))
            seq_lens.append(len(tweet + text))
            texts_tweet.pop(0)
            seq_lens.pop(0)            
        return texts_tweet, labels_tweet, seq_lens
    
    def __getitem__(self, idx):

        return self.texts_tweet[idx], self.labels[idx], self.seq_lens_tweet[idx]

    def __len__(self):

        return len(self.labels)
    

class PHEMEDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path
        self.texts_tweet, self.labels, self.seq_lens_tweet= self.read_data(data_path)
    

    def read_data(self, data_path):
        max_len = 512
        labels_tweet, texts_tweet, seq_lens  = [], [], []
        texts_temp  = []
        
        with open(data_path,'r',encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            text_num = 0
            for row in reader:
                id = row['id']
                if(text_num ==0):
                    id_now =id
                    if(row['veracityLabel']== 'true'):
                        labels_tweet.append(1)
                    elif(row['veracityLabel']== 'false'):
                        labels_tweet.append(0)
                    else:
                        continue

                if (id == id_now):
                    if(row['post_comment']=='text'):
                        texts_temp.insert(0,row['text'])
                        text_num+=1               
                    else:
                        texts_temp.append(row['text'])
                        text_num+=1
                else:
                    if(len(texts_temp)>=2 and texts_temp[0] == texts_temp[1]):
                        texts_temp.pop(1)
                    text = " ".join(texts_temp[1:])
                    if len(texts_temp[0] + text) > 512:
                        text = text[:512-len(texts_temp[0])]
                    texts_tweet.append((texts_temp[0],text))
                    seq_lens.append(len(texts_temp[0] + text))

                    id_now =id
                    text_num = 0
                    texts_temp =[]
                    if(row['veracityLabel']== 'true'):
                        labels_tweet.append(1)
                    elif(row['veracityLabel']== 'false'):
                        labels_tweet.append(0)
                    else:
                        continue
                    texts_temp.append(row['text'])
                    text_num = 1

            if(len(texts_temp)>=2 and texts_temp[0] == texts_temp[1]):
                texts_temp.pop(1)
                text = " ".join(texts_temp[1:])
            if len(texts_temp[0] + text) > 512:
                text = text[:512-len(texts_temp[0])]
            texts_tweet.append((texts_temp[0],text))
            seq_lens.append(len(texts_temp[0] + text))
                
            return texts_tweet, labels_tweet, seq_lens
    
    def __getitem__(self, idx):
        return self.texts_tweet[idx], self.labels[idx], self.seq_lens_tweet[idx]

    def __len__(self):
        return len(self.labels)


class TokenizedForFakeNews(): 
    def __init__(self, args, tokenizer, token_idx, sort_key, using_prompt=True ,need_mask_token = True):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.sort_key = sort_key  # sort key
        self.tokenizer = tokenizer
        

        self.tag_tweet = "<claim>"
        self.tag_reply = "<replys>"
        self.tag_tweet_ids = self.tokenizer(self.tag_tweet, padding=False, return_tensors="pt")['input_ids']
        self.tag_tweet_ids = self.tag_tweet_ids[0][1:-1]  # ignore <cls> and <\s>
        self.tag_reply_ids = self.tokenizer(self.tag_reply, padding=False, return_tensors="pt")['input_ids']
        self.tag_reply_ids = self.tag_reply_ids[0][1:-1]  # ignore <cls> and <\s>
        
        if using_prompt:
            self.prefix_prompt = "Here is a piece of claim with <mask> information ."
            # "Here is a piece of claim with <mask> information . " 
            # "The content of this statement is <mask> ."
            # "This statement's message is <mask>."
            # "What this declaration says is <mask>."
            # "The essence of this proclamation is <mask>."
            # "This announcement conveys a <mask> message."
        else:
            self.prefix_prompt = "<mask>"

        # self.prefix_prompt = "Here is a piece of news with [MASK] information . "
        self.postfix_prompt = " This article is <mask> news ."
        self.unused_ids = torch.tensor([2023]* 10) 
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.eos_ids = torch.tensor([self.prefix_ids[0][-1]])
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        
        self.tag_attention_tweet_mask = torch.ones(3)
        self.tag_attention_replys_mask = torch.ones(4)
        
        self.cls_id = torch.tensor([self.prefix_ids[0]])
        
        if using_prompt and need_mask_token:
            self.prefix_ids = torch.cat([self.cls_id, self.prefix_ids[1:]], dim=0)

        elif need_mask_token:
            self.used_ids = torch.cat([self.cls_id,self.unused_ids], dim=0)
            self.prefix_ids = torch.cat([self.used_ids,self.prefix_ids[1:]], dim=0)         
        else:
            self.prefix_ids = self.cls_id

        self.add_len = int(len(self.prefix_ids)) + int(len(self.eos_ids))
        self.add_attention_pre_mask = torch.ones(len(self.prefix_ids))
        self.add_attention_post_mask = torch.ones(len(self.eos_ids))
        self.max_len = 512 - self.add_len -10
        
    def _collate_fn(self, batch):
        ret = []
        batch.sort(key=self.sort_key, reverse=True)  
        
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # max_len = max(len(sentence.split()) for sentence in samples)
                input_ids_lst, attention_mask_lst = [], []
                for (sample_tweet,sample_reply) in samples:
                    inputs_tweet = self.tokenizer(sample_tweet,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    inputs_reply = self.tokenizer(sample_reply,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    if len(inputs_tweet) == 2:  # roberta
                        input_ids_tweet, attention_mask_tweet = inputs_tweet
                        input_ids_reply, attention_mask_reply = inputs_reply
                    elif len(inputs_tweet) == 3:  # bert
                        input_ids_tweet, _, attention_mask_tweet = inputs_tweet
                        input_ids_reply, _, attention_mask_reply = inputs_reply
                    else:
                        raise RuntimeError

                    input_ids_tweet = input_ids_tweet[0][1:-1]
                    attention_mask_tweet = attention_mask_tweet[0][1:-1]
                    input_ids_reply = input_ids_reply[0][1:-1]
                    attention_mask_reply = attention_mask_reply[0][1:-1]

                    input_ids = torch.cat([self.tag_tweet_ids, input_ids_tweet, self.tag_reply_ids, input_ids_reply], dim=0)
                    attention_mask = torch.cat([self.tag_attention_tweet_mask,attention_mask_tweet,self.tag_attention_replys_mask,attention_mask_reply], dim=0)
                    if len(input_ids) > (self.max_len):
                        input_ids = input_ids[:self.max_len]    
                        attention_mask = attention_mask[:self.max_len] 

                    input_ids = torch.cat([self.prefix_ids, input_ids, self.eos_ids], dim=0)
                    attention_mask = torch.cat([self.add_attention_pre_mask, attention_mask, self.add_attention_post_mask], dim=0)
                    
                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)
    

                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
               
                ret.append(input_ids)
                ret.append(attention_mask)       

            else:
                ret.append(torch.tensor(samples))
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)
    

class AllDataset(Dataset):
    def __init__(self, data_path, task) -> None:
        super().__init__()

        self.data_path = data_path
        self.texts, self.labels, self.seq_lens= self.read_data(data_path, task)

    def read_data(self, data_path,task):
        max_len = 512
        labels, texts, seq_lens  = [], [], []

        with open(data_path,'r',encoding='UTF-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if(row['labels']== '1'):
                    labels.append(1)
                elif(row['labels']== '0'):
                    labels.append(0)
                else:
                    continue
                if task == 'fakenews' or task == 'nli':
                    sentence =  eval(row['texts'])
                else:
                    sentence = row['texts']
                # if len(sentence) > 512:
                #     sentence = sentence[:512]
                
                seq_len = int(row['seq_lens'])
                texts.append(sentence)
                seq_lens.append(seq_len)
        return texts, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.labels)


class SentimentDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path
        self.texts, self.labels, self.seq_lens= self.read_data(data_path)

    def read_data(self, data_path):
        max_len = 512
        labels, texts, seq_lens  = [], [], []

        with open(data_path,'r',encoding='UTF-8') as file:
            for line in file:
                row = line.strip().split('\t')
                if(row[1]== '1'):
                    labels.append(1)
                elif(row[1]== '0'):
                    labels.append(0)
                else:
                    continue
                sentence = row[0]

                if len(sentence) > 512:
                    sentence = sentence[:512]
                
                seq_len = len(sentence)
                texts.append(sentence)
                seq_lens.append(seq_len)
            return texts, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.labels)


class TokenizedForSentiment(): 
    def __init__(self, args, tokenizer, token_idx, sort_key, using_prompt=True,need_mask_token =True):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.sort_key = sort_key  # sort key
        self.tokenizer = tokenizer  
        self.unused_ids = torch.tensor([2023]*10) 
        if using_prompt and need_mask_token:
            self.prefix_prompt = "Here is a piece of review with <mask> sentiment ."
        #self.prefix_prompt = "Here is a piece of review with <mask> sentiment ."
        # self.prefix_prompt = "The sentiment of the review is <mask> ."
        # self.prefix_prompt = "The outlook portrayed in this appraisal is <mask> ."
        # self.prefix_prompt = "The emotional tone of this testimonial is <mask> ."    
        # self.prefix_prompt = "The sentiment of this critique is <mask> ."
        # self.prefix_prompt = "The overall feeling conveyed by this evaluation is <mask> ."
        # self.prefix_prompt = "The mood expressed in this assessment is <mask> ."
        else:
            self.prefix_prompt = "<mask>"

        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.eos_ids = torch.tensor([tokenizer.eos_token_id])
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        self.cls_id = torch.tensor([tokenizer.cls_token_id])
        
        if using_prompt and need_mask_token:
    
            self.prefix_ids = torch.cat([self.cls_id,self.prefix_ids[1:]], dim=0)
        elif need_mask_token:
            self.used_ids = torch.cat([self.cls_id,self.unused_ids], dim=0)
            self.prefix_ids = torch.cat([self.used_ids,self.prefix_ids[1:]], dim=0)
        else:
            self.prefix_ids = self.cls_id

        self.add_len = int(len(self.prefix_ids)) + 1
        self.add_attention_pre_mask = torch.ones(len(self.prefix_ids))
        self.add_attention_post_mask = torch.ones(len(self.eos_ids))
        self.max_len = 512 - self.add_len - 2
        
    def _collate_fn(self, batch):
        ret = []
        batch.sort(key=self.sort_key, reverse=True)  
        
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # max_len = max(len(sentence.split()) for sentence in samples)
                input_ids_lst, attention_mask_lst = [], []
                for text in samples:
                    text = self.tokenizer(text,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
        
                    if len(text) == 2:  # roberta
                        input_ids_text, attention_mask_text = text
                    elif len(text) == 3:  # bert
                        input_ids_text, _, attention_mask_text = text
                    else:
                        raise RuntimeError

                    input_ids = input_ids_text[0][1:-1]
                    attention_mask= attention_mask_text[0][1:-1]

                    if len(input_ids) > (self.max_len):
                        input_ids = input_ids[:self.max_len]    
                        attention_mask = attention_mask[:self.max_len] 

                    input_ids = torch.cat([self.prefix_ids, input_ids, self.eos_ids], dim=0)
                    attention_mask = torch.cat([self.add_attention_pre_mask, attention_mask, self.add_attention_post_mask], dim=0)
                    
                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)
    
                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
                ret.append(input_ids)
                ret.append(attention_mask)   

            else:
                ret.append(torch.tensor(samples))
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)

class QNLIDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path
        self.texts, self.labels, self.seq_lens= self.read_data(data_path)

    def read_data(self, data_path):
        max_len = 512
        labels, texts, seq_lens  = [], [], []

        with open(data_path,'r',encoding='UTF-8') as file:
            for line in file:
                row = line.strip().split('\t')
                if(row[3]== 'entailment'):
                    labels.append(1)
                elif(row[3]== 'not_entailment'):
                    labels.append(0)
                else:
                    continue
                question = row[1]
                sentence = row[2]
          
                seq_len = len(sentence + question)
                texts.append((question,sentence))
                seq_lens.append(seq_len)
            return texts, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.labels)

class TokenizedForNLI(): 
    def __init__(self, args, tokenizer, token_idx, sort_key, using_prompt=True,need_mask_token =True):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.sort_key = sort_key  # sort key
        self.tokenizer = tokenizer  
        self.unused_ids = torch.tensor([2023]*10) 
        self.dataset = args.dataset
        self.method = args.method
    
        if using_prompt and need_mask_token:
            if args.dataset == 'rte':
                self.middle_prompt = "<mask> , I believe"
            else:
                self.middle_prompt = " <mask> "
        else:
            self.middle_prompt = "<mask>"
        
        self.middle_ids = self.tokenizer(self.middle_prompt, padding=False, return_tensors="pt")['input_ids']
        # self.tag_ids = self.tokenizer(self.tag, padding=False, return_tensors="pt")['input_ids']
        self.middle_ids = self.middle_ids[0][1:-1]
        self.cls_id = torch.tensor([tokenizer.cls_token_id])
        self.eos_ids = torch.tensor([tokenizer.eos_token_id])
        self.pad_ids = torch.tensor([tokenizer.pad_token_id])
        self.mask_ids = torch.tensor([tokenizer.mask_token_id])
        self.prefix_ids = self.cls_id

        self.add_attention_pre_mask = torch.ones(len(self.prefix_ids))
        self.add_attention_mid_mask = torch.ones(len(self.middle_ids))
        self.add_attention_post_mask = torch.ones(len(self.eos_ids))
 
        self.max_len = 512 - int(len(self.middle_ids)) - int(len(self.eos_ids)) -int(len(self.prefix_ids))

    def _collate_fn(self, batch):
        ret = []
        batch.sort(key=self.sort_key, reverse=True)  
        
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # max_len = max(len(sentence.split()) for sentence in samples)
                input_ids_lst, attention_mask_lst = [], []
                for (sample_tweet,sample_reply) in samples:
                    inputs_tweet = self.tokenizer(sample_tweet,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    inputs_reply = self.tokenizer(sample_reply,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    if len(inputs_tweet) == 2:  # roberta
                        input_ids_tweet, attention_mask_tweet = inputs_tweet
                        input_ids_reply, attention_mask_reply = inputs_reply
                    elif len(inputs_tweet) == 3:  # bert
                        input_ids_tweet, _, attention_mask_tweet = inputs_tweet
                        input_ids_reply, _, attention_mask_reply = inputs_reply
                    else:
                        raise RuntimeError
                   
                    input_ids_tweet = input_ids_tweet[0][1:]
                    attention_mask_tweet = attention_mask_tweet[0][1:]
                    input_ids_reply = input_ids_reply[0][1:]
                    attention_mask_reply = attention_mask_reply[0][1:]
                   
                    input_ids = torch.cat([input_ids_tweet,self.middle_ids,input_ids_reply],dim=0)
                    attention_mask = torch.cat([attention_mask_tweet,self.add_attention_mid_mask,attention_mask_reply],dim=0)

                    if len(input_ids) > (self.max_len):
                        input_ids = input_ids[:self.max_len]    
                        attention_mask = attention_mask[:self.max_len] 

                    input_ids = torch.cat([self.prefix_ids, input_ids, self.eos_ids], dim=0)
                    attention_mask = torch.cat([self.add_attention_pre_mask, attention_mask, self.add_attention_post_mask], dim=0)
                    
                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)
    

                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
               
                ret.append(input_ids)
                ret.append(attention_mask)       

            else:
                ret.append(torch.tensor(samples))
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)

    def __call__(self, batch):
        return self._collate_fn(batch)
