from typing import Any
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from utils import to_cuda, seq_mask_by_lens
# import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
class FinetuneFoward():
    def __init__(self, loss_fn, metrics_fn) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
    def compute_forward(self, args ,batch, model, step, cuda:bool = False, evaluate:bool = False,class_balance:bool = False):
        input_ids, attention_mask, labels, seq_lens = batch
        batch_sz = len(labels)
        if cuda and torch.cuda.is_available():  # type: ignore
            input_ids, attention_mask, labels= to_cuda(args,data=(input_ids, attention_mask, labels))
            model = model.to(device = torch.device(args.device))
        if evaluate:
            with torch.no_grad():
                #logits = model(input_ids, attention_mask)
                # logits,loss_contra= model(input_ids, attention_mask,labels)
                logits,loss_contra,prompt_show= model(input_ids, attention_mask,labels)  #for t-SNE
                prompt_show = prompt_show.cpu().numpy()   #for t-SNE
                return logits ,loss_contra, prompt_show
        else:
            #logits = model(input_ids, attention_mask)
            # logits, loss_contra= model(input_ids, attention_mask,labels)
            logits, loss_contra, _= model(input_ids, attention_mask,labels)# for t-SNE
                    
        # prediction = logits.max(1)[1]
        cnt = Counter(labels.cpu().tolist())  # type: ignore
        weight = [0.5, 0.5]
        weight[0] = 1 - cnt[0] / sum(cnt.values())  # type: ignore
        weight[1] = 1 - cnt[1] / sum(cnt.values())  # type: ignore
        loss = self.loss_fn(input=logits, target=labels, weight=torch.tensor(weight).to(device = torch.device(args.device)))
        #loss = self.loss_fn(input=logits, target=labels)
        if args.method == 'cptuning':
            loss = 0.5 * loss + 0.5 * loss_contra
        else:
            loss =  loss + loss_contra 
        #loss =loss_contra
        metrics = self.metrics_fn(logits, labels)
        
        return logits, loss, metrics ,loss_contra
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_forward(*args, **kwds)
    
    def sclLoss(self, pooled, labels):
        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
        mask = mask - torch.diag(torch.diag(mask))
        cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
        cos_loss = -torch.log(cos_loss + 1e-5)
        cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
        cos_loss = cos_loss.mean()
        return cos_loss