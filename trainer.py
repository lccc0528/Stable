from typing import ContextManager, List
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
import numpy as np
from utils import to_cuda, compute_acc, compute_measures, print_measures
from forward_calculator import FinetuneFoward
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ray import tune
class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.loss_fn = cross_entropy
        self.metrics_fn = compute_measures
    
        self.foward_calculator = FinetuneFoward(loss_fn=cross_entropy, metrics_fn=compute_measures)
    def train(self, 
              args,
              model: nn.Module, 
              train_iter: DataLoader, 
              val_iter: DataLoader,
              batch_size: int,
              class_balance: bool = False):
        model.train()
        if self.args.using_cuda and torch.cuda.is_available():
            model.to(device = torch.device(args.device))

        self.optim = AdamW(model.parameters(), 
                          lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay
                          )

        best_res = [0, {"accuracy": 0, 
                        "bi_precision": 0, "bi_recall": 0, "bi_f1": 0, 
                        "micro_precision": 0, "micro_recall": 0, "micro_f1": 0, 
                        "macro_precision": 0, "macro_recall": 0, "macro_f1": 0, 
                        "weighted_precision": 0, "weighted_recall": 0, "weighted_f1": 0,
                        "auc": 0}]
        best_model = None
        early_stop_cnt = 0
        step = 0
        label_cnt = {"real": 0, "fake":0}
        logits, labels = [], []  # for print
        
        for epoch in range(self.args.epoch):
            for batch in train_iter:
                logit, loss,metrics,loss_cl= self.foward_calculator(args,batch, model, step, cuda=self.args.using_cuda, class_balance=class_balance)
                loss.backward()
              
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.args.clip)
                self.optim.step()
                self.optim.zero_grad()

                logits.append(logit)  # type: ignore
                labels.append(batch[2])  # type: ignore
                  
                if step > 0 and step % self.args.print_every == 0:
                    print_logits = torch.cat(logits, dim=0) # type: ignore
                    print_labels = torch.cat(labels, dim=0).to(device = torch.device(args.device)) # type: ignore
                    #print_loss, print_metrics = self.loss_fn(print_logits, print_labels), self.metrics_fn(print_logits, print_labels)  # type: ignore
                    print_loss, print_metrics =loss, metrics
                    print(f"--Epoch {epoch}, Step {step}, Loss {print_loss}")
                    print_measures(print_loss, print_metrics)
                    logits, labels = [], []
                
                if epoch > 0 and step % self.args.eval_every == 0:
                    avg_loss, avg_metrics = self.evaluate(args, step, model, val_iter)
                    res = [avg_loss, avg_metrics]
                    if avg_metrics['bi_f1'] > best_res[1]['bi_f1']:   # type: ignore
                        best_res = res
                        best_model = model.cpu().state_dict()

                        model.to(device = torch.device(args.device))
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    print("--Best Evaluation: ")
                    print_measures(best_res[0], best_res[1])
                    model.train()
        
                step += 1
         
        if best_model is not None:  # type: ignore
            # return best_res,merged_state_dict
            return best_res ,best_model 
        # return best_res, merged_state_dict
        return best_res, model.cpu().state_dict() # type: ignore

    # eval func
    def evaluate(self, args, step ,model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        logits, labels, masks = [], [], []
        for step, batch in enumerate(eval_iter):
            logit, loss_contra,_= self.foward_calculator(args, batch, model, step ,cuda=self.args.using_cuda, evaluate=True)
       
            logits.append(logit)
            labels.append(batch[2])
            
        logits = torch.cat(logits, dim=0).to(device = torch.device(args.device)) # type: ignore
        labels = torch.cat(labels, dim=0).to(device = torch.device(args.device))  # type: ignore
        loss, metrics = self.loss_fn(logits, labels), self.metrics_fn(logits, labels)  # type: ignore

        loss = loss +  loss_contra #
        #loss = loss_contra
        print("--Evaluation:")

        print_measures(loss, metrics)
        if save_file != "":
            results = [save_title, avg_loss, avg_metrics.values()]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return loss, metrics  # type: ignore
    
    
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
