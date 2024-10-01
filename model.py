import torch
from torch import nn
import copy
from torch.nn import functional as F
from transformers import RobertaModel, BertModel, AutoModelForMaskedLM, RobertaPreTrainedModel, RobertaConfig,AutoModelForSequenceClassification,BertConfig
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from transformers import BertTokenizer
class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args, hidden_size: int = 384):  # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out_proj = nn.Linear(hidden_size, args.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = torch.mean(features,dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GeneratedPrompt(nn.Module):
    def __init__(self,
                 args,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_answer_weights=True,
                 fine_tune_all=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        if not fine_tune_all:  # freeze the pretrained encoder
            for param in self.roberta.base_model.parameters():  # type: ignore
                param.requires_grad = False
        bert_config =RobertaConfig.from_pretrained('roberta-base')
        self.vocab_size = bert_config.vocab_size
        self.mask_token_id = mask_token_id
        self.prompt_len = args.prompt_len
        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids

        self.prompt_embedding = nn.Parameter(torch.randn(args.prompt_len,args.hidden_size),requires_grad=True)
        self.prompt_attention = nn.Parameter(torch.ones(args.prompt_len))
        self.token_attention =nn.Parameter(torch.ones(1))

        self.encoder = TransformerDecoder(args.hidden_size, depth = 1, num_heads=8, mlp_ratio=0.5)
        self.decoder = TransformerEncoder(args.hidden_size, depth = 1, num_heads=8, mlp_ratio=0.5)

        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=bert_config.layer_norm_eps)
        self.contrastive_head = ContrastiveHead(temperature=0.1)

        self.masklm = RobertaLMHead(args, bert_config)
        self.classfier = Classifier(args)
        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        
    def forward(self, input_ids, attention_mask,labels):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        prompt_embeds = self.prompt_embedding.unsqueeze(0).expand(batch_size,-1, -1)
            # batch_size, num_learnable_token, hidden_size
        roberta_outputs = self.roberta(input_ids, attention_mask,output_hidden_states=True)
        sequence_output = roberta_outputs

        sequence_output_decode = self.encoder(sequence_output) 
        x = torch.cat((prompt_embeds, sequence_output_decode),dim=1)
        x = self.decoder(x)
        x = self.layer_norm(x)
        prompt_embeds = x[:, 0:self.prompt_len]
        cls_prompt_emb = torch.mean(prompt_embeds,dim=1)
        
        # loss_contra = self.contrastive_head(similarity, select)
        loss_contra = self.sclLoss(cls_prompt_emb,labels)
    
        logits = self.masklm(sequence_output_decode)
        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()

        mask_logits = mask_logits.sum(dim=1).squeeze(
            1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                                      self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                                      self.negative_token_ids] * negative_weight
        
        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([positive_logits, negative_logits], dim=1)

        return cls_logits , loss_contra , cls_prompt_emb.detach()
    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))
    
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


class ContrastiveHead(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity, select):
        B = similarity.size(0)
        losses = torch.zeros(B).to(similarity.device)
        for i in range(B):
            pos = torch.masked_select(similarity[i], select[i] == 1)
            neg = torch.masked_select(similarity[i], select[i] == 0)
            pos = torch.mean(pos, dim=0, keepdim=True)
            logits = torch.cat((pos, neg)).reshape(1, -1)
            label = torch.zeros(1, dtype=torch.long).to(similarity.device)
            losses[i] = self.criterion(logits/self.temperature, label)
        loss = losses.mean()
        return loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., with_contrast=False, finetune=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.with_contrast = with_contrast
        self.finetune = finetune
        if self.with_contrast:
            self.norm1_contrast = norm_layer(dim)
            self.norm2_contrast = norm_layer(dim)
            self.mlp_contrast = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.attn_contrast = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        if self.with_contrast:
            cls = x[:, :10]
            vis_x = x[:, 10:]

            cls = self.norm1_contrast(cls)
            vis_x = self.norm1(vis_x)
            if self.finetune:
                cls = cls + self.drop_path(self.attn_contrast(cls, vis_x))
            else:
                cls = cls + self.drop_path(self.attn_contrast(cls, vis_x.detach()))
                # cls = cls + self.drop_path(self.attn_contrast(cls, vis_x))
            vis_x = vis_x + self.drop_path(self.attn(vis_x))

            cls = self.norm2_contrast(cls)
            vis_x = self.norm2(vis_x)
            cls = cls + self.drop_path(self.mlp_contrast(cls))
            vis_x = vis_x + self.drop_path(self.mlp(vis_x))

            x = torch.cat((cls, vis_x), dim=1)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=3, mlp_ratio=0.5, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, finetune=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, with_contrast=True, finetune=finetune,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x



class CircleLoss(nn.Module):
    def __init__(self,
                 margin: float = 0.4,
                 gamma: float = 128,
                 k: float = 1,
                 distance_function='cos') -> None:
        super(CircleLoss, self).__init__()
        self.m = margin
        self.gamma = gamma
        self.k = k
        self.soft_plus = nn.Softplus()
        if distance_function == 'cos':
            self.dist_fcn = lambda X: X @ X.transpose(1, 0)
        else:
            raise NotImplementedError

    def forward(self, features, labels):
        sim = self.dist_fcn(features).view(-1)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos = mask.triu(diagonal=1).view(-1)
        neg = mask.logical_not().triu(diagonal=1).view(-1)
        sp = sim[pos]
        sn = sim[neg]
        ap = (1 / self.k) * torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, args,config, hidden_size: int = 300):   # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)  # type: ignore
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=3, mlp_ratio=0.5, qkv_bias=False, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, with_contrast=False,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x )

        return x
