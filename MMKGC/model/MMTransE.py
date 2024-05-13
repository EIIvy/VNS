import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .Model import Model


class MMTransE(Model):

    def __init__(self, args, img_emb,rel_emb, p_norm=1, norm_flag=True, test_mode='lp'):
        super(MMTransE, self).__init__()
        self.args = args
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.test_mode = test_mode
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )

        self.ent_embeddings = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_embeddings = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        # nn.init.uniform_(tensor=self.ent_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        # nn.init.uniform_(tensor=self.rel_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

        # 新增的投影矩阵和图像embeddings
        self.img_proj = nn.Linear(self.args.img_dim, self.args.emb_dim)
    
        if self.args.random:
            self.img_embeddings = nn.Embedding(self.args.num_ent, self.args.img_dim)
        else:
            self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
            self.img_rel_embeddings = nn.Embedding.from_pretrained(rel_emb).requires_grad_(True)
        #self.beta = beta
        #new 2是1-2，new3是1-3
        # self.log_file = open('{}.txt'.format(time.time()), 'w')

        #if margin is None or epsilon is None:
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        #else:
        #     self.embedding_range = nn.Parameter(
        #         torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
        #     )
        #     nn.init.uniform_(
        #         tensor=self.ent_embeddings.weight.data,
        #         a=-self.embedding_range.item(),
        #         b=self.embedding_range.item()
        #     )
        #     nn.init.uniform_(
        #         tensor=self.rel_embeddings.weight.data,
        #         a=-self.embedding_range.item(),
        #         b=self.embedding_range.item()
        #     )

        # if margin is not None:
        #     self.margin = nn.Parameter(torch.Tensor([margin]))
        #     self.margin.requires_grad = False
        #     self.margin_flag = True
        # else:
        self.margin_flag = False
    
    def _calc1(self, h, t, r):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)   #[128, 1, 500]
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        
        score = (h + r) - t
        #score = self.margin.item() - torch.norm(score, self.p_norm, -1)
        score = self.margin.item() - torch.norm(score,self.p_norm, -1)
        return score

    # def _calc(self, h, t, r, mode):
    #     if self.norm_flag:
    #         h = F.normalize(h, 2, -1)
    #         r = F.normalize(r, 2, -1)
    #         t = F.normalize(t, 2, -1)
    #     if mode != 'normal':
    #         h = h.view(-1, r.shape[0], h.shape[-1])
    #         t = t.view(-1, r.shape[0], t.shape[-1])
    #         r = r.view(-1, r.shape[0], r.shape[-1])
    #     if mode == 'head_batch':
    #         score = h + (r - t)
    #     else:
    #         score = (h + r) - t
    #     score = torch.norm(score, self.p_norm, -1).flatten()
    #     return score
    def forward(self, triples, negs=None, mode='single'):
        s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb = self.tri2emb(triples, mode, negs)
        
        score = (
                self._calc1(s_h_emb, s_t_emb, s_r_emb)
                + self._calc1(h_img_emb, t_img_emb, s_r_emb)
                + self._calc1(h_img_emb, s_t_emb, s_r_emb)
                + self._calc1(s_h_emb, t_img_emb, s_r_emb)
                + self._calc1(s_h_emb, s_t_emb, r_img_emb)
                + self._calc1(h_img_emb, t_img_emb, r_img_emb)
                + self._calc1(h_img_emb, s_t_emb, r_img_emb)
                + self._calc1(s_h_emb, t_img_emb, r_img_emb)
                # + self._calc(h + h_img_emb, t + t_img_emb, r, mode)
        ) / 8
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    
    def get_score(self, batch, mode):
        triples = batch["positive_sample"]
        s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb= self.tri2emb(triples, mode)
        
        score = (
                self._calc1(s_h_emb, s_t_emb, s_r_emb)
                + self._calc1(h_img_emb, t_img_emb, s_r_emb)
                + self._calc1(h_img_emb, s_t_emb, s_r_emb)
                + self._calc1(s_h_emb, t_img_emb, s_r_emb)
                + self._calc1(s_h_emb, s_t_emb, r_img_emb)
                + self._calc1(h_img_emb, t_img_emb, r_img_emb)
                + self._calc1(h_img_emb, s_t_emb, r_img_emb)
                + self._calc1(s_h_emb, t_img_emb, r_img_emb)

                # + self._calc(h + h_img_emb, t + t_img_emb, r, mode)
        ) / 8
        if self.margin_flag:
            return self.margin - score
        else:
            return score


    def tri2emb(self, triples, mode="single", negs=None):
        if mode == 'single':
            batch_h = triples[:, 0]
            batch_t = triples[:, 2]
            batch_r = triples[:, 1]
            h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
            s_h_emb = self.ent_embeddings(h_ent).unsqueeze(1)
            s_r_emb = self.rel_embeddings(batch_r).unsqueeze(1)
            s_t_emb = self.ent_embeddings(t_ent).unsqueeze(1)
            h_img_emb = self.img_proj(self.img_embeddings(h_img)).unsqueeze(1)
            t_img_emb = self.img_proj(self.img_embeddings(t_img)).unsqueeze(1)
        elif mode =='head-batch'or mode == 'head_predict':
            if negs is None:
                s_h_emb = self.ent_embeddings.weight.data.unsqueeze(0)
                h_img_emb = self.img_proj(self.img_embeddings.weight.data).unsqueeze(0)
            else:
                s_h_emb = self.ent_embeddings(negs)
                h_img_emb = self.img_proj(self.img_embeddings(negs))
            batch_t = triples[:, 2]
            batch_r = triples[:, 1]
            t_ent, t_img =  batch_t, batch_t
            
            s_r_emb = self.rel_embeddings(batch_r).unsqueeze(1)
            s_t_emb = self.ent_embeddings(t_ent).unsqueeze(1)
            t_img_emb = self.img_proj(self.img_embeddings(t_img)).unsqueeze(1)
        elif mode == 'tail-batch' or mode == 'tail_predict':
            if negs is None:
                s_t_emb = self.ent_embeddings.weight.data.unsqueeze(0)  #[1, 15404, 500]
                t_img_emb = self.img_proj(self.img_embeddings.weight.data).unsqueeze(0) #[1, 15404, 500]
            else:
                s_t_emb = self.ent_embeddings(negs)
                t_img_emb = self.img_proj(self.img_embeddings(negs))
            batch_h = triples[:, 0]
            batch_r = triples[:, 1]
            h_ent, h_img =  batch_h, batch_h
            
            s_r_emb = self.rel_embeddings(batch_r).unsqueeze(1)  #[128, 1, 500]
            s_h_emb = self.ent_embeddings(h_ent).unsqueeze(1)  #[128, 1, 500]
            h_img_emb = self.img_proj(self.img_embeddings(h_img)).unsqueeze(1) #[128, 1, 500]
        r_img_emb = self.img_proj(self.img_rel_embeddings(triples[:, 1])).unsqueeze(1)

        return s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb


    # def forward(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
    #     mode = data['mode']
    #     h = self.ent_embeddings(h_ent)
    #     t = self.ent_embeddings(t_ent)
    #     r = self.rel_embeddings(batch_r)
    #     h_img_emb = self.img_proj(self.img_embeddings(h_img))
    #     t_img_emb = self.img_proj(self.img_embeddings(t_img))
    #     score = (
    #             self._calc(h, t, r, mode)
    #             + self._calc(h_img_emb, t_img_emb, r, mode)
    #             + self._calc(h_img_emb, t, r, mode)
    #             + self._calc(h, t_img_emb, r, mode)
    #             # + self._calc(h + h_img_emb, t + t_img_emb, r, mode)
    #     ) / 4
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score
    


    # def regularization(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     regul = (torch.mean(h ** 2) +
    #              torch.mean(t ** 2) +
    #              torch.mean(r ** 2)) / 3
    #     return regul

    # def cross_modal_score_ent2img(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
    #     mode = data['mode']
    #     h = self.ent_embeddings(h_ent)
    #     t = self.ent_embeddings(t_ent)
    #     r = self.rel_embeddings(batch_r)
    #     h_img_emb = self.img_proj(self.img_embeddings(h_img))
    #     t_img_emb = self.img_proj(self.img_embeddings(t_img))
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t_img_emb, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score

    # def score_ent2ent(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     mode = data['mode']
    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score

    # def score_vis2vis(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     mode = data['mode']
    #     h = self.img_proj(self.img_embeddings(batch_h))
    #     t = self.img_proj(self.img_embeddings(batch_t))
    #     r = self.rel_embeddings(batch_r)
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score
    
    # def score_vis2ent(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     mode = data['mode']
    #     h = self.img_proj(self.img_embeddings(batch_h))
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score
    
    # def score_all2ent(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     mode = data['mode']
    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     h_img = self.img_proj(self.img_embeddings(batch_h))
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t, r, mode) + self._calc(h_img, t, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score
    
    # def score_all2vis(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     mode = data['mode']
    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     h_img = self.img_proj(self.img_embeddings(batch_h))
    #     t_img = self.img_proj(self.img_embeddings(batch_t))
    #     # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
    #     score = self._calc(h, t_img, r, mode) + self._calc(h_img, t_img, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score

    # def predict(self, data):
    #     if self.test_mode == 'cmlp':
    #         score = self.cross_modal_score_ent2img(data)
    #     else:
    #         score = self.forward(data)
    #     if self.margin_flag:
    #         score = self.margin - score
    #         return score.cpu().data.numpy()
    #     else:
    #         return score.cpu().data.numpy()

    # def set_test_mode(self, new_mode):
    #     self.test_mode = new_mode

    # def get_rel_rank(self, data):
    #     head, tail, rel = data
    #     h_img_emb = self.img_proj(self.img_embeddings(head))
    #     t_img_emb = self.img_proj(self.img_embeddings(tail))
    #     relations = self.rel_embeddings.weight
    #     h = h_img_emb.reshape(-1, h_img_emb.shape[0]).expand((relations.shape[0], h_img_emb.shape[0]))
    #     t = t_img_emb.reshape(-1, t_img_emb.shape[0]).expand((relations.shape[0], t_img_emb.shape[0]))
    #     scores = self._calc(h, t, relations, mode='normal')
    #     ranks = torch.argsort(scores)
    #     rank = 0
    #     for (index, val) in enumerate(ranks):
    #         if val.item() == rel.item():
    #             rank = index
    #             break
    #     return rank + 1
