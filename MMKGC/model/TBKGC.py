import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class TBKGC(Model):

    def __init__(self, args, img_emb, rel_emb, p_norm=1,
                norm_flag=True,beta=None):
        super(TBKGC, self).__init__()

        self.args = args
        self.norm_flag = norm_flag
        self.p_norm = p_norm
       

        self.ent_embeddings = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_embeddings = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        # 新增的投影矩阵和图像embeddings
        self.img_proj = nn.Linear(self.args.img_dim, self.args.emb_dim)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.rel_img_embeddings = nn.Embedding.from_pretrained(rel_emb).requires_grad_(True)
        self.beta = beta
        #new 2是1-2，new3是1-3
        # self.log_file = open('{}.txt'.format(time.time()), 'w')

        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        

        
        self.margin = nn.Parameter(torch.Tensor([self.args.margin]))
        self.margin.requires_grad = False
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

    def forward(self, triples, negs=None, mode='single'):
        s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb = self.tri2emb(triples, mode, negs)
        score = (
                self._calc1(s_h_emb, s_t_emb, s_r_emb)
                + self._calc1(h_img_emb, t_img_emb, s_r_emb)
                + self._calc1(h_img_emb, s_t_emb, s_r_emb)
                + self._calc1(s_h_emb, t_img_emb, s_r_emb)
                + self._calc1(s_h_emb + h_img_emb, s_t_emb + t_img_emb, s_r_emb)
                + self._calc1(s_h_emb, s_t_emb, r_img_emb)
                + self._calc1(h_img_emb, t_img_emb, r_img_emb)
                + self._calc1(h_img_emb, s_t_emb, r_img_emb)
                + self._calc1(s_h_emb, t_img_emb, r_img_emb)
                + self._calc1(s_h_emb + h_img_emb, s_t_emb + t_img_emb, r_img_emb)
        ) / 10

        if self.margin_flag:
            return self.margin - score
        else:
            return score

    
    def get_score(self, batch, mode):
        triples = batch['positive_sample']
        s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb = self.tri2emb(triples, mode)
        score = (
                self._calc1(s_h_emb, s_t_emb, s_r_emb)
                + self._calc1(h_img_emb, t_img_emb, s_r_emb)
                + self._calc1(h_img_emb, s_t_emb, s_r_emb)
                + self._calc1(s_h_emb, t_img_emb, s_r_emb)
                + self._calc1(s_h_emb + h_img_emb, s_t_emb + t_img_emb, s_r_emb)
                + self._calc1(s_h_emb, s_t_emb, r_img_emb)
                + self._calc1(h_img_emb, t_img_emb, r_img_emb)
                + self._calc1(h_img_emb, s_t_emb, r_img_emb)
                + self._calc1(s_h_emb, t_img_emb, r_img_emb)
                + self._calc1(s_h_emb + h_img_emb, s_t_emb + t_img_emb, r_img_emb)
        ) / 10

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

        r_img_emb = self.img_proj(self.rel_img_embeddings(triples[:, 1])).unsqueeze(1)
        return s_h_emb, s_r_emb, s_t_emb, h_img_emb, t_img_emb, r_img_emb

   