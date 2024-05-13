from collections import defaultdict as ddict
rel2 = []
ent = {}
rel1 = {}
triple_len = []
from urllib.parse import urlparse
with open("/home/xuyajing/data/mmkg/FBDB15K/norm/ent_ids_1","r") as f:
    for line in f.readlines():
        id, name = line.strip('\n').split('\t')
        ent[name] = id
        # head_url = urlparse(name.strip('<>').split('//')[-1])
        # last_head = head_url.path.split('/')[-1]
        # ent[id] = last_head
# file2 = open("/home/xuyajing/data/mmkg/FBDB15K/DB15K/entities.txt","w+")
# for key, value in ent.items():
#     file2.write(key+'\t'+value+'\n')

# for key, value in triple.items():
#     triple_len.append(len(value))

from urllib.parse import urlparse
with open("/home/xuyajing/data/mmkg/FBDB15K/FB15K_EntityTriples.txt","r") as f:
    for line in f.readlines():
        head,rel,tail =line.strip(' .\n').split(' ')
        # head_url = urlparse(head.strip('<>').split('//')[-1])
        # last_head = head_url.path.split('/')[-1]
        # rel_url = urlparse(rel.strip('<>').split('//')[-1])
        # last_rel = rel_url.path.split('/')[-1]
        # tail_url = urlparse(tail.strip('<>').split('//')[-1])
        # last_tail = tail_url.path.split('/')[-1]
        rel2.append(rel)
i = 0
for r in set(rel2):
    rel1[r] = i
    i = i+1
file2 = open("/home/xuyajing/NeuralKG/NeuralKG-main/dataset/FB15K2/re_triple.txt","w+")
file3 = open("/home/xuyajing/NeuralKG/NeuralKG-main/dataset/FB15K2/relation.txt","w+")
for key, value in rel1.items():
    file3.write(str(value)+'\t'+key+'\n')
with open("/home/xuyajing/data/mmkg/FBDB15K/FB15K_EntityTriples.txt","r") as f:
    for line in f.readlines():
        head,rel,tail =line.strip(' .\n').split(' ')
        # head_url = urlparse(head.strip('<>').split('//')[-1])
        # last_head = head_url.path.split('/')[-1]
        # rel_url = urlparse(rel.strip('<>').split('//')[-1])
        # last_rel = rel_url.path.split('/')[-1]
        # tail_url = urlparse(tail.strip('<>').split('//')[-1])
        # last_tail = tail_url.path.split('/')[-1]
        id_head, id_rel, id_tail = ent[head], rel1[rel], ent[tail]
        file2.write(id_head+'\t'+str(id_rel)+'\t'+id_tail+'\n')


