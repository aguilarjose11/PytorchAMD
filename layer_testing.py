from Transformer.Transformer import TranslatorTransformer
import torch
import torch.nn as nn

import torch
from AttentionModel.AttentionModel import AttentionModel

d_m = 16
d_c = 16
d_k = 16
h = 4
N = 2
d_ff = 24
n_nodes = 20
embeder = 3
d_v = 16
c = 10.
head_split = True
dropout = 0.1
use_graph_emb = True
batches = None # None defined, be flexible. Really, this is useless!

am = AttentionModel(16, 16, 16, 4, 2, 24, 20, 3, 16, 10, True, 0.1)
graph = torch.rand(7, 10, 3)
ctxt = torch.rand(7, 1, 16)
mask_emb_graph = torch.triu(torch.ones(7, 10), diagonal=1).unsqueeze(1).bool()
mask_dec_graph = torch.zeros(7, 1, 10).bool()
mask_dec_graph[:,:,0] = True
a, b = am(graph, ctxt, mask_emb_graph, mask_dec_graph)

print("Inference completed")