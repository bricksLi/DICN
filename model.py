import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCNModelGumbel(nn.Module):  
    def __init__(self, size, embedding_dim, categorical_dim, dropout, device):  
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim  # 128
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size  

        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)  
        self.node_embeddings = nn.Embedding(size, embedding_dim)  
        self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, size),
        ).to(device)  # （128,2708）
        
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, temp):
        
        w = self.node_embeddings(w).to(self.device)  
        c = self.node_embeddings(c).to(self.device)
        
        q = self.community_embeddings(w * c)  
       
        if self.training:  
            
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True)  
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)  # (5278,1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.) 

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)
        # prior.shape [batch_num_nodes,categorical_dim]

        
        new_z = torch.mm(z, self.community_embeddings.weight)  
        recon = self.decoder(new_z)  
        
        return recon, F.softmax(q, dim=-1), prior,self.node_embeddings,self.community_embeddings 
        # return recon, q, prior,self.node_embeddings,self.community_embeddings 
