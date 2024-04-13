import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.finetuner import LoRALayer, RoSALayer


class Attention(nn.Module):
    def __init__(self, embed_size, heads, rank=None, sparsity=None, adaptation_layer=None):
        super(Attention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.adaptation_layer = adaptation_layer

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        if adaptation_layer == "LoRA" and rank is not None:
            self.lora = LoRALayer(input_dim=embed_size, output_dim=embed_size, rank=rank)
        elif adaptation_layer == "RoSA" and rank is not None and sparsity is not None:
            self.rosa = RoSALayer(input_dim=embed_size, output_dim=embed_size, rank=rank, sparsity=sparsity)
        else:
            self.lora = None
            self.rosa = None
            
    def forward(self, value, key, query, mask=None):
        N = query.shape[0]

        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        if self.adaptation_layer == "LoRA" and self.lora is not None:
            queries = self.lora(queries)
        elif self.adaptation_layer == "RoSA" and self.rosa is not None:
            queries = self.rosa(queries)

        queries = queries / (self.embed_size ** (0.5))
        keys = keys / (self.embed_size ** (0.5))

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention, values)
        out = self.fc_out(out)
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, num_layers, max_length, vocab_size, device, adaptation_layer=None, rank=None, sparsity=None):
        super(TransformerClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size).to(device)
        self.positional_embeddings = self.positional_encoding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout=0.1, forward_expansion=forward_expansion, adaptation_layer=adaptation_layer, rank=rank, sparsity=sparsity)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        N, seq_length = x.size()
        x = self.dropout(self.embedding(x) + self.positional_embeddings[:seq_length, :])
        
        for layer in self.layers:
            x = layer(x, x, x, None)
        
        x = x.mean(dim=1)
        return self.fc(x)

    def positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(self.device) 
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, rank=None, sparsity=None, adaptation_layer=None):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_size, heads, rank=rank, sparsity=sparsity, adaptation_layer=adaptation_layer)
        self.norm1 = nn.LayerNorm(embed_size, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(embed_size, elementwise_affine=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(query + attention)
        forward = self.feed_forward(x)
        out = self.norm2(x + forward)
        return out

