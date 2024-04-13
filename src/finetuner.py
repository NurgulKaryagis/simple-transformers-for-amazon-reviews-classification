import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(rank, input_dim))
        self.B = nn.Parameter(torch.randn(output_dim, rank))

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x_reshaped = x.view(-1, feature_dim) 
        ΔWx = torch.matmul(x_reshaped, self.A.t())  
        ΔWx = torch.matmul(ΔWx, self.B.t()) 
        return ΔWx.view(batch_size, seq_len, -1)  


class RoSALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=2, sparsity=0.9):
        super(RoSALayer, self).__init__()
        self.rank = rank
        self.low_rank_u = nn.Parameter(torch.randn(input_dim, rank))
        self.low_rank_v = nn.Parameter(torch.randn(rank, output_dim))
        self.sparse_weight = nn.Parameter(torch.randn(output_dim)) 
        self.sparsity = sparsity
        self._create_sparse_mask()

    def _create_sparse_mask(self):
        mask = torch.rand(self.sparse_weight.size()) > self.sparsity
        self.sparse_weight.data *= mask

    def forward(self, x):
        x_low_rank = torch.matmul(x, self.low_rank_u)
        x_low_rank = torch.matmul(x_low_rank, self.low_rank_v)
        
        sparse_component = self.sparse_weight.unsqueeze(0).unsqueeze(1) 

        sparse_component = sparse_component.expand(x.size(0), x.size(1), self.sparse_weight.size(0))
        
        result = x + x_low_rank + sparse_component
        return result




