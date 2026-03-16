import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def precompute_freqs_cis(d_model, max_len=1024):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    freqs_cis = freqs_cis[:, :xq.shape[1], :, :]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LLaMALayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nhead = config.nhead
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.nhead
        
        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        
        hidden_dim = int(config.d_model * 4) 
        self.feed_forward = SwiGLU(config.d_model, hidden_dim)
        
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, freqs_cis, mask):
        batch, seq_len, _ = x.shape
        
        h = self.attention_norm(x)
        xq = self.wq(h).view(batch, seq_len, self.nhead, self.head_dim)
        xk = self.wk(h).view(batch, seq_len, self.nhead, self.head_dim)
        xv = self.wv(h).view(batch, seq_len, self.nhead, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        h = x + self.wo(attn_out)
        
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class CalculatorTransformer(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([LLaMALayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model)
        
        self.fc_out = nn.Linear(config.d_model, vocab_size, bias=False)
        
        self.embedding.weight = self.fc_out.weight
        
        freqs_cis = precompute_freqs_cis(self.d_model // config.nhead, max_len=512)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.embedding(x)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask)
            
        h = self.norm(h)
        logits = self.fc_out(h)
        
        return logits
