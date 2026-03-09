import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape: (max_len, d_model), unsqueeze to (1, max_len, d_model) for batch matching
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class CalculatorTransformer(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc_out = nn.Linear(config.d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x is (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Causal mask for autoregressive training
        mask = self.generate_square_subsequent_mask(seq_len, x.device)
        
        # (batch_size, seq_len, d_model)
        output = self.transformer(x, mask=mask, is_causal=True)
        # (batch_size, seq_len, vocab_size)
        logits = self.fc_out(output)
        
        return logits
