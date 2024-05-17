import torch
import torch.nn as nn

class PerceiverLayer(nn.Module):
    def __init__(self, d_model=1024, num_heads=1, hidden_size=2048, seq_length=4096):
        super().__init__()
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm((seq_length, d_model))
        self.norm2 = nn.LayerNorm((seq_length, d_model))

        self.attention = nn.MultiheadAttention(num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model)
        )


    def forward(self, x, kv):
        q = self.self.q_projection(self.norm1(x))
        
        kv = self.norm1(kv)
        k = self.k_projection(kv)
        v = self.v_projection(kv)

        x += self.attention(q, k, v)

        x+=self.ff(self.norm2(x))

        return x


class PerceiverAdapter(nn.Module):
    def __init__(self, d_model=1024, num_heads=1, num_layers=2, extra_tokens=2):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, extra_tokens, d_model))
        enc_layer = PerceiverLayer(d_model, num_heads)
        self.transformer = nn.TransformerDecoder(enc_layer, num_layers)

    def forward(self, x):

        x = self.transformer(self.queries, x)

        return x

        

