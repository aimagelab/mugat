import torch
import torch.nn as nn

class PerceiverLayer(nn.Module):
    def __init__(self, d_model=1024, num_heads=1, hidden_size=2048, seq_length=4096, extra_tokens=2):
        super().__init__()

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        
        self.norm1q = nn.LayerNorm((extra_tokens, d_model))
        self.norm1kv = nn.LayerNorm((seq_length, d_model))
        self.norm2 = nn.LayerNorm((extra_tokens, d_model))

        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model)
        )


    def forward(self, x, kv):
        q = self.q_projection(self.norm1q(x))
        
        kv = self.norm1kv(kv)
        k = self.k_projection(kv)
        v = self.v_projection(kv)

        y = x + self.attention(q, k, v)[0] + self.ff(self.norm2(x))

        return y


class PerceiverAdapter(nn.Module):
    def __init__(self, d_model=1024, num_heads=1, num_layers=2, extra_tokens=2):
        super().__init__()

        self.queries = nn.Parameter(torch.zeros(1, extra_tokens, d_model))
        self.transformer = [PerceiverLayer(d_model, num_heads) for i in range(num_layers)]

    def forward(self, x):
        y = self.queries
        for layer in self.transformer:
            y = layer(y, x)

        return y

        

