import torch
import torch.nn as nn

class PerceiverLayer(nn.Module):
    # default sequence length is 588, we multiply by three (we encode three pages by default) to make it 1764
    def __init__(self, d_model=1024, num_heads=1, hidden_size=2048, seq_length=1764, extra_tokens=2):
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
    def __init__(self, d_model=1024, num_heads=1, num_layers=2, extra_tokens=2, seq_length=1764, pages=3):
        super().__init__()
        self.page_embeddings = nn.Parameter(torch.randn(1, pages, d_model))
        self.intra_page_embeddings = nn.Parameter(torch.randn(1, seq_length//3, d_model))
        
        self.transformer = nn.ModuleList(
            [PerceiverLayer(d_model, num_heads) for i in range(num_layers)]
        )
        self.queries = nn.Parameter(torch.zeros(1, extra_tokens, d_model))

    def forward(self, x):
        y = self.queries
        



        # this is because we want all three pages to have the same token embeddings intra-page
        # so we repeat the token embeddings for each page (second dimension of the page embedding
        # tensor is the page count, as we can see in the init function)
        x+=self.intra_page_embeddings.repeat(1, self.page_embeddings.shape[1], 1)

        for layer in self.transformer:
            y = layer(y, x)

        return y

        

