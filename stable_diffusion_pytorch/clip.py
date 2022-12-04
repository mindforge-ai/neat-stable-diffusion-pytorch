import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_len: int, embedding_len: int, window_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_len, embedding_len)
        self.position_value = nn.Parameter(torch.zeros((window_len, embedding_len)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_value
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        vocab_len: int = 49408,
        embedding_len: int = 768,
        window_len: int = 77,
        num_layers: int = 12,
    ):
        super().__init__()
        self.embedding = CLIPEmbedding(vocab_len, embedding_len, window_len)
        self.layers = nn.ModuleList(
            [CLIPLayer(num_layers, embedding_len) for i in range(num_layers)]
        )
        self.layernorm = nn.LayerNorm(embedding_len)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
