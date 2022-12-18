import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CLIPSelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_len: int, embedding_len: int, window_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_len, embedding_len)
        self.position_embedding = nn.Embedding(window_len, embedding_len)
        self.register_buffer("position_indices", torch.arange(window_len).unsqueeze(0))

    def forward(self, tokens):
        embeddings = self.token_embedding(tokens)
        positions = self.position_embedding(self.position_indices)
        hidden_states = embeddings + positions
        return hidden_states


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.self_attention = CLIPSelfAttention(n_head, n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x
        x = self.layer_norm_1(x)
        x = self.self_attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layer_norm_2(x)
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
        self.stack = nn.ModuleList(
            [CLIPLayer(num_layers, embedding_len) for i in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_len)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)
        for layer in self.stack:
            state = layer(state)
        output = self.final_layer_norm(state)
        return output
