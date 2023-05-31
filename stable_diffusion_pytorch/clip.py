import torch
from torch import nn

from .attention import SelfAttention
from .embeddings import TokenAndPositionEmbedding


class TransformerLayer(nn.Module):
    """
    Encoder layer.
    """

    def __init__(
        self, num_heads: int, embedding_len: int, scale_attention_scores: bool = True
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_len)
        self.self_attention = SelfAttention(
            num_heads, embedding_len, scale=scale_attention_scores
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_len)
        self.linear_1 = nn.Linear(
            embedding_len, 4 * embedding_len
        )  # hidden_size, intermediate_size
        self.linear_2 = nn.Linear(4 * embedding_len, embedding_len)

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


class CLIPTextEncoder(nn.Module):
    """
    No decoder.
    """

    def __init__(
        self,
        vocab_len: int = 49408,
        embedding_len: int = 768,
        window_len: int = 77,
        num_layers: int = 12,
    ):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_len, embedding_len, window_len)
        self.stack = nn.ModuleList(
            [
                TransformerLayer(
                    num_layers, embedding_len, scale_attention_scores=True
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        tokens = tokens.to(torch.long)

        hidden_states = self.embedding(tokens)

        for layer in self.stack:
            hidden_states = layer(hidden_states)

        output = self.final_layer_norm(hidden_states)

        return output
