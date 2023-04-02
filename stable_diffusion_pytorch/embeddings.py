import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_len: int, embedding_len: int, window_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_len, embedding_len)
        self.position_embedding = nn.Embedding(window_len, embedding_len)
        self.register_buffer("position_indices", torch.arange(window_len).unsqueeze(0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded_tokens = self.token_embedding(tokens)
        embedded_positions = self.position_embedding(self.position_indices)
        hidden_states = embedded_tokens + embedded_positions
        return hidden_states