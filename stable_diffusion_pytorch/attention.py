import math

import torch
from torch import nn, einsum
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        in_proj_bias=True,
        out_proj_bias=True,
        scale=True,
    ):
        super().__init__()
        self.to_query = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_key = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_value = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=out_proj_bias)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_dim_per_head = embedding_dim // num_heads
        self.scale = scale

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        batch_size, sequence_length, _ = input_size = x.size()

        query, key, value = self.to_query(x), self.to_key(x), self.to_value(x)

        split_into_heads_shape = (
            batch_size,
            sequence_length,
            self.num_heads,  # add a dimension for the different attention heads
            self.embedding_dim_per_head,
        )

        query = query.view(split_into_heads_shape).transpose(1, 2)
        key = key.view(split_into_heads_shape).transpose(1, 2)
        value = value.view(split_into_heads_shape).transpose(1, 2)

        scores = query @ key.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)

        if self.scale:
            scores = scores / math.sqrt(self.embedding_dim)

        softmaxed_scores = F.softmax(scores, dim=-1)

        output = softmaxed_scores @ value
        output = output.transpose(1, 2)
        output = output.reshape(
            input_size
        )  # might be better to do view or something here
        output = self.out_proj(output)
        return output


class EncoderSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        in_proj_bias=True,
        out_proj_bias=True,
        scale=True,
    ):
        super().__init__()
        self.to_query = torch.nn.Conv2d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.to_key = torch.nn.Conv2d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.to_value = torch.nn.Conv2d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.out_proj = torch.nn.Conv2d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_dim_per_head = embedding_dim // num_heads
        self.scale = scale

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:

        batch_size, sequence_length, h, w = input_size = x.size()

        query, key, value = self.to_query(x), self.to_key(x), self.to_value(x)

        query = query.reshape(batch_size, sequence_length, h * w)
        key = key.reshape(batch_size, sequence_length, h * w)
        value = value.reshape(batch_size, sequence_length, h * w)

        scores = query.transpose(-1, -2) @ key

        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)

        if self.scale:
            scores = scores / math.sqrt(self.embedding_dim)

        softmaxed_scores = F.softmax(scores, dim=-1)

        output = value @ softmaxed_scores.transpose(-1, -2)
        output = output.reshape(
            input_size
        )  # might be better to do view or something here
        output = self.out_proj(output)
        return output


class EinsumSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        in_proj_bias=True,
        out_proj_bias=True,
        scale=True,
    ):
        super().__init__()
        self.to_query = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_key = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_value = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=out_proj_bias)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_dim_per_head = embedding_dim // num_heads
        self.scale = scale

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        batch_size, sequence_length, _ = input_size = x.size()

        query, key, value = self.to_query(x), self.to_key(x), self.to_value(x)

        split_into_heads_shape = (
            batch_size,
            sequence_length,
            self.num_heads,  # add a dimension for the different attention heads
            self.embedding_dim_per_head,
        )

        query = query.view(split_into_heads_shape).transpose(1, 2)
        key = key.view(split_into_heads_shape).transpose(1, 2)
        value = value.view(split_into_heads_shape).transpose(1, 2)

        scores = einsum("b i d, b j d -> b i j", query.squeeze(0), key.squeeze(0))

        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)

        if self.scale:
            scores = scores * (
                self.embedding_dim_per_head**-0.5
            )  # NOTE: different scale type

        softmaxed_scores = F.softmax(scores, dim=-1)

        output = softmaxed_scores @ value
        output = output.transpose(1, 2)
        output = output.reshape(
            input_size
        )  # might be better to do view or something here
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(
        self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.to_query = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.to_key = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.to_value = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q, k, v = self.to_query(x), self.to_key(y), self.to_value(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output
