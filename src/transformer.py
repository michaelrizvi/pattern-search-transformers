from typing import Any

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3 * self.d_k).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn_weights) @ v

        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


class DecoderOnlyClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, n_classes, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, T = x.size()
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x[:, -1, :]  # or x.mean(dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = DecoderOnlyTransformer(
        vocab_size=10,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=512
    )
    x = torch.randint(0, 10, (2, 5))  # Batch size of 2, sequence length of 50
    print(x)
    print(x.shape)
    output = model(x)
    print(output.shape)  # Should be (2, 50, 10000)
    print(output)