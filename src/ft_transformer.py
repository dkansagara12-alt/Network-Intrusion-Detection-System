"""
FT-Transformer (Feature Tokenizer + Transformer) for tabular data.

Implementation follows the architecture described in:
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
"Revisiting Deep Learning Models for Tabular Data." NeurIPS 2021.

Every feature becomes one d-dimensional token: each numerical feature via its
own learned linear projection, each categorical feature via a learned
embedding (see FeatureTokenizer). A [CLS] token is prepended, the sequence is
passed through a standard Transformer encoder, and the [CLS] token's
representation is used for classification.
"""

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Turns each input feature into its own d-dimensional token.

    Numerical feature j:   token = x_j * w_j + b_j
    Categorical feature:   arrives one-hot encoded (so the baselines can use
                           the same matrix); its one-hot block is mapped to a
                           single token by a linear layer. For a one-hot input
                           this is exactly an embedding lookup (one learned
                           vector per category), as in the original paper.

    cat_groups lists, per categorical feature, the column indices of its
    one-hot block. Without it every column is treated as numerical, which
    turns e.g. UNSW-NB15's 163 one-hot columns into 163 separate tokens.
    """

    def __init__(self, num_features: int, d_token: int, cat_groups=()):
        super().__init__()
        cat_cols = {c for group in cat_groups for c in group}
        num_idx = [i for i in range(num_features) if i not in cat_cols]
        self.register_buffer("num_idx", torch.tensor(num_idx, dtype=torch.long))
        self.num_cat = len(cat_groups)
        self.num_tokens = len(num_idx) + len(cat_groups)

        # one weight + bias per numerical feature -> equivalent to independent
        # linear layers, implemented efficiently with broadcasting
        self.weight = nn.Parameter(torch.empty(len(num_idx), d_token))
        self.bias = nn.Parameter(torch.empty(len(num_idx), d_token))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

        for k, group in enumerate(cat_groups):
            self.register_buffer(f"cat_idx_{k}", torch.tensor(group, dtype=torch.long))
        self.cat_embed = nn.ModuleList(nn.Linear(len(group), d_token) for group in cat_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features) -> tokens: (batch, num_tokens, d_token)
        x_num = x[:, self.num_idx]
        tokens = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if self.num_cat:
            cat_tokens = torch.stack(
                [embed(x[:, getattr(self, f"cat_idx_{k}")])
                 for k, embed in enumerate(self.cat_embed)],
                dim=1,
            )
            tokens = torch.cat([tokens, cat_tokens], dim=1)
        return tokens


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_token, num_heads, ffn_dim, dropout, attn_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # pre-norm transformer block
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_classes: int = 2,
        cat_groups=(),
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token, cat_groups)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_token, num_heads, ffn_dim, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = self.tokenizer(x)                              # (B, T, D)
        cls = self.cls_token.expand(batch_size, -1, -1)          # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                 # (B, T+1, D)

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = self.final_norm(tokens[:, 0])                  # (B, D)
        logits = self.head(cls_out)                              # (B, num_classes)
        return logits
