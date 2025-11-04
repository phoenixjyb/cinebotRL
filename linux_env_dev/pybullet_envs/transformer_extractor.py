import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    A small Transformer-based feature extractor for vector observations.

    This module projects the input observation vector into a sequence of tokens,
    runs a TransformerEncoder over them, and returns the pooled embedding as
    the feature vector for SB3 policies.

    Configurable hyperparameters:
    - seq_len: number of tokens to split the observation into
    - embed_dim: embedding dimension (output feature size)
    - n_heads: attention heads
    - n_layers: Transformer encoder layers
    - dropout: dropout inside TransformerEncoderLayer
    """

    def __init__(self, observation_space: spaces.Box, seq_len: int = 8, embed_dim: int = 256,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        assert isinstance(observation_space, spaces.Box), "TransformerFeaturesExtractor expects Box observation"
        obs_shape = observation_space.shape
        assert len(obs_shape) == 1, "Only 1D Box observation supported by this extractor"
        obs_dim = int(obs_shape[0])

        # features_dim is the dimension returned to the policy (embedding dim)
        super().__init__(observation_space, features_dim=embed_dim)

        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)

        # Project flat observation into seq_len * embed_dim and reshape to (batch, seq_len, embed_dim)
        self.proj = nn.Linear(obs_dim, self.seq_len * self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=max(1, n_heads), dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, n_layers))

        # simple layernorm & final projection (identity size)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, 28(obs_dim) )
        # 这一段是在做tokenization，把当前观测投影成一个token序列
        x = self.proj(observations)  # (batch, seq_len * embed_dim)  (bs, 2047)
        bsz = x.shape[0]
        x = x.view(bsz, self.seq_len, self.embed_dim)  # (batch, seq_len, embed_dim) (bs, 8, 256)

        # 编码
        x = x.permute(1, 0, 2) # (8, bs, 256)
        x = self.transformer(x)  # (seq_len, batch, embed_dim) (8, bs, 256)

        # back to (batch, seq_len, embed_dim)
        x = x.permute(1, 2, 0) # (8, bs, 256) -> (bs, 256, 8)
        # pool over sequence dimension
        x = torch.mean(x, dim=2) # (batch, embed_dim) (bs, 256)
        x = self.ln(x) # (batch, embed_dim) (bs, 256)

        return x

    def _get_policy_from_name(self, policy):
        return "QiTransformer"
