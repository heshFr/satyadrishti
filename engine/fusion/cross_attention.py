"""
Cross-Modal Attention Fusion — The Core Innovation
=====================================================
Custom Cross-Attention Transformer that fuses Audio, Vision,
and Text embeddings to learn cross-modal correlations for the
final multimodal threat decision.

The key insight: a synthetic voice (Audio) combined with a
flatline rPPG signal (Video) and coercive language (Text)
is exponentially more threatening than any single modality alone.

Architecture: Asymmetric cross-attention with modality-specific
projection layers to handle varying embedding dimensionalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModalityProjector(nn.Module):
    """
    Projects each modality's embedding into a shared latent space.
    Handles dimensionality mismatch between Audio (768), Video (768),
    and Text (768) embeddings.
    """

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer where one modality attends to another.

    Uses multi-head attention where the Query comes from one modality
    and Keys/Values come from another, allowing each modality to
    "look at" the other modalities for contextual evidence.
    """

    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert latent_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, latent_dim) — the modality doing the looking
            context: (batch, N, latent_dim) — the modalities being looked at

        Returns:
            (batch, latent_dim) — updated query representation
        """
        # Add sequence dim to query if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (batch, 1, latent_dim)

        batch_size = query.size(0)

        # Pre-norm
        q = self.norm1(query)

        # Multi-head projections
        Q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        x = query + attn_output

        # Feed-forward with pre-norm + residual
        x = x + self.ffn(self.norm2(x))

        return x.squeeze(1)  # (batch, latent_dim)


class MultimodalFusionNetwork(nn.Module):
    """
    The Heart of Satya Drishti.

    Takes embeddings from the three frozen encoders (Audio, Video, Text)
    and fuses them via iterative cross-attention to produce a unified
    threat prediction.

    Architecture:
        1. Project each modality into shared latent space
        2. Stack modality tokens into a context sequence
        3. Apply N layers of cross-attention (each modality attends to all)
        4. Aggregate via self-attention pooling
        5. Classify: [Real, Deepfake, Coercion, Deepfake+Coercion]

    Args:
        audio_embed_dim: Dimension of audio encoder output (AST [CLS] token)
        video_embed_dim: Dimension of video encoder output (ViT)
        text_embed_dim: Dimension of text encoder output (DeBERTaV3 [CLS])
        latent_dim: Shared cross-attention latent dimension
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        num_classes: Number of threat categories
        dropout: Dropout rate
    """

    def __init__(
        self,
        audio_embed_dim: int = 768,
        video_embed_dim: int = 768,
        text_embed_dim: int = 768,
        latent_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Modality-specific projectors
        self.audio_proj = ModalityProjector(audio_embed_dim, latent_dim, dropout)
        self.video_proj = ModalityProjector(video_embed_dim, latent_dim, dropout)
        self.text_proj = ModalityProjector(text_embed_dim, latent_dim, dropout)

        # Learnable modality type tokens
        self.audio_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.video_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Self-attention pooling over fused modality tokens
        self.pool_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(latent_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> dict:
        """
        Fuse multimodal embeddings and predict threat category.

        Args:
            audio_emb: (batch, audio_embed_dim) from frozen AST
            video_emb: (batch, video_embed_dim) from frozen ViT
            text_emb: (batch, text_embed_dim) from frozen DeBERTaV3

        Returns:
            dict with:
                - logits: (batch, num_classes)
                - fused_embedding: (batch, latent_dim) for explainability
                - modality_weights: attention weights per modality
        """
        batch_size = audio_emb.size(0)

        # Project to shared latent space
        a = self.audio_proj(audio_emb)  # (batch, latent_dim)
        v = self.video_proj(video_emb)
        t = self.text_proj(text_emb)

        # Add modality type tokens
        a = a + self.audio_token.squeeze(1)
        v = v + self.video_token.squeeze(1)
        t = t + self.text_token.squeeze(1)

        # Stack into context sequence: (batch, 3, latent_dim)
        context = torch.stack([a, v, t], dim=1)

        # Iterative cross-attention: each modality attends to the full context
        for layer in self.cross_attn_layers:
            a_new = layer(a, context)
            v_new = layer(v, context)
            t_new = layer(t, context)

            # Update context with refined representations
            a, v, t = a_new, v_new, t_new
            context = torch.stack([a, v, t], dim=1)

        # Self-attention pooling to aggregate modality tokens
        pooled, attn_weights = self.pool_attn(
            self.pool_norm(context),
            self.pool_norm(context),
            self.pool_norm(context),
        )

        # Mean pool across modality tokens
        fused = pooled.mean(dim=1)  # (batch, latent_dim)

        # Classify
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused_embedding": fused,
            "modality_weights": attn_weights,
        }
