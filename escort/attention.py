import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    
    This is the core attention computation used in various
    attention modules throughout ESCORT.
    """
    
    def __init__(self, temperature: float = 1.0, dropout: float = 0.0):
        """
        Initialize scaled dot-product attention.
        
        Args:
            temperature: Temperature for attention softmax
            dropout: Dropout probability
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch, n_queries, d_k)
            key: Key tensor of shape (batch, n_keys, d_k)
            value: Value tensor of shape (batch, n_keys, d_v)
            mask: Optional mask of shape (batch, n_queries, n_keys)
        
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadParticleAttention(nn.Module):
    """
    Multi-head attention specifically designed for particle interactions.
    
    This module enables particles to share information while preserving
    multi-modal structure in the belief distribution.
    """
    
    def __init__(self,
                d_model: int,
                num_heads: int = 8,
                dropout: float = 0.0,
                temperature: float = 1.0,
                use_projection_bias: bool = True):
        """
        Initialize multi-head particle attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Temperature for attention
            use_projection_bias: Whether to use bias in projections
        """
        super(MultiHeadParticleAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=use_projection_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_projection_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_projection_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=use_projection_bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(temperature, dropout)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Apply multi-head attention to particles.
        
        Args:
            x: Input tensor of shape (batch, n_particles, d_model)
            mask: Optional mask of shape (batch, n_particles)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (and optionally attention weights)
        """
        batch_size, n_particles, _ = x.size()
        residual = x
        
        # Linear projections in batch from d_model => h x d_k
        query = self.w_q(x).view(batch_size, n_particles, self.num_heads, self.d_k)
        key = self.w_k(x).view(batch_size, n_particles, self.num_heads, self.d_k)
        value = self.w_v(x).view(batch_size, n_particles, self.num_heads, self.d_k)
        
        # Transpose for attention: b x h x n x d_k
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Prepare mask if needed
        if mask is not None:
            # Expand mask for heads: (batch, 1, 1, n_particles)
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Create attention mask: (batch, 1, n_particles, n_particles)
            mask = mask * mask.transpose(-1, -2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(query, key, value, mask)
        
        # Concatenate heads: b x n x (h*d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, n_particles, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(residual + self.dropout(output))
        
        if return_attention:
            # Average attention weights across heads
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        
        return output


class ModeAwareAttention(nn.Module):
    """
    Attention mechanism that is aware of modes in the particle distribution.
    
    This module helps preserve multi-modal structure by computing attention
    weights that respect mode boundaries.
    """
    
    def __init__(self,
                d_model: int,
                num_heads: int = 4,
                mode_embedding_dim: int = 32,
                dropout: float = 0.0):
        """
        Initialize mode-aware attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            mode_embedding_dim: Dimension of mode embeddings
            dropout: Dropout probability
        """
        super(ModeAwareAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.mode_embedding_dim = mode_embedding_dim
        
        # Mode detection network
        self.mode_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, mode_embedding_dim),
            nn.Tanh()
        )
        
        # Mode-conditioned attention
        self.attention = MultiHeadParticleAttention(
            d_model, num_heads, dropout
        )
        
        # Mode similarity threshold
        self.similarity_threshold = nn.Parameter(torch.tensor(0.5))
    

    def compute_mode_similarity(self,
                            mode_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise mode similarity scores.
        
        Args:
            mode_embeddings: Mode embeddings of shape (batch, n_particles, mode_dim)
            
        Returns:
            Similarity matrix of shape (batch, n_particles, n_particles)
        """
        # Normalize embeddings
        mode_embeddings = F.normalize(mode_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(mode_embeddings, mode_embeddings.transpose(-2, -1))
        
        # Apply threshold to create mode mask
        mode_mask = (similarity > self.similarity_threshold).float()
        
        return mode_mask
    

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply mode-aware attention.
        
        Args:
            x: Input tensor of shape (batch, n_particles, d_model)
            mask: Optional mask of shape (batch, n_particles)
            
        Returns:
            Output tensor of shape (batch, n_particles, d_model)
        """
        # Detect modes
        mode_embeddings = self.mode_detector(x)
        
        # Compute mode-based attention mask
        mode_mask = self.compute_mode_similarity(mode_embeddings)
        
        # Combine with input mask if provided
        if mask is not None:
            # Expand mask for pairwise comparison
            mask_expanded = mask.unsqueeze(1) * mask.unsqueeze(2)
            mode_mask = mode_mask * mask_expanded
        
        # Apply attention with mode mask
        output = self.attention(x, mode_mask)
        
        return output


class TemporalParticleAttention(nn.Module):
    """
    Attention mechanism for temporal sequences of particle beliefs.
    
    This module enables particles to attend to their past states,
    supporting temporal consistency in belief evolution.
    """
    
    def __init__(self,
                d_model: int,
                num_heads: int = 4,
                max_sequence_length: int = 100,
                dropout: float = 0.0):
        """
        Initialize temporal particle attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super(TemporalParticleAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_sequence_length
        )
        
        # Gating mechanism for temporal information
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    

    def forward(self,
                current_particles: torch.Tensor,
                past_particles: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention between current and past particles.
        
        Args:
            current_particles: Current particles (batch, n_particles, d_model)
            past_particles: Past particles (batch, n_timesteps, n_particles, d_model)
            temporal_mask: Optional mask for past timesteps (batch, n_timesteps)
            
        Returns:
            Updated particles with temporal information
        """
        batch_size, n_particles, d_model = current_particles.shape
        n_timesteps = past_particles.shape[1]
        
        # Reshape for processing each particle's history
        # (batch * n_particles, 1, d_model)
        current_flat = current_particles.view(-1, 1, d_model)
        
        # (batch * n_particles, n_timesteps, d_model)
        past_flat = past_particles.transpose(1, 2).contiguous().view(
            batch_size * n_particles, n_timesteps, d_model
        )
        
        # Add positional encoding to past particles
        past_flat = self.positional_encoding(past_flat)
        
        # Apply temporal attention
        attended, _ = self.temporal_attention(
            current_flat, past_flat, past_flat,
            key_padding_mask=temporal_mask.repeat_interleave(n_particles, dim=0) if temporal_mask is not None else None
        )
        
        # Reshape back
        attended = attended.view(batch_size, n_particles, d_model)
        
        # Gating mechanism
        combined = torch.cat([current_particles, attended], dim=-1)
        gate = self.temporal_gate(combined)
        
        # Combine current and temporal information
        output = gate * attended + (1 - gate) * current_particles
        
        # Output projection and normalization
        output = self.output_projection(output)
        output = self.layer_norm(output + current_particles)
        
        return output


class ProjectionAwareAttention(nn.Module):
    """
    Attention mechanism that leverages GSWD projection directions.
    
    This module aligns attention computation with the correlation-aware
    projections learned by GSWD, ensuring consistency across ESCORT components.
    """
    
    def __init__(self,
                d_model: int,
                n_projections: int = 10,
                projection_dim: int = 64,
                num_heads: int = 4,
                dropout: float = 0.0):
        """
        Initialize projection-aware attention.
        
        Args:
            d_model: Model dimension
            n_projections: Number of projection directions
            projection_dim: Dimension of projected space
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(ProjectionAwareAttention, self).__init__()
        
        self.d_model = d_model
        self.n_projections = n_projections
        self.projection_dim = projection_dim
        
        # Learnable projection matrices (similar to GSWD)
        self.projection_matrices = nn.Parameter(
            torch.randn(n_projections, d_model, projection_dim)
        )
        
        # Projection weights
        self.projection_weights = nn.Parameter(
            torch.ones(n_projections) / n_projections
        )
        
        # Attention in projected spaces
        self.projected_attention = nn.ModuleList([
            ScaledDotProductAttention(dropout=dropout)
            for _ in range(n_projections)
        ])
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(d_model + n_projections * projection_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize projections
        self._init_projections()
    

    def _init_projections(self):
        """Initialize projection matrices orthogonally."""
        for i in range(self.n_projections):
            nn.init.orthogonal_(self.projection_matrices[i])
    

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply projection-aware attention.
        
        Args:
            x: Input tensor of shape (batch, n_particles, d_model)
            mask: Optional mask of shape (batch, n_particles)
            
        Returns:
            Output tensor of shape (batch, n_particles, d_model)
        """
        batch_size, n_particles, _ = x.shape
        residual = x
        
        projected_outputs = []
        
        # Apply attention in each projected space
        for i in range(self.n_projections):
            # Project particles
            projection = self.projection_matrices[i]  # (d_model, projection_dim)
            x_projected = torch.matmul(x, projection)  # (batch, n_particles, projection_dim)
            
            # Apply attention in projected space
            attn_output, _ = self.projected_attention[i](
                x_projected, x_projected, x_projected, mask
            )
            
            # Weight by projection importance
            weighted_output = attn_output * self.projection_weights[i]
            projected_outputs.append(weighted_output)
        
        # Concatenate all projected outputs
        all_projections = torch.cat(projected_outputs, dim=-1)
        
        # Combine with original features
        combined = torch.cat([x, all_projections], dim=-1)
        
        # Fusion and output
        output = self.fusion(combined)
        output = self.layer_norm(output + residual)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating particle features.
    
    This provides a learnable alternative to mean/max pooling
    for creating belief representations from particles.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        """
        Initialize attention pooling.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super(AttentionPooling, self).__init__()
        
        # Learnable query vector for pooling
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
    

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool particle features using attention.
        
        Args:
            x: Input tensor of shape (batch, n_particles, d_model)
            mask: Optional mask of shape (batch, n_particles)
            
        Returns:
            Pooled features of shape (batch, d_model)
        """
        batch_size = x.shape[0]
        
        # Expand query for batch
        query = self.pooling_query.expand(batch_size, -1, -1)
        
        # Apply attention pooling
        pooled, attention_weights = self.attention(query, x, x, mask)
        
        # Remove sequence dimension and project
        pooled = pooled.squeeze(1)
        output = self.output_projection(pooled)
        
        return output
