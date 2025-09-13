"""
Multi-omics transformer models with advanced attention mechanisms.

This module contains the main transformer architectures for multi-omics integration:
- EnhancedMultiOmicsTransformer: Basic enhanced transformer with interpretability
- AdvancedMultiOmicsTransformer: Advanced transformer with MoE, GQA, and flexible sample handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Union

from .attention import (
    InterpretableAttentionPool,
    CrossModalAttentionBlock
)


class EnhancedMultiOmicsTransformer(nn.Module):
    """
    Enhanced transformer model with interpretability and analysis features.
    
    This model provides a solid foundation for multi-omics integration with:
    - Modality-specific projection layers
    - Transformer encoder with attention tracking
    - Interpretable attention pooling
    - Enhanced classifier head
    """
    
    def __init__(self, input_dims: Dict[str, int], embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 3, num_classes: int = 2, dropout: float = 0.3):
        """
        Args:
            input_dims: Dictionary mapping modality names to feature dimensions
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.modality_names = list(input_dims.keys())
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Modality-specific projection layers with batch normalization
        self.proj_layers = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(input_dims[mod], embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for mod in self.modality_names
        })

        # Modality embeddings for positional encoding
        self.modality_embeddings = nn.Embedding(len(input_dims), embed_dim)

        # Transformer Encoder with custom attention tracking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, 
            batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Interpretable attention pooling
        self.attn_pool = InterpretableAttentionPool(embed_dim)

        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], return_embeddings: bool = False, 
                return_attention: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the enhanced multi-omics transformer.
        
        Args:
            x_dict: Dictionary mapping modality names to input tensors
            return_embeddings: Whether to return intermediate embeddings
            return_attention: Whether to return attention weights
            
        Returns:
            Logits tensor or dictionary with logits and additional information
        """
        batch_size = next(iter(x_dict.values())).shape[0]
        device = next(iter(x_dict.values())).device

        # Project each modality and add modality embeddings
        embedded = []
        modality_embeddings = {}
        
        for i, mod in enumerate(self.modality_names):
            if mod in x_dict:
                proj = self.proj_layers[mod](x_dict[mod])  # (B, embed_dim)
                mod_emb = self.modality_embeddings(torch.tensor(i, device=device))  # (embed_dim,)
                embedded_mod = proj + mod_emb
                embedded.append(embedded_mod)
                modality_embeddings[mod] = proj  # Store pre-transformer embeddings

        # Stack into a sequence: (B, M, E) where M is number of modalities
        x = torch.stack(embedded, dim=1)

        # Apply Transformer encoder
        transformer_output = self.transformer_encoder(x)  # (B, M, E)

        # Attention-based pooling with optional attention weights
        if return_attention:
            pooled_output, attention_weights = self.attn_pool(transformer_output, return_attention=True)
        else:
            pooled_output = self.attn_pool(transformer_output)
            attention_weights = None

        # Classification
        logits = self.classifier(pooled_output)

        # Prepare return values
        results = {"logits": logits}
        
        if return_embeddings:
            results.update({
                "pre_transformer_embeddings": modality_embeddings,
                "transformer_output": transformer_output,
                "pooled_embedding": pooled_output
            })
        
        if return_attention:
            results["attention_weights"] = attention_weights

        return results if (return_embeddings or return_attention) else logits


class AdvancedMultiOmicsTransformer(nn.Module):
    """
    Advanced transformer with MoE, GQA, and flexible sample handling.
    
    This model includes state-of-the-art features:
    - Mixture of Experts for biological pattern specialization
    - Grouped Query Attention for memory efficiency
    - Missing data handling with learned tokens
    - Cross-modal fusion attention
    - Load balancing for optimal expert utilization
    """
    
    def __init__(self, input_dims: Dict[str, int], embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, num_classes: int = 2, num_experts: int = 4, 
                 dropout: float = 0.2, use_moe: bool = True, use_gqa: bool = True, 
                 load_balance_weight: float = 0.01):
        """
        Args:
            input_dims: Dictionary mapping modality names to feature dimensions
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            num_experts: Number of MoE experts
            dropout: Dropout rate
            use_moe: Whether to use Mixture of Experts
            use_gqa: Whether to use Grouped Query Attention
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__()

        self.modality_names = list(input_dims.keys())
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_moe = use_moe
        self.load_balance_weight = load_balance_weight

        # Enhanced modality-specific projection layers
        self.proj_layers = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(input_dims[mod], embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for mod in self.modality_names
        })

        # Learnable modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(input_dims), embed_dim) * 0.02
        )
        
        # Missing data handling
        self.missing_token = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Advanced transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CrossModalAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                use_moe=use_moe,
                use_gqa=use_gqa,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Multi-modal fusion with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, batch: Dict[str, Any], return_embeddings: bool = False, 
                return_attention: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the advanced multi-omics transformer.
        
        Args:
            batch: Batch dictionary containing modality data and optional sample masks
            return_embeddings: Whether to return intermediate embeddings
            return_attention: Whether to return attention information
            
        Returns:
            Logits tensor or dictionary with logits and additional information
        """
        batch_size = len(batch['sample_id']) if 'sample_id' in batch else next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device if isinstance(next(iter(batch.values())), torch.Tensor) else next(self.parameters()).device

        # Project each modality
        modality_embeddings = []
        modality_masks = []
        
        for i, mod in enumerate(self.modality_names):
            if mod in batch:
                # Project modality data
                proj = self.proj_layers[mod](batch[mod])  # (B, embed_dim)
                
                # Add modality embedding
                mod_emb = self.modality_embeddings[i]  # (embed_dim,)
                embedded_mod = proj + mod_emb
                
                # Handle missing data
                if 'sample_masks' in batch and mod in batch['sample_masks']:
                    mask = batch['sample_masks'][mod]  # (B,)
                    # Use missing token for samples without real data
                    embedded_mod = torch.where(
                        mask.unsqueeze(-1),  # (B, 1)
                        embedded_mod,
                        self.missing_token.unsqueeze(0).expand(batch_size, -1)
                    )
                    modality_masks.append(mask)
                else:
                    modality_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))
                
                modality_embeddings.append(embedded_mod)

        # Stack into sequence: (B, M, E) where M is number of modalities
        x = torch.stack(modality_embeddings, dim=1)
        
        # Create attention mask for missing modalities
        modality_mask = torch.stack(modality_masks, dim=1)  # (B, M)

        # Apply transformer blocks
        attention_info = []
        load_balance_losses = []
        
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_info = block(x, return_attention=True)
                attention_info.append(attn_info)
            else:
                x = block(x)
            
            # Collect load balancing losses
            if self.use_moe:
                load_balance_losses.append(block.get_load_balancing_loss())

        # Multi-modal fusion with attention pooling
        # Use self-attention to aggregate across modalities
        fusion_out, fusion_weights = self.fusion_attention(x, x, x)
        
        # Pool across modalities (weighted by availability and attention)
        if modality_mask.all():
            # All modalities available - use attention-weighted pooling
            pooled = torch.mean(fusion_out, dim=1)
        else:
            # Some modalities missing - use masked pooling
            mask_expanded = modality_mask.unsqueeze(-1).float()  # (B, M, 1)
            masked_fusion = fusion_out * mask_expanded
            pooled = masked_fusion.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        # Classification
        logits = self.classifier(pooled)

        # Calculate total load balancing loss
        total_lb_loss = sum(load_balance_losses) if load_balance_losses else torch.tensor(0.0, device=device)

        # Prepare return values
        results = {"logits": logits, "load_balance_loss": total_lb_loss}
        
        if return_embeddings:
            results.update({
                "modality_embeddings": modality_embeddings,
                "transformer_output": x,
                "pooled_embedding": pooled,
                "fusion_weights": fusion_weights
            })
        
        if return_attention:
            results["attention_info"] = attention_info

        return results if (return_embeddings or return_attention) else logits