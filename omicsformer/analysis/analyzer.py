"""
Analysis and interpretation tools for multi-omics transformers.

This module provides comprehensive analysis capabilities including:
- Attention visualization and interpretation
- Feature importance analysis
- Cross-modal relationship analysis
- Biological pathway enrichment
- Model interpretability tools
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx


class MultiOmicsAnalyzer:
    """
    Comprehensive analyzer for multi-omics transformer models.
    
    Provides tools for:
    - Attention pattern analysis
    - Feature importance extraction
    - Cross-modal interaction analysis
    - Dimensionality reduction and visualization
    - Statistical analysis of omics relationships
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """
        Args:
            model: Trained multi-omics transformer model
            device: Device to use for computations
        """
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for analysis results
        self.attention_patterns = {}
        self.feature_importances = {}
        self.embeddings = {}
    
    def extract_attention_patterns(self, dataloader, max_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Extract attention patterns from the model.
        
        Args:
            dataloader: DataLoader with samples to analyze
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with attention patterns for different layers/heads
        """
        attention_patterns = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Get attention weights
                outputs = self.model(batch, return_attention=True)
                
                if isinstance(outputs, dict) and 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights']
                    
                    if isinstance(attention_weights, torch.Tensor):
                        attention_patterns.append(attention_weights.cpu().numpy())
                    elif isinstance(attention_weights, list):
                        # Multiple attention layers
                        for i, att in enumerate(attention_weights):
                            if len(attention_patterns) <= i:
                                attention_patterns.append([])
                            attention_patterns[i].append(att.cpu().numpy())
                
                sample_count += batch['sample_id'].shape[0] if isinstance(batch['sample_id'], torch.Tensor) else len(batch['sample_id'])
        
        # Concatenate patterns
        if attention_patterns and isinstance(attention_patterns[0], np.ndarray):
            # Single attention layer
            self.attention_patterns['main'] = np.concatenate(attention_patterns, axis=0)
        elif attention_patterns:
            # Multiple attention layers
            for i, layer_patterns in enumerate(attention_patterns):
                if layer_patterns:
                    self.attention_patterns[f'layer_{i}'] = np.concatenate(layer_patterns, axis=0)
        
        return self.attention_patterns
    
    def extract_embeddings(self, dataloader, embedding_type: str = 'pooled') -> Dict[str, np.ndarray]:
        """
        Extract embeddings from different layers of the model.
        
        Args:
            dataloader: DataLoader with samples to analyze
            embedding_type: Type of embeddings to extract ('pooled', 'transformer', 'modality')
            
        Returns:
            Dictionary with embeddings and corresponding labels
        """
        embeddings = []
        labels = []
        sample_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Get embeddings
                outputs = self.model(batch, return_embeddings=True)
                
                if isinstance(outputs, dict):
                    if embedding_type == 'pooled' and 'pooled_embedding' in outputs:
                        emb = outputs['pooled_embedding']
                    elif embedding_type == 'transformer' and 'transformer_output' in outputs:
                        emb = outputs['transformer_output'].mean(dim=1)  # Average across modalities
                    elif embedding_type == 'modality' and 'pre_transformer_embeddings' in outputs:
                        # Concatenate modality embeddings
                        mod_embs = outputs['pre_transformer_embeddings']
                        emb = torch.cat([mod_embs[mod] for mod in sorted(mod_embs.keys())], dim=1)
                    else:
                        # Fallback to logits
                        emb = outputs.get('logits', outputs.get('pooled_embedding'))
                    
                    if emb is not None:
                        embeddings.append(emb.cpu().numpy())
                        
                        if 'label' in batch:
                            labels.extend(batch['label'].cpu().numpy())
                        
                        if 'sample_id' in batch:
                            if isinstance(batch['sample_id'], torch.Tensor):
                                sample_ids.extend(batch['sample_id'].cpu().numpy())
                            else:
                                sample_ids.extend(batch['sample_id'])
        
        if embeddings:
            self.embeddings[embedding_type] = {
                'embeddings': np.concatenate(embeddings, axis=0),
                'labels': np.array(labels) if labels else None,
                'sample_ids': sample_ids
            }
        
        return self.embeddings
    
    def visualize_attention_heatmap(self, attention_key: str = 'main', figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap visualization of attention patterns.
        
        Args:
            attention_key: Key for attention patterns to visualize
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if attention_key not in self.attention_patterns:
            raise ValueError(f"Attention patterns for '{attention_key}' not found. Run extract_attention_patterns first.")
        
        attention = self.attention_patterns[attention_key]
        
        # Average attention across samples and heads if multi-dimensional
        if len(attention.shape) > 2:
            if len(attention.shape) == 4:  # (samples, heads, seq_len, seq_len)
                attention_avg = attention.mean(axis=(0, 1))
            elif len(attention.shape) == 3:  # (samples, seq_len, seq_len)
                attention_avg = attention.mean(axis=0)
            else:
                attention_avg = attention.reshape(-1, attention.shape[-1])
        else:
            attention_avg = attention
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(attention_avg, annot=True, cmap='Blues', fmt='.3f', ax=ax)
        ax.set_title(f'Attention Patterns - {attention_key}')
        ax.set_xlabel('Attention To (Modalities)')
        ax.set_ylabel('Attention From (Modalities)')
        
        # Add modality labels if available
        if hasattr(self.model, 'modality_names'):
            modality_names = self.model.modality_names
            ax.set_xticklabels(modality_names, rotation=45)
            ax.set_yticklabels(modality_names, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_embedding_visualization(self, embedding_type: str = 'pooled', method: str = 'umap',
                                   color_by: str = 'label', figsize: Tuple[int, int] = (10, 8),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2D visualization of embeddings using dimensionality reduction.
        
        Args:
            embedding_type: Type of embeddings to visualize
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            color_by: What to color points by ('label', 'sample_id')
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if embedding_type not in self.embeddings:
            raise ValueError(f"Embeddings for '{embedding_type}' not found. Run extract_embeddings first.")
        
        emb_data = self.embeddings[embedding_type]
        embeddings = emb_data['embeddings']
        
        # Apply dimensionality reduction
        if method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embedding_2d = reducer.fit_transform(embeddings)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color points
        if color_by == 'label' and emb_data['labels'] is not None:
            colors = emb_data['labels']
            scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Class Label')
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(f'{embedding_type.title()} Embeddings - {method.upper()} Visualization')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_cross_modal_correlations(self, dataloader, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlations between different modalities.
        
        Args:
            dataloader: DataLoader with samples to analyze
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            DataFrame with correlation matrix
        """
        modality_data = {}
        modality_names = []
        
        # Collect data from all samples
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Get modality embeddings
                outputs = self.model(batch, return_embeddings=True)
                
                if isinstance(outputs, dict) and 'pre_transformer_embeddings' in outputs:
                    mod_embs = outputs['pre_transformer_embeddings']
                    
                    for mod, emb in mod_embs.items():
                        if mod not in modality_data:
                            modality_data[mod] = []
                            if mod not in modality_names:
                                modality_names.append(mod)
                        
                        modality_data[mod].append(emb.mean(dim=1).cpu().numpy())  # Average across features
        
        # Concatenate data
        for mod in modality_data:
            modality_data[mod] = np.concatenate(modality_data[mod], axis=0)
        
        # Compute correlations
        correlations = np.zeros((len(modality_names), len(modality_names)))
        
        for i, mod1 in enumerate(modality_names):
            for j, mod2 in enumerate(modality_names):
                if method.lower() == 'pearson':
                    corr, _ = pearsonr(modality_data[mod1], modality_data[mod2])
                elif method.lower() == 'spearman':
                    corr, _ = spearmanr(modality_data[mod1], modality_data[mod2])
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                correlations[i, j] = corr
        
        # Create DataFrame
        correlation_df = pd.DataFrame(
            correlations,
            index=modality_names,
            columns=modality_names
        )
        
        return correlation_df
    
    def plot_correlation_heatmap(self, correlation_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(correlation_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', ax=ax, square=True)
        ax.set_title('Cross-Modal Correlations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_feature_importance(self, dataloader, method: str = 'gradient', 
                                 target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance using various methods.
        
        Args:
            dataloader: DataLoader with samples to analyze
            method: Method for importance analysis ('gradient', 'attention', 'permutation')
            target_class: Target class for importance analysis (None for predicted class)
            
        Returns:
            Dictionary with feature importance scores per modality
        """
        feature_importances = {}
        
        if method == 'gradient':
            feature_importances = self._gradient_based_importance(dataloader, target_class)
        elif method == 'attention':
            feature_importances = self._attention_based_importance(dataloader)
        elif method == 'permutation':
            feature_importances = self._permutation_based_importance(dataloader)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        self.feature_importances.update(feature_importances)
        return feature_importances
    
    def _gradient_based_importance(self, dataloader, target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compute gradient-based feature importance."""
        self.model.train()  # Enable gradients
        
        importances = {}
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            
            # Enable gradients for input
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    batch[key] = value.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(batch)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get target class
            if target_class is None:
                target_class_batch = torch.argmax(logits, dim=1)
            else:
                target_class_batch = torch.full((logits.shape[0],), target_class, device=self.device)
            
            # Compute gradients
            loss = nn.CrossEntropyLoss()(logits, target_class_batch)
            loss.backward()
            
            # Extract gradients
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.grad is not None:
                    if key not in importances:
                        importances[key] = []
                    importances[key].append(torch.abs(value.grad).mean(dim=0).cpu().numpy())
            
            # Clear gradients
            self.model.zero_grad()
            break  # Just use first batch for now
        
        # Average importances
        for key in importances:
            importances[key] = np.mean(importances[key], axis=0)
        
        self.model.eval()
        return importances
    
    def _attention_based_importance(self, dataloader) -> Dict[str, np.ndarray]:
        """Compute attention-based feature importance."""
        if not self.attention_patterns:
            self.extract_attention_patterns(dataloader)
        
        importances = {}
        
        # Use attention weights as importance scores
        for key, attention in self.attention_patterns.items():
            # Average attention across samples and heads
            if len(attention.shape) > 2:
                attention_avg = attention.mean(axis=tuple(range(len(attention.shape)-2)))
            else:
                attention_avg = attention
            
            # Sum attention received by each modality
            modality_importance = attention_avg.sum(axis=0)
            importances[f'attention_{key}'] = modality_importance
        
        return importances
    
    def _permutation_based_importance(self, dataloader) -> Dict[str, np.ndarray]:
        """Compute permutation-based feature importance."""
        # Get baseline performance
        baseline_acc = self._compute_accuracy(dataloader)
        
        importances = {}
        modality_names = getattr(self.model, 'modality_names', list(range(10)))  # Fallback
        
        for modality in modality_names:
            # Permute this modality and compute performance drop
            permuted_acc = self._compute_accuracy_with_permuted_modality(dataloader, modality)
            importance = baseline_acc - permuted_acc
            importances[modality] = np.array([importance])
        
        return importances
    
    def _compute_accuracy(self, dataloader) -> float:
        """Compute model accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                if 'label' in batch:
                    labels = batch['label'].squeeze()
                    outputs = self.model(batch)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    predicted = torch.argmax(logits, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _compute_accuracy_with_permuted_modality(self, dataloader, modality: str) -> float:
        """Compute accuracy with one modality permuted."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Permute the specified modality
                if modality in batch:
                    batch_size = batch[modality].shape[0]
                    perm_indices = torch.randperm(batch_size)
                    batch[modality] = batch[modality][perm_indices]
                
                if 'label' in batch:
                    labels = batch['label'].squeeze()
                    outputs = self.model(batch)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    predicted = torch.argmax(logits, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
            else:
                moved_batch[key] = value
        return moved_batch
    
    def generate_analysis_report(self, save_path: str = 'omicsformer_analysis_report.html') -> str:
        """
        Generate a comprehensive HTML analysis report.
        
        Args:
            save_path: Path to save the HTML report
            
        Returns:
            HTML content as string
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OmicsFormer Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; }
                .metric { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>OmicsFormer Analysis Report</h1>
            
            <div class="section">
                <h2>Model Information</h2>
                <div class="metric">
                    <strong>Model Type:</strong> {model_type}
                </div>
                <div class="metric">
                    <strong>Device:</strong> {device}
                </div>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <div class="metric">
                    <strong>Attention Patterns Extracted:</strong> {attention_patterns}
                </div>
                <div class="metric">
                    <strong>Embeddings Extracted:</strong> {embeddings_types}
                </div>
                <div class="metric">
                    <strong>Feature Importance Analysis:</strong> {feature_importance}
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Use attention visualizations to understand cross-modal interactions</li>
                    <li>Analyze feature importance to identify key biomarkers</li>
                    <li>Use embedding visualizations for sample clustering analysis</li>
                    <li>Consider correlation analysis for modality relationship insights</li>
                </ul>
            </div>
        </body>
        </html>
        """.format(
            model_type=self.model.__class__.__name__,
            device=str(self.device),
            attention_patterns=list(self.attention_patterns.keys()),
            embeddings_types=list(self.embeddings.keys()),
            feature_importance=list(self.feature_importances.keys())
        )
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        return html_content


def compute_modality_statistics(modality_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute basic statistics for each modality.
    
    Args:
        modality_data: Dictionary mapping modality names to DataFrames
        
    Returns:
        DataFrame with statistics for each modality
    """
    stats = []
    
    for modality, df in modality_data.items():
        stat = {
            'Modality': modality,
            'Samples': len(df),
            'Features': len(df.columns),
            'Missing_Values': df.isnull().sum().sum(),
            'Mean_Feature_Mean': df.mean().mean(),
            'Mean_Feature_Std': df.std().mean(),
            'Min_Value': df.min().min(),
            'Max_Value': df.max().max()
        }
        stats.append(stat)
    
    return pd.DataFrame(stats)


def plot_modality_distributions(modality_data: Dict[str, pd.DataFrame], 
                               sample_features: int = 5, figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of features across modalities.
    
    Args:
        modality_data: Dictionary mapping modality names to DataFrames
        sample_features: Number of features to sample per modality
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    n_modalities = len(modality_data)
    fig, axes = plt.subplots(n_modalities, 1, figsize=figsize)
    
    if n_modalities == 1:
        axes = [axes]
    
    for i, (modality, df) in enumerate(modality_data.items()):
        # Sample features
        sampled_features = df.columns[:sample_features] if len(df.columns) >= sample_features else df.columns
        
        # Plot distributions
        for feature in sampled_features:
            axes[i].hist(df[feature].dropna(), alpha=0.7, bins=30, label=feature)
        
        axes[i].set_title(f'{modality} Feature Distributions')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig