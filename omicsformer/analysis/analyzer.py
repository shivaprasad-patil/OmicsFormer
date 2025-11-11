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
            # Only set labels if dimensions match
            if len(modality_names) == attention_avg.shape[0] and len(modality_names) == attention_avg.shape[1]:
                ax.set_xticklabels(modality_names, rotation=45)
                ax.set_yticklabels(modality_names, rotation=0)
        elif hasattr(self.model, 'config') and 'modality_names' in self.model.config:
            modality_names = self.model.config['modality_names']
            # Only set labels if dimensions match
            if len(modality_names) == attention_avg.shape[0] and len(modality_names) == attention_avg.shape[1]:
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
            
            # Ensure colors is a 1D array
            if isinstance(colors, list):
                colors = np.array(colors)
            if len(colors.shape) > 1:
                colors = colors.flatten()
            
            # Ensure embedding_2d and colors have compatible shapes
            if len(colors) != embedding_2d.shape[0]:
                print(f"Warning: Shape mismatch - embeddings: {embedding_2d.shape}, labels: {colors.shape}")
                # Truncate to minimum length
                min_len = min(len(colors), embedding_2d.shape[0])
                colors = colors[:min_len]
                embedding_2d = embedding_2d[:min_len]
            
            # Use discrete colors for categorical labels
            unique_labels = np.unique(colors)
            color_map = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = colors == label
                # Ensure mask is 1D boolean array
                if len(mask.shape) > 1:
                    mask = mask.flatten()
                
                # Select points using the boolean mask
                points_x = embedding_2d[mask, 0]
                points_y = embedding_2d[mask, 1]
                
                ax.scatter(points_x, points_y, 
                          c=[color_map(i)], label=f'Class {label}', alpha=0.7)
            
            ax.legend(title='Class Labels')
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
        """
        Compute gradient-based feature importance using input gradients.
        
        Args:
            dataloader: DataLoader with samples to analyze
            target_class: Target class for importance analysis (None for predicted class)
            
        Returns:
            Dictionary with feature importance scores per modality (scaled to [0, 1])
        """
        self.model.eval()
        self.model.zero_grad()
        
        importances = {}
        modality_importance_values = {}
        
        # Collect gradients across all batches
        for batch in dataloader:
            # Move batch to device
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            # Process each modality
            for modality_name, modality_data in batch_device.items():
                if modality_name == 'label' or not isinstance(modality_data, torch.Tensor):
                    continue
                
                # Enable gradient tracking for this modality
                modality_data.requires_grad = True
                batch_device[modality_name] = modality_data
                
                # Forward pass
                logits = self.model(batch_device)
                
                # Get predictions or use target class
                if target_class is not None:
                    preds = torch.full((len(logits),), target_class, dtype=torch.long, device=self.device)
                else:
                    preds = torch.argmax(logits, dim=1)
                
                # Compute gradient of prediction with respect to input
                for i in range(len(preds)):
                    if logits[i, preds[i]].requires_grad:
                        logits[i, preds[i]].backward(retain_graph=True)
                        
                        if modality_data.grad is not None:
                            # Take absolute value of gradients
                            importance = torch.abs(modality_data.grad[i]).detach().cpu().numpy()
                            
                            if modality_name not in modality_importance_values:
                                modality_importance_values[modality_name] = []
                            modality_importance_values[modality_name].append(importance)
                            
                            # Zero gradients for next iteration
                            modality_data.grad.zero_()
                
                # Disable gradient tracking
                modality_data.requires_grad = False
        
        # Average importance across all samples and apply min-max scaling
        for modality_name, importance_list in modality_importance_values.items():
            if len(importance_list) > 0:
                # Average across samples
                mean_importance = np.mean(importance_list, axis=0)
                
                # Apply min-max scaling to [0, 1] range
                min_val, max_val = mean_importance.min(), mean_importance.max()
                if max_val > min_val:
                    scaled_importance = (mean_importance - min_val) / (max_val - min_val)
                else:
                    scaled_importance = np.zeros_like(mean_importance)
                
                importances[modality_name] = scaled_importance
                
                print(f"Computed {modality_name} importance: {len(scaled_importance)} features, scaled range [0.0, 1.0]")
        
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
            
            # Apply min-max scaling to [0, 1] range
            min_val, max_val = modality_importance.min(), modality_importance.max()
            if max_val > min_val:
                modality_importance = (modality_importance - min_val) / (max_val - min_val)
            else:
                modality_importance = np.zeros_like(modality_importance)
            
            importances[f'attention_{key}'] = modality_importance
        
        return importances
    
    def _permutation_based_importance(self, dataloader) -> Dict[str, np.ndarray]:
        """Compute permutation-based feature importance."""
        # Get baseline performance
        baseline_acc = self._compute_accuracy(dataloader)
        
        importances = {}
        raw_importances = {}
        modality_names = getattr(self.model, 'modality_names', list(range(10)))  # Fallback
        
        for modality in modality_names:
            # Permute this modality and compute performance drop
            permuted_acc = self._compute_accuracy_with_permuted_modality(dataloader, modality)
            importance = baseline_acc - permuted_acc
            raw_importances[modality] = importance
        
        # Apply min-max scaling to [0, 1] range across all modalities
        if raw_importances:
            all_values = np.array(list(raw_importances.values()))
            min_val, max_val = all_values.min(), all_values.max()
            
            for modality, importance in raw_importances.items():
                if max_val > min_val:
                    scaled_importance = (importance - min_val) / (max_val - min_val)
                else:
                    scaled_importance = 0.0
                importances[modality] = np.array([scaled_importance])
        
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
    
    def generate_analysis_report(self, save_path: str = 'omicsformer_analysis_report.html',
                               include_plots: bool = True, plot_paths: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a comprehensive HTML analysis report with embedded plots.
        
        Args:
            save_path: Path to save the HTML report
            include_plots: Whether to include plot images in the report
            plot_paths: Dictionary mapping plot types to file paths
            
        Returns:
            HTML content as string
        """
        # Prepare plot content
        plots_html = ""
        if include_plots and plot_paths:
            plots_html = self._generate_plots_html(plot_paths)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OmicsFormer Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .plot-container {{ text-align: center; margin: 20px 0; }}
                .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
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
            
            {plots_section}
            
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
            feature_importance=list(self.feature_importances.keys()),
            plots_section=plots_html
        )
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        return html_content
    
    def _generate_plots_html(self, plot_paths: Dict[str, str]) -> str:
        """Generate HTML for embedding plots in the report."""
        import base64
        import os
        
        plots_html = '<div class="section"><h2>Analysis Visualizations</h2>'
        
        plot_titles = {
            'attention': 'Attention Patterns',
            'embeddings_pca': 'PCA Embeddings',
            'embeddings_tsne': 't-SNE Embeddings', 
            'embeddings_umap': 'UMAP Embeddings',
            'correlations': 'Cross-Modal Correlations',
            'distributions': 'Feature Distributions'
        }
        
        # Add pathway plots
        for modality in ['genomics', 'transcriptomics', 'proteomics', 'metabolomics']:
            plot_titles[f'pathways_{modality}'] = f'{modality.title()} Pathway Enrichment'
        
        # Handle both list and dictionary formats for plot_paths
        if isinstance(plot_paths, dict):
            # Dictionary format: {key: path}
            plot_items = plot_paths.items()
        else:
            # List format: [path1, path2, ...]
            plot_items = [(os.path.basename(path).replace('.png', ''), path) for path in plot_paths]
        
        for plot_key, plot_path in plot_items:
            if os.path.exists(plot_path):
                try:
                    with open(plot_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        
                    title = plot_titles.get(plot_key.replace('test_', ''), plot_key.replace('test_', '').title())
                    
                    plots_html += f'''
                    <div class="plot-container">
                        <h3>{title}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{title}">
                    </div>
                    '''
                except Exception as e:
                    print(f"Warning: Could not embed plot {plot_path}: {e}")
        
        plots_html += '</div>'
        return plots_html
    
    def analyze_pathway_enrichment(self, feature_importance_results: Dict[str, np.ndarray], 
                                 feature_names: Optional[Dict[str, List[str]]] = None,
                                 top_k: int = 50, method: str = 'hypergeometric') -> Dict[str, pd.DataFrame]:
        """
        Perform biological pathway enrichment analysis on important features.
        
        Args:
            feature_importance_results: Results from analyze_feature_importance
            feature_names: Dictionary mapping modality names to feature name lists
            top_k: Number of top important features to consider per modality
            method: Enrichment method ('hypergeometric', 'fisher', 'gsea')
            
        Returns:
            Dictionary with enrichment results per modality
        """
        try:
            # Try importing enrichment libraries
            import gseapy as gp
            from gseapy.plot import barplot, dotplot
            GSEA_AVAILABLE = True
            print("gseapy is available. Using real pathway enrichment analysis.")
        except ImportError:
            GSEA_AVAILABLE = False
            print("Warning: gseapy not available. Using mock pathway enrichment.")
        
        enrichment_results = {}
        
        for modality, importance_scores in feature_importance_results.items():
            # Get top important features
            if len(importance_scores.shape) > 1:
                # Multiple importance values per feature, take mean
                importance_scores = importance_scores.mean(axis=0)
            
            # Get indices of top important features (scores are in [0, 1] range)
            top_indices = np.argsort(importance_scores)[-top_k:][::-1]
            
            print(f"  {modality}: Using top {len(top_indices)} features from {len(importance_scores)} total")
            print(f"    Scaled importance range: [0.0, 1.0]")
            print(f"    Top feature importance: {importance_scores[top_indices[0]]:.3f}")
            
            # Get feature names
            if feature_names and modality in feature_names:
                top_features = [feature_names[modality][i] for i in top_indices if i < len(feature_names[modality])]
            else:
                top_features = [f"{modality}_feature_{i}" for i in top_indices]
            
            if GSEA_AVAILABLE and len(top_features) > 5:
                try:
                    # Clean feature names (remove any numeric suffixes from duplicated names)
                    clean_features = []
                    for feature in top_features:
                        # Remove numeric suffixes that might have been added for duplicates
                        clean_feature = feature.split('_')[0] if '_' in feature and feature.split('_')[-1].isdigit() else feature
                        clean_features.append(clean_feature)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_features = []
                    for item in clean_features:
                        if item not in seen:
                            seen.add(item)
                            unique_features.append(item)
                    
                    print(f"Running real pathway enrichment for {modality} with {len(unique_features)} genes: {unique_features[:10]}")
                    
                    # Perform pathway enrichment using real databases
                    if modality.lower() in ['genomics', 'transcriptomics', 'rnaseq']:
                        # Use gene-related databases
                        databases = ['KEGG_2021_Human', 'GO_Biological_Process_2021']
                    elif modality.lower() in ['proteomics', 'protein']:
                        # Use protein-related databases  
                        databases = ['KEGG_2021_Human', 'GO_Biological_Process_2021']
                    else:
                        # Default databases for metabolomics (use enzyme genes)
                        databases = ['KEGG_2021_Human', 'GO_Molecular_Function_2021']
                    
                    # Run enrichment analysis
                    enr_results = []
                    for db in databases:
                        try:
                            print(f"  Trying database: {db}")
                            enr = gp.enrichr(gene_list=unique_features,
                                           gene_sets=db,
                                           organism='Human',
                                           cutoff=0.1,  # More lenient cutoff
                                           background=None)
                            
                            if not enr.results.empty:
                                print(f"    Found {len(enr.results)} pathways in {db}")
                                enr.results['Database'] = db
                                enr_results.append(enr.results)
                            else:
                                print(f"    No significant pathways found in {db}")
                                
                        except Exception as e:
                            print(f"    Warning: Could not run enrichment for {db}: {e}")
                            continue
                    
                    if enr_results:
                        combined_results = pd.concat(enr_results, ignore_index=True)
                        # Sort by adjusted p-value
                        combined_results = combined_results.sort_values('Adjusted P-value')
                        enrichment_results[modality] = combined_results.head(15)  # Top 15 pathways
                        print(f"  ✓ Real pathway enrichment completed for {modality}: {len(combined_results)} pathways found")
                    else:
                        print(f"  No enrichment results found, using mock data for {modality}")
                        enrichment_results[modality] = self._create_mock_pathway_results(top_features, modality)
                        
                except Exception as e:
                    print(f"  Warning: Real pathway enrichment failed for {modality}: {e}")
                    print(f"  Falling back to mock results")
                    enrichment_results[modality] = self._create_mock_pathway_results(top_features, modality)
            else:
                # Create mock results for demonstration or when gseapy not available
                print(f"Using mock pathway enrichment for {modality} (gseapy available: {GSEA_AVAILABLE}, features: {len(top_features)})")
                enrichment_results[modality] = self._create_mock_pathway_results(top_features, modality)
        
        return enrichment_results
    
    def plot_pathway_enrichment(self, enrichment_results: Dict[str, pd.DataFrame], 
                              modality: str, top_n: int = 10, 
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pathway enrichment results.
        
        Args:
            enrichment_results: Results from analyze_pathway_enrichment
            modality: Modality to plot results for
            top_n: Number of top pathways to plot
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if modality not in enrichment_results:
            raise ValueError(f"No enrichment results found for modality: {modality}")
        
        df = enrichment_results[modality].head(top_n)
        
        if df.empty:
            print(f"No significant pathways found for {modality}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        if 'Adjusted P-value' in df.columns:
            pvalues = -np.log10(df['Adjusted P-value'] + 1e-10)
            ylabel = '-log10(Adjusted P-value)'
        elif 'P-value' in df.columns:
            pvalues = -np.log10(df['P-value'] + 1e-10)
            ylabel = '-log10(P-value)'
        else:
            pvalues = df['Score'] if 'Score' in df.columns else range(len(df))
            ylabel = 'Enrichment Score'
        
        pathway_names = df['Term'] if 'Term' in df.columns else df.index
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(pathway_names)), pvalues, color='steelblue', alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(pathway_names)))
        ax.set_yticklabels([name[:50] + '...' if len(name) > 50 else name for name in pathway_names])
        ax.set_xlabel(ylabel)
        ax.set_title(f'Pathway Enrichment - {modality.title()}')
        ax.invert_yaxis()  # Top pathways at the top
        
        # Add significance threshold line if p-values
        if 'P-value' in ylabel:
            ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _create_mock_pathway_results(self, top_features: List[str], modality: str) -> pd.DataFrame:
        """Create mock pathway enrichment results for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        # Mock pathway names based on modality
        if modality.lower() in ['genomics', 'transcriptomics']:
            pathways = [
                "Cell cycle regulation", "DNA repair", "Apoptosis signaling",
                "p53 pathway", "MAPK signaling", "PI3K-Akt pathway",
                "Wnt signaling", "TGF-beta signaling", "JAK-STAT pathway",
                "NF-kappa B signaling"
            ]
        elif modality.lower() in ['proteomics', 'protein']:
            pathways = [
                "Protein folding", "Proteasomal degradation", "Metabolic pathways",
                "Signal transduction", "Cytoskeletal organization", "Membrane transport",
                "Enzyme regulation", "Protein phosphorylation", "Ubiquitination",
                "Protein-protein interactions"
            ]
        else:
            pathways = [
                "Metabolic regulation", "Cellular signaling", "Gene expression",
                "Protein synthesis", "Cell division", "Stress response",
                "Immune response", "Development", "Differentiation",
                "Homeostasis"
            ]
        
        # Create mock results
        n_pathways = min(len(pathways), 8)
        
        # Create gene lists for each pathway
        gene_lists = []
        for i in range(n_pathways):
            # Ensure we have at least 2 features and don't exceed available features
            max_genes = min(8, len(top_features))
            min_genes = min(2, max_genes)
            n_genes = np.random.randint(min_genes, max_genes + 1) if max_genes > min_genes else min_genes
            gene_lists.append(';'.join(top_features[:n_genes]))
        
        mock_results = pd.DataFrame({
            'Term': pathways[:n_pathways],
            'P-value': np.random.exponential(0.01, n_pathways),
            'Adjusted P-value': np.random.exponential(0.02, n_pathways),
            'Score': np.random.exponential(5, n_pathways),
            'Combined Score': np.random.exponential(10, n_pathways),
            'Genes': gene_lists
        })
        
        # Sort by p-value
        mock_results = mock_results.sort_values('P-value')
        
        return mock_results

    def plot_feature_importance(self, importance_results: Dict[str, np.ndarray], 
                               modality: str, feature_names: Optional[List[str]] = None,
                               top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance results for a specific modality.
        
        Args:
            importance_results: Results from analyze_feature_importance
            modality: Modality to plot results for
            feature_names: Names of features (if None, uses indices)
            top_n: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if modality not in importance_results:
            raise ValueError(f"No importance results found for modality: {modality}")
        
        importance_scores = importance_results[modality]
        
        # Handle multi-dimensional importance scores
        if len(importance_scores.shape) > 1:
            importance_scores = importance_scores.mean(axis=0)
        
        # Apply min-max scaling to ensure [0, 1] range
        min_val, max_val = importance_scores.min(), importance_scores.max()
        if max_val > min_val:
            importance_scores = (importance_scores - min_val) / (max_val - min_val)
        
        # Get top important features (use absolute values for ranking if needed)
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
        top_scores = importance_scores[top_indices]
        
        # Get feature names
        if feature_names is not None:
            top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                        for i in top_indices]
        else:
            top_names = [f'Feature_{i}' for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_names)), top_scores, color='steelblue', alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_names])
        ax.set_xlabel('Scaled Feature Importance (0-1)')
        ax.set_title(f'Feature Importance - {modality.title()} (Min-Max Scaled)')
        ax.set_xlim([0, 1.05])  # Set x-axis range for scaled values
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # Top features at the top
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(bar.get_width() + max(top_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


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