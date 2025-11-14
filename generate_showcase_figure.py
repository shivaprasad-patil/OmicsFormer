"""
OmicsFormer Showcase Figure Generator

Creates a comprehensive visualization showcasing the OmicsFormer framework:
- Multi-study data integration
- Batch correction effects
- Transformer architecture
- Classification results
- Feature importance
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")

# Create figure with custom layout - architecture on top, examples on bottom
fig = plt.figure(figsize=(20, 10), facecolor='white')
gs = GridSpec(2, 4, figure=fig, hspace=0.20, wspace=0.35,
              left=0.05, right=0.95, top=0.92, bottom=0.08)

# Color scheme
COLORS = {
    'genomics': '#3B82F6',
    'transcriptomics': '#10B981',
    'proteomics': '#8B5CF6',
    'metabolomics': '#F59E0B',
    'neural': '#EC4899',
    'batch': '#EF4444',
    'corrected': '#22C55E'
}

# ============================================================================
# TITLE
# ============================================================================
fig.text(0.5, 0.97, 'OmicsFormer: Multi-Omics & Multi-Study Integration', 
         ha='center', va='top', fontsize=20, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))


# ============================================================================
# SECTION 1: TRANSFORMER ARCHITECTURE (Top - Full Width)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Transformer Architecture: General Framework for Multi-Omics Integration', 
              fontweight='bold', fontsize=11, pad=10)

ax1.axis('off')
ax1.set_xlim(0, 13)
ax1.set_ylim(0, 10)

# Input modalities (left side) - only 3 omics, no Study ID
input_y = 7.5
modalities = ['Omics 1', 'Omics 2', 'Omics 3']
colors = [COLORS['genomics'], COLORS['transcriptomics'], 
          COLORS['proteomics']]

for i, (mod, color) in enumerate(zip(modalities, colors)):
    y = input_y - i * 0.7
    rect = FancyBboxPatch((0.3, y), 1.3, 0.45, boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor=color, alpha=0.4, linewidth=2.5)
    ax1.add_patch(rect)
    ax1.text(0.95, y + 0.225, mod, ha='center', va='center', 
            fontsize=10, fontweight='bold')
    # Arrow to embedding
    ax1.annotate('', xy=(2.0, 6.7 + (2-i)*0.12), xytext=(1.65, y + 0.225),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='gray', alpha=0.7))

# Embedding layer
emb_rect = FancyBboxPatch((2.0, 5.8), 1.7, 2.2, boxstyle="round,pad=0.1",
                          edgecolor=COLORS['neural'], facecolor=COLORS['neural'],
                          alpha=0.3, linewidth=2.5)
ax1.add_patch(emb_rect)
ax1.text(2.85, 7.15, 'Embedding\nLayer', ha='center', va='center',
        fontsize=10, fontweight='bold')
ax1.text(2.85, 6.4, 'd=49', ha='center', va='center', fontsize=9, style='italic')

# Arrow to transformer layers
ax1.annotate('', xy=(4.1, 6.9), xytext=(3.8, 6.9),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Transformer layers (3 blocks)
for i in range(3):
    x = 4.1 + i * 2.2
    
    # Multi-head attention
    attn_rect = FancyBboxPatch((x, 7.4), 1.8, 0.85, boxstyle="round,pad=0.05",
                              edgecolor='#2563EB', facecolor='#DBEAFE', linewidth=2.5)
    ax1.add_patch(attn_rect)
    ax1.text(x + 0.9, 7.95, 'Multi-Head', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(x + 0.9, 7.65, 'Attention', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Feed-forward
    ff_rect = FancyBboxPatch((x, 6.1), 1.8, 0.85, boxstyle="round,pad=0.05",
                            edgecolor='#059669', facecolor='#D1FAE5', linewidth=2.5)
    ax1.add_patch(ff_rect)
    ax1.text(x + 0.9, 6.65, 'Feed-', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(x + 0.9, 6.35, 'Forward', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Vertical arrow between attention and feedforward
    ax1.annotate('', xy=(x + 0.9, 6.95), xytext=(x + 0.9, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Horizontal arrow to next layer
    if i < 2:
        ax1.annotate('', xy=(x + 2.2, 6.9), xytext=(x + 1.8, 6.9),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Output classification (right side)
output_rect = FancyBboxPatch((10.7, 6.35), 1.7, 1.3, boxstyle="round,pad=0.1",
                            edgecolor='#DC2626', facecolor='#FECACA',
                            alpha=1.0, linewidth=2.5)
ax1.add_patch(output_rect)
ax1.text(11.55, 7.3, 'Classification', ha='center', va='center',
        fontsize=10, fontweight='bold')
ax1.text(11.55, 6.85, 'Layer', ha='center', va='center',
        fontsize=10, fontweight='bold')

# Arrow from last transformer to output
ax1.annotate('', xy=(10.7, 7.0), xytext=(10.5, 6.9),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Add layer count annotation
ax1.text(7.4, 5.3, 'L = 3 Layers', ha='center', va='center',
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', linewidth=1.5))

# Add parameter count
ax1.text(7.4, 4.7, '726K Parameters', ha='center', va='center',
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', linewidth=1.5))

# ============================================================================
# SECTION 2: MULTI-STUDY INPUT DATA (Bottom Left - Smaller)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Example: Multi-Study SLE Analysis', fontweight='bold', fontsize=10)

# Simulate multi-study data with batch effects
np.random.seed(42)
n_samples = 550
n_features = 50  # Reduced for visualization

# Create 8 studies with batch effects
studies = np.array([0]*65 + [1]*70 + [2]*68 + [3]*72 + 
                   [4]*75 + [5]*60 + [6]*70 + [7]*70)
data = np.zeros((n_samples, n_features))

for study in range(8):
    mask = studies == study
    n = mask.sum()
    # Add study-specific batch effect
    batch_effect = np.random.randn(1, n_features) * 0.5 + study * 0.3
    data[mask] = np.random.randn(n, n_features) + batch_effect

# Create smaller heatmap
im2 = ax2.imshow(data[:100].T, aspect='auto', cmap='RdYlBu_r', 
                 vmin=-2, vmax=2, interpolation='nearest')
ax2.set_ylabel('Features', fontsize=8)
ax2.set_xlabel('Samples', fontsize=8)
ax2.set_yticks([0, 25, 49])
ax2.set_yticklabels(['1', '1000', '2000'], fontsize=7)
ax2.set_xticks([0, 50, 99])
ax2.set_xticklabels(['1', '50', '100'], fontsize=7)

# Add study indicators
study_colors = plt.cm.tab10(np.linspace(0, 1, 8))
for i, study_id in enumerate(['SRP062966', 'SRP095109', 'SRP131775', 'SRP132939'][:4]):
    ax2.text(1.12, 0.85 - i*0.22, f'• {study_id}', 
            transform=ax2.transAxes, fontsize=7, color=study_colors[i])

# ============================================================================
# SECTION 3: DISEASE CLASSIFICATION (Bottom Center-Left)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Disease Classification', fontweight='bold', fontsize=11)

# Add metrics box at the top
ax3.text(0.5, 1.12, 'Accuracy: 90.9%  |  F1: 0.91', transform=ax3.transAxes,
        ha='center', va='bottom', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', linewidth=2))

# Confusion matrix
cm = np.array([[39, 2], [8, 61]])
im = ax3.imshow(cm, cmap='Blues', alpha=0.6)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center", fontsize=20, fontweight='bold',
                       color='white' if cm[i, j] > 30 else 'black')

ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Control', 'SLE'], fontsize=10)
ax3.set_yticklabels(['Control', 'SLE'], fontsize=10)
ax3.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax3.set_ylabel('True', fontsize=10, fontweight='bold')

# ============================================================================
# SECTION 4: FEATURE IMPORTANCE (Bottom Right)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2:4])
ax4.set_title('Top Biomarkers', fontweight='bold', fontsize=11, pad=8)

# Top genes and their importance scores - reduced to 9 genes
genes = ['TNFSF13B', 'TPM2', 'PNLIPPP3', 'SLIT2', 'COL1A2', 
         'CTS2', 'KRT16', 'ACOT7', 'NPAS2']
importance = np.array([1.000, 0.983, 0.991, 0.973, 0.962, 
                       0.945, 0.926, 0.924, 0.919])

# Create color gradient - darker red
colors_grad = ['#8B0000'] * len(genes)  # Dark red for all bars

# Horizontal bar chart
bars = ax4.barh(range(len(genes)), importance, color=colors_grad, 
                edgecolor='white', linewidth=1.5)

ax4.set_yticks(range(len(genes)))
ax4.set_yticklabels(genes, fontsize=11, fontweight='bold')
ax4.set_xlabel('Importance Score', fontsize=10)
ax4.set_xlim(0, 1.05)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add value labels
for i, (gene, imp) in enumerate(zip(genes, importance)):
    ax4.text(1.02, i, f'{imp:.3f}', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# FOOTER
# ============================================================================
footer_text = (
    'GitHub: github.com/shivaprasad-patil/omicsformer  |  '
    'License: Apache 2.0  |  Framework: PyTorch 2.0+'
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, 
         style='italic', color='#666')

# ============================================================================
# SAVE FIGURE
# ============================================================================
plt.savefig('omicsformer_figure.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('omicsformer_figure.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Figure saved as 'omicsformer_figure.png' and 'omicsformer_figure.pdf'")

# Also save as lower resolution for web
plt.savefig('omicsformer_figure_web.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Web version saved as 'omicsformer_figure_web.png'")

# plt.show()  # Commented out to prevent hanging
