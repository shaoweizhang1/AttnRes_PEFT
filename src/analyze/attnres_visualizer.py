"""
Visualization utilities for Attention Residual adapter analysis.

Generates plots and visualizations for understanding attention patterns across layers and datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AttnResVisualizer:
    """Generate visualizations of attention residual usage patterns."""
    
    def __init__(self, output_dir: str = "./attnres_analysis", dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set style
        sns.set_style("white")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_layer_depth_attention(self, stats: Dict, num_layers: int, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot heatmap of which layers attend to which earlier layers.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            num_layers: Total number of layers
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        
        # Build attention matrix with explicit embedding row:
        # row 0 = Embedding (-1), rows 1.. = Layer 0..N-1
        attention_matrix = np.full((num_layers + 1, num_layers), np.nan, dtype=float)
        
        layer_stats = stats.get('layer_depth_stats', {})
        for layer_idx, layer_stat in layer_stats.items():
            if layer_stat.get('aggregated'):
                depth_attn = layer_stat['aggregated'].get('mean_depth_attention', [])
                source_layer_ids = layer_stat['aggregated'].get('source_layer_ids', [])
                if depth_attn:
                    # Map local attention positions to true source layer ids (including embedding row).
                    if source_layer_ids and len(source_layer_ids) == len(depth_attn):
                        for local_idx, weight in enumerate(depth_attn):
                            src = source_layer_ids[local_idx]
                            row_idx = 0 if src == -1 else src + 1
                            if 0 <= row_idx < (num_layers + 1):
                                attention_matrix[row_idx, layer_idx] = weight
                    else:
                        # Backward-compatible fallback
                        for source_idx, weight in enumerate(depth_attn[:layer_idx + 1]):
                            attention_matrix[source_idx + 1, layer_idx] = weight
        
        # Create heatmap with exact cell rendering (no smoothing)
        cmap = plt.cm.get_cmap('YlOrRd').copy()
        cmap.set_bad(color='white')
        mask = np.isnan(attention_matrix)
        y_labels_full = ['Emb'] + [f'L{i}' for i in range(num_layers)]
        sns.heatmap(
            attention_matrix,
            ax=axes,
            cmap=cmap,
            mask=mask,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            linecolor=None,
            cbar=True,
            cbar_kws={'label': 'Mean Attention Weight'},
            square=True,
            xticklabels=list(range(num_layers)),
            yticklabels=y_labels_full,
        )
        axes.set_xlabel('Attending Layer', fontsize=12, fontweight='bold')
        axes.set_ylabel('Source (Earlier) State', fontsize=12, fontweight='bold')
        axes.set_title('Attention Residual: Which Sources Are Attended To (Embedding + Layers)', fontsize=14, fontweight='bold')
        axes.grid(False)
        axes.tick_params(axis='x', labelrotation=0)

        # Keep rendering clean: no extra grid overlays that visually duplicate cells.
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_depth_attention_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: layer_depth_attention_heatmap.png")
        plt.close()
    
    def plot_gate_activation_across_layers(self, stats: Dict, num_layers: int):
        """
        Plot gate activation values across layers.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            num_layers: Total number of layers
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        gate_means = []
        gate_stds = []
        layers = list(range(num_layers))
        
        gate_stats = stats.get('gate_stats', {})
        for layer_idx in layers:
            gate_stat = gate_stats.get(layer_idx, {})
            agg = gate_stat.get('aggregated', {})
            gate_means.append(agg.get('mean', 0.0))
            gate_stds.append(agg.get('std', 0.0))
        
        # Plot with error bars
        ax.errorbar(layers, gate_means, yerr=gate_stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gate Activation Value', fontsize=12, fontweight='bold')
        ax.set_title('Gate Activation Values Across Layers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gate_activation_across_layers.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: gate_activation_across_layers.png")
        plt.close()
    
    def plot_cross_dataset_comparison(self, stats: Dict, layers_to_plot: int = 8):
        """
        Compare attention patterns across different datasets.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            layers_to_plot: Number of layers to visualize (plot first N layers)
        """
        dataset_comparison = stats.get('dataset_comparison', {})
        if not dataset_comparison:
            print("No cross-dataset data available")
            return
        
        datasets = list(dataset_comparison.keys())
        num_datasets = len(datasets)
        
        fig, axes = plt.subplots(num_datasets, 1, figsize=(14, 5 * num_datasets))
        if num_datasets == 1:
            axes = [axes]
        
        for ax, dataset_id in zip(axes, datasets):
            dataset_stats = dataset_comparison[dataset_id]
            dataset_name = dataset_stats.get('dataset_name', dataset_id)
            
            # Extract top attended layers for each layer
            for layer_idx in range(min(layers_to_plot, 8)):
                layer_analysis = dataset_stats['layer_analysis'].get(layer_idx, {})
                ranking = layer_analysis.get('layer_ranking', [])
                
                if ranking:
                    top_layers = ranking[:3]
                    def _fmt_src(src):
                        return "Emb" if src == -1 else f"L{src}"
                    top_layers_str = ', '.join([f"{_fmt_src(idx)}({w:.3f})" for idx, w in top_layers])
                    ax.text(0.05, 0.95 - layer_idx * 0.1, f"Layer {layer_idx}: {top_layers_str}",
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'Dataset: {dataset_name}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_dataset_comparison.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: cross_dataset_comparison.png")
        plt.close()
    
    def plot_attention_flow_matrix(self, stats: Dict):
        """
        Plot the full attention flow matrix.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
        """
        attention_flow = stats.get('attention_flow', {})
        flow_matrix = np.array(attention_flow.get('attention_flow_matrix', []), dtype=float)
        
        if flow_matrix.size == 0:
            print("No attention flow data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        n_sources, n_attending = flow_matrix.shape
        
        # Hide impossible/empty cells for cleaner visual focus.
        flow_matrix_masked = flow_matrix.copy()
        flow_matrix_masked[flow_matrix_masked <= 0] = np.nan
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_bad(color='white')
        mask = np.isnan(flow_matrix_masked)
        source_labels = attention_flow.get('source_labels', [])
        y_labels_full = ['Emb'] + [f'L{i}' for i in range(n_sources - 1)]
        if source_labels and len(source_labels) == n_sources:
            y_labels_full = ['Emb'] + [f'L{i-1}' for i in range(1, n_sources)]
        sns.heatmap(
            flow_matrix_masked,
            ax=ax,
            cmap=cmap,
            mask=mask,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            linecolor=None,
            cbar=True,
            cbar_kws={'label': 'Mean Attention Weight'},
            square=True,
            xticklabels=list(range(n_attending)),
            yticklabels=y_labels_full,
        )
        ax.set_xlabel('Attending Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Source Layer', fontsize=12, fontweight='bold')
        ax.set_title('Attention Flow: Source → Attending Layers', fontsize=14, fontweight='bold')
        ax.grid(False)
        ax.tick_params(axis='x', labelrotation=0)

        # Keep rendering clean: no extra grid overlays that visually duplicate cells.
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_flow_matrix.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: attention_flow_matrix.png")
        plt.close()
    
    def plot_layer_ranking_comparison(self, stats: Dict, layers_to_plot: int = 12):
        """
        Plot which layers are most frequently attended to across the model.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            layers_to_plot: Number of layers to show
        """
        layer_stats = stats.get('layer_depth_stats', {})
        
        # Collect rankings for each layer
        layer_rankings = {}
        for layer_idx in range(min(layers_to_plot, 16)):
            if layer_idx in layer_stats:
                ranking = layer_stats[layer_idx]['aggregated'].get('layer_ranking', [])
                if ranking:
                    top_layer = ranking[0]
                    layer_rankings[layer_idx] = top_layer
        
        if not layer_rankings:
            print("No ranking data available")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        layers = sorted(layer_rankings.keys())
        top_attended = [layer_rankings[l][0] for l in layers]  # Source layer index
        attention_weights = [layer_rankings[l][1] for l in layers]  # Attention weight
        
        bars = ax.bar(layers, attention_weights, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Color code bars by the source layer (embedding is -1)
        valid_sources = [src for src in top_attended if src >= 0]
        max_source = max(valid_sources) if valid_sources else 0
        colors = plt.cm.Set3(np.linspace(0, 1, max_source + 1))
        for bar, source_layer in zip(bars, top_attended):
            if source_layer == -1:
                bar.set_color('gray')
                bar.set_label('Embedding')
            else:
                bar.set_color(colors[source_layer])
                bar.set_label(f'Layer {source_layer}')
        
        ax.set_xlabel('Attending Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title('Most Attended Layer Per Depth Position', fontsize=14, fontweight='bold')
        ax.set_xticks(layers)
        ax.legend(title='Source Layer', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_ranking_comparison.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: layer_ranking_comparison.png")
        plt.close()

    def plot_top_relation_map(self, stats: Dict, num_layers: int, top_k: int = None):
        """
        Plot a relation map from ranked sources.

        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            num_layers: Total number of layers
            top_k: Number of top sources per attending layer to keep.
                If None, keep all available relations for each attending layer.
        """
        # row 0 = Embedding, rows 1.. = Layer 0..N-1
        relation_matrix = np.full((num_layers + 1, num_layers), np.nan, dtype=float)

        layer_stats = stats.get('layer_depth_stats', {})
        for layer_idx in range(num_layers):
            layer_stat = layer_stats.get(layer_idx, {})
            agg = layer_stat.get('aggregated', {})
            ranking_all = agg.get('layer_ranking', [])
            ranking = ranking_all if top_k is None else ranking_all[:max(1, top_k)]
            for source_id, weight in ranking:
                row_idx = 0 if source_id == -1 else source_id + 1
                if 0 <= row_idx < (num_layers + 1):
                    relation_matrix[row_idx, layer_idx] = weight

        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.get_cmap('viridis').copy()
        cmap.set_bad(color='white')
        mask = np.isnan(relation_matrix)
        y_labels_full = ['Emb'] + [f'L{i}' for i in range(num_layers)]

        sns.heatmap(
            relation_matrix,
            ax=ax,
            cmap=cmap,
            mask=mask,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            cbar=True,
            cbar_kws={'label': 'Attention Weight (Selected Relations)'},
            square=True,
            xticklabels=list(range(num_layers)),
            yticklabels=y_labels_full,
        )
        ax.set_xlabel('Attending Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Selected Source', fontsize=12, fontweight='bold')
        title_suffix = 'All Valid Relations' if top_k is None else f'Top-{top_k} per Attending Layer'
        ax.set_title(f'Relation Map ({title_suffix})', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=0)
        ax.grid(False)

        plt.tight_layout()
        out_name = 'all_relation_map.png' if top_k is None else f'top{top_k}_relation_map.png'
        plt.savefig(self.output_dir / out_name, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {out_name}")
        plt.close()
    
    def plot_dataset_gate_distributions(self, stats: Dict):
        """
        Compare gate distributions across datasets.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
        """
        dataset_comparison = stats.get('dataset_comparison', {})
        if not dataset_comparison:
            print("No dataset comparison data")
            return
        
        fig, axes = plt.subplots(1, len(dataset_comparison), figsize=(6 * len(dataset_comparison), 5))
        if len(dataset_comparison) == 1:
            axes = [axes]
        
        for ax, (dataset_id, dataset_stats) in zip(axes, dataset_comparison.items()):
            dataset_name = dataset_stats.get('dataset_name', dataset_id)
            
            layers = sorted(dataset_stats['layer_analysis'].keys())
            mean_gates = []
            
            for layer_idx in layers:
                layer_info = dataset_stats['layer_analysis'].get(layer_idx, {})
                mean_gate = layer_info.get('mean_gate')
                if mean_gate is not None:
                    mean_gates.append(mean_gate)
            
            if mean_gates:
                ax.bar(range(len(mean_gates)), mean_gates, color='teal', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
                ax.set_ylabel('Mean Gate Value', fontsize=11, fontweight='bold')
                ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(mean_gates)))
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_gate_distributions.png', dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: dataset_gate_distributions.png")
        plt.close()
    
    def generate_all_plots(self, stats: Dict, num_layers: int):
        """
        Generate all available visualizations.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
            num_layers: Total number of layers in the model
        """
        print("\nGenerating visualizations...")
        print(f"Output directory: {self.output_dir}")
        
        self.plot_layer_depth_attention(stats, num_layers)
        self.plot_gate_activation_across_layers(stats, num_layers)
        self.plot_attention_flow_matrix(stats)
        self.plot_layer_ranking_comparison(stats)
        self.plot_top_relation_map(stats, num_layers, top_k=None)
        self.plot_cross_dataset_comparison(stats)
        self.plot_dataset_gate_distributions(stats)
        
        print("\nAll visualizations saved!")
