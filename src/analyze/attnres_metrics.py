"""
Advanced metrics and analysis for Attention Residual adapters.

Includes:
- Cross-dataset metrics (task similarity, divergence)
- Layer importance ranking
- Temporal attention evolution tracking
- Statistical significance tests
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.preprocessing import normalize


class AttnResMetrics:
    """Advanced metrics for attention residual adapter analysis."""
    
    @staticmethod
    def layer_importance_score(attention_weights: np.ndarray) -> float:
        """
        Compute importance score for a layer based on attention entropy.
        
        Lower entropy (more focused) → higher importance
        
        Args:
            attention_weights: [Lprev] attention weights that sum to 1
        
        Returns:
            Importance score in [0, 1], where 1 = most focused attention
        """
        # Remove zeros to avoid log(0)
        weights = attention_weights[attention_weights > 1e-8]
        if len(weights) == 0:
            return 0.0
        
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(attention_weights))
        
        # Normalize entropy to [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Convert to importance (1 - entropy)
        importance = 1.0 - normalized_entropy
        return float(importance)
    
    @staticmethod
    def compute_all_layer_importances(layer_stats: Dict) -> Dict[int, float]:
        """
        Compute importance scores for all layers.
        
        Args:
            layer_stats: Output from AttnResAnalyzer.aggregated_stats['layer_depth_stats']
        
        Returns:
            Dict mapping layer_idx to importance score
        """
        importances = {}
        for layer_idx, stat in layer_stats.items():
            if stat.get('aggregated'):
                weights = np.array(stat['aggregated'].get('mean_depth_attention', []))
                if len(weights) > 0:
                    importance = AttnResMetrics.layer_importance_score(weights)
                    importances[layer_idx] = importance
        return importances
    
    @staticmethod
    def jensen_shannon_divergence(dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        
        Returns:
            Distance in [0, 1], where 0 = identical, 1 = maximally different
        """
        # Ensure distributions are valid probabilities
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)
        
        # Pad to same length
        max_len = max(len(dist1), len(dist2))
        dist1_padded = np.zeros(max_len)
        dist2_padded = np.zeros(max_len)
        dist1_padded[:len(dist1)] = dist1
        dist2_padded[:len(dist2)] = dist2
        
        # Normalize to probabilities
        dist1_padded = dist1_padded / (np.sum(dist1_padded) + 1e-10)
        dist2_padded = dist2_padded / (np.sum(dist2_padded) + 1e-10)
        
        return float(jensenshannon(dist1_padded, dist2_padded))
    
    @staticmethod
    def compute_task_divergence(stats: Dict) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise divergence between datasets' attention patterns.
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
        
        Returns:
            Dict mapping (dataset_a, dataset_b) to divergence score
        """
        dataset_comparison = stats.get('dataset_comparison', {})
        datasets = list(dataset_comparison.keys())
        
        divergences = {}
        
        for i, dataset_a in enumerate(datasets):
            for dataset_b in datasets[i+1:]:
                divergence = 0.0
                count = 0
                
                # Compare layer-by-layer attention patterns
                for layer_idx in range(12):  # Compare first 12 layers
                    layer_a_data = dataset_comparison[dataset_a]['layer_analysis'].get(layer_idx, {})
                    layer_b_data = dataset_comparison[dataset_b]['layer_analysis'].get(layer_idx, {})
                    
                    attn_a = layer_a_data.get('mean_depth_attention', [])
                    attn_b = layer_b_data.get('mean_depth_attention', [])
                    
                    if attn_a and attn_b:
                        div = AttnResMetrics.jensen_shannon_divergence(attn_a, attn_b)
                        divergence += div
                        count += 1
                
                if count > 0:
                    avg_divergence = divergence / count
                    divergences[(dataset_a, dataset_b)] = avg_divergence
        
        return divergences
    
    @staticmethod
    def identify_specialized_layers(layer_stats: Dict, dataset_comparison: Dict) -> Dict[int, List]:
        """
        Identify layers that show task-specific attention patterns.
        
        A layer is "specialized" if different datasets have significantly different
        attention distributions for that layer.
        
        Args:
            layer_stats: Per-layer depth statistics
            dataset_comparison: Per-dataset layer analysis
        
        Returns:
            Dict mapping layer_idx to list of (dataset_pair, divergence) tuples
        """
        specialized = {}
        datasets = list(dataset_comparison.keys())
        
        for layer_idx in layer_stats.keys():
            divergences = []
            
            for i, dataset_a in enumerate(datasets):
                for dataset_b in datasets[i+1:]:
                    layer_a_data = dataset_comparison[dataset_a]['layer_analysis'].get(layer_idx, {})
                    layer_b_data = dataset_comparison[dataset_b]['layer_analysis'].get(layer_idx, {})
                    
                    attn_a = layer_a_data.get('mean_depth_attention', [])
                    attn_b = layer_b_data.get('mean_depth_attention', [])
                    
                    if attn_a and attn_b:
                        div = AttnResMetrics.jensen_shannon_divergence(attn_a, attn_b)
                        if div > 0.3:  # Threshold for "specialized"
                            divergences.append(((dataset_a, dataset_b), div))
            
            if divergences:
                specialized[layer_idx] = sorted(divergences, key=lambda x: x[1], reverse=True)
        
        return specialized
    
    @staticmethod
    def compute_gate_learning_curve(analyzer, layer_idx: int, dataset_id: str) -> np.ndarray:
        """
        Extract gate value progression during analysis (if recorded chronologically).
        
        Args:
            analyzer: AttnResAnalyzer instance
            layer_idx: Layer to analyze
            dataset_id: Dataset identifier
        
        Returns:
            Array of gate values in chronological order
        """
        gate_values = analyzer.layer_gate_values[layer_idx].get(dataset_id, [])
        return np.array(gate_values)
    
    @staticmethod
    def statistical_significance_test(
        dist1: List[float],
        dist2: List[float],
        test_type: str = "ttest"
    ) -> Tuple[float, float]:
        """
        Perform statistical significance test between two distributions.
        
        Args:
            dist1: First distribution (e.g., gate values)
            dist2: Second distribution
            test_type: 'ttest' (t-test) or 'mannwhitneyu' (Mann-Whitney U test)
        
        Returns:
            (test_statistic, p_value)
        """
        if test_type == "ttest":
            stat, pval = scipy_stats.ttest_ind(dist1, dist2)
        elif test_type == "mannwhitneyu":
            stat, pval = scipy_stats.mannwhitneyu(dist1, dist2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return float(stat), float(pval)
    
    @staticmethod
    def rank_layer_contribution(stats: Dict) -> List[Tuple[int, float]]:
        """
        Rank layers by their overall contribution (importance).
        
        Combines:
        - Attention focus (importance score)
        - Gate activation magnitude
        - Adapter output norm
        
        Args:
            stats: Output from AttnResAnalyzer.aggregated_stats
        
        Returns:
            Sorted list of (layer_idx, contribution_score)
        """
        contributions = {}
        
        layer_stats = stats.get('layer_depth_stats', {})
        gate_stats = stats.get('gate_stats', {})
        
        for layer_idx in layer_stats.keys():
            # Importance from attention focus
            layer_stat = layer_stats[layer_idx]
            if layer_stat.get('aggregated'):
                weights = layer_stat['aggregated'].get('mean_depth_attention', [])
                importance = AttnResMetrics.layer_importance_score(np.array(weights))
            else:
                importance = 0.0
            
            # Gate activation
            gate_stat = gate_stats.get(layer_idx, {})
            gate_value = abs(gate_stat.get('aggregated', {}).get('mean', 0))
            
            # Combined score (equally weighted)
            contribution = (importance + gate_value) / 2
            contributions[layer_idx] = contribution
        
        # Sort by contribution
        ranking = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return ranking
    
    @staticmethod
    def compute_attention_concentration(attention_weights: np.ndarray) -> float:
        """
        Compute how concentrated attention is on top layers (higher = more concentrated).
        
        Uses Gini coefficient for concentration measurement.
        
        Args:
            attention_weights: [Lprev] attention weights
        
        Returns:
            Gini coefficient in [0, 1]
        """
        weights = np.array(attention_weights)
        weights = weights / (np.sum(weights) + 1e-10)  # Normalize
        
        n = len(weights)
        sorted_weights = np.sort(weights)
        
        # Gini coefficient
        gini = (2 * np.sum(np.arange(1, n+1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        
        return float(gini)
    
    @staticmethod
    def identify_attention_hubs(layer_stats: Dict, threshold: float = 0.25) -> Dict[int, List]:
        """
        Identify "hub" layers that are attended to by many other layers.
        
        Args:
            layer_stats: Per-layer statistics
            threshold: Attention weight threshold for considering a layer as attended
        
        Returns:
            Dict mapping layer_idx to list of layers attending to it
        """
        hub_analysis = {}
        
        # Build reverse attention map
        for attending_layer_idx, layer_stat in layer_stats.items():
            if layer_stat.get('aggregated'):
                ranking = layer_stat['aggregated'].get('layer_ranking', [])
                for source_layer_idx, weight in ranking:
                    if weight >= threshold:
                        if source_layer_idx not in hub_analysis:
                            hub_analysis[source_layer_idx] = []
                        hub_analysis[source_layer_idx].append(attending_layer_idx)
        
        # Sort by number of attendees
        return {k: sorted(v) for k, v in sorted(
            hub_analysis.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )}


class AttnResReport:
    """Generate detailed analysis reports."""
    
    @staticmethod
    def generate_metrics_report(stats: Dict, analyzer, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive metrics report.
        
        Args:
            stats: Analysis statistics from analyzer
            analyzer: AttnResAnalyzer instance
            output_path: Optional path to save report
        
        Returns:
            Report text
        """
        report = []
        report.append("="*80)
        report.append("ATTENTION RESIDUAL ADAPTER - ADVANCED METRICS REPORT")
        report.append("="*80)
        
        # Section 1: Layer Importance
        report.append("\n1. LAYER IMPORTANCE RANKING")
        report.append("-"*80)
        layer_stats = stats.get('layer_depth_stats', {})
        importances = AttnResMetrics.compute_all_layer_importances(layer_stats)
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        for layer_idx, importance in sorted_importances[:10]:
            report.append(f"  Layer {layer_idx}: {importance:.4f} importance")
        
        # Section 2: Task Divergence
        report.append("\n2. CROSS-TASK DIVERGENCE")
        report.append("-"*80)
        divergences = AttnResMetrics.compute_task_divergence(stats)
        
        for (dataset_a, dataset_b), div in sorted(divergences.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {dataset_a} ↔ {dataset_b}: {div:.4f}")
        
        # Section 3: Specialized Layers
        report.append("\n3. TASK-SPECIALIZED LAYERS")
        report.append("-"*80)
        dataset_comparison = stats.get('dataset_comparison', {})
        specialized = AttnResMetrics.identify_specialized_layers(layer_stats, dataset_comparison)
        
        for layer_idx, divergence_list in sorted(specialized.items()):
            report.append(f"\n  Layer {layer_idx}:")
            for (d1, d2), div in divergence_list[:3]:
                report.append(f"    {d1} vs {d2}: divergence {div:.4f}")
        
        # Section 4: Attention Hubs
        report.append("\n4. ATTENTION HUB ANALYSIS")
        report.append("-"*80)
        hubs = AttnResMetrics.identify_attention_hubs(layer_stats, threshold=0.20)
        
        for source_layer, attendees in sorted(hubs.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            report.append(f"  Layer {source_layer}: attended by {len(attendees)} layers: {attendees}")
        
        # Section 5: Layer Concentration
        report.append("\n5. ATTENTION CONCENTRATION")
        report.append("-"*80)
        
        concentrations = {}
        for layer_idx, layer_stat in layer_stats.items():
            if layer_stat.get('aggregated'):
                weights = layer_stat['aggregated'].get('mean_depth_attention', [])
                conc = AttnResMetrics.compute_attention_concentration(np.array(weights))
                concentrations[layer_idx] = conc
        
        for layer_idx in sorted(concentrations.keys())[:10]:
            conc = concentrations[layer_idx]
            report.append(f"  Layer {layer_idx}: {conc:.4f} concentration (0=spread, 1=focused)")
        
        # Section 6: Overall Layer Contribution
        report.append("\n6. LAYER CONTRIBUTION RANKING")
        report.append("-"*80)
        contributions = AttnResMetrics.rank_layer_contribution(stats)
        
        for layer_idx, score in contributions[:8]:
            report.append(f"  Layer {layer_idx}: {score:.4f}")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
