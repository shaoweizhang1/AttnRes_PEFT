"""
Analyzer for Attention Residual (AttnRes) adapter usage patterns.

Tracks which earlier layers are attended to and how attention patterns vary across:
- Different layers in the model
- Different tasks/datasets
- Different sequence positions
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from collections import defaultdict


class AttnResAnalyzer:
    """Analyzes attention residual adapter usage patterns across layers and tasks."""
    
    def __init__(self, num_layers: int, num_datasets: Optional[int] = None, lookback: Optional[int] = None):
        """
        Initialize the analyzer.
        
        Args:
            num_layers: Number of transformer layers in the model
            num_datasets: Optional number of datasets for pre-allocation
            lookback: Optional lookback window size (e.g., 8 means each adapter can attend to up to 8 earlier layers)
        """
        self.num_layers = num_layers
        self.num_datasets = num_datasets or 1
        self.lookback = lookback
        
        # Per-layer per-dataset statistics
        self.layer_source_attention = defaultdict(lambda: defaultdict(list))  # [layer][dataset] = list of [B, Lprev, T]
        self.layer_gate_values = defaultdict(lambda: defaultdict(list))  # [layer][dataset] = list of scalar values
        self.layer_adapter_output_norm = defaultdict(lambda: defaultdict(list))  # [layer][dataset] = list of norms
        
        # Per-dataset metadata
        self.dataset_names = {}  # {dataset_id} -> name
        self.dataset_sample_counts = defaultdict(int)  # {dataset_id} -> count
        
        # Aggregated statistics
        self.aggregated_stats = {}
    
    def record_forward_pass(
        self,
        layer_idx: int,
        alpha: torch.Tensor,  # [B, Lprev, T] attention weights over previous layers
        gate_value: torch.Tensor,  # scalar gate value
        adapter_output: torch.Tensor,  # [B, T, D] adapter output
        dataset_id: str = "default",
        batch_size: int = 1,
    ):
        """
        Record attention residual statistics from a forward pass.
        
        Args:
            layer_idx: Index of the transformer layer
            alpha: Attention weights from the depth attention mechanism [B, Lprev, T]
            gate_value: Gate parameter value (scalar or shape [1])
            adapter_output: Output of the adapter [B, T, D]
            dataset_id: Name/identifier of the dataset
            batch_size: Batch size for normalization
        """
        # Ensure tensors are on CPU and detached
        alpha = alpha.detach().cpu()
        gate_value = gate_value.detach().cpu()
        adapter_output = adapter_output.detach().cpu()
        
        # Compute norm of adapter output
        output_norm = torch.norm(adapter_output, p=2, dim=-1).mean().item()  # Average over batch and seq
        
        # Extract scalar gate value
        if gate_value.numel() > 1:
            gate_value = gate_value.mean()
        gate_value = gate_value.item()
        
        # Store statistics
        self.layer_source_attention[layer_idx][dataset_id].append(alpha)
        self.layer_gate_values[layer_idx][dataset_id].append(gate_value)
        self.layer_adapter_output_norm[layer_idx][dataset_id].append(output_norm)
        self.dataset_sample_counts[dataset_id] += batch_size
    
    def set_dataset_name(self, dataset_id: str, name: str):
        """Set a human-readable name for a dataset."""
        self.dataset_names[dataset_id] = name
    
    def compute_aggregated_stats(self) -> Dict:
        """
        Compute aggregated statistics across all recorded data.
        
        Returns:
            Dictionary containing:
            - layer_depth_usage: which layers each layer attends to
            - layer_gate_activations: how active each layer's gate is
            - layer_variance: variance of attention across sequences
            - cross_dataset_comparison: per-dataset statistics
        """
        self.aggregated_stats = {}
        
        # 1. Per-layer depth attention analysis
        layer_depth_stats = {}
        for layer_idx in range(self.num_layers):
            layer_depth_stats[layer_idx] = self._analyze_layer_depth_attention(layer_idx)
        self.aggregated_stats['layer_depth_stats'] = layer_depth_stats
        
        # 2. Gate activation analysis
        gate_stats = {}
        for layer_idx in range(self.num_layers):
            gate_stats[layer_idx] = self._analyze_layer_gates(layer_idx)
        self.aggregated_stats['gate_stats'] = gate_stats
        
        # 3. Cross-dataset comparison
        dataset_comparison = {}
        for dataset_id in self.dataset_sample_counts.keys():
            dataset_comparison[dataset_id] = self._analyze_dataset(dataset_id)
        self.aggregated_stats['dataset_comparison'] = dataset_comparison
        
        # 4. Layer-to-layer attention flow
        self.aggregated_stats['attention_flow'] = self._compute_attention_flow()
        
        return self.aggregated_stats
    
    def _analyze_layer_depth_attention(self, layer_idx: int) -> Dict:
        """Analyze which earlier layers are attended to by a given layer."""
        stats = {
            'layer_idx': layer_idx,
            'per_dataset': {},
            'aggregated': {},
        }
        
        for dataset_id in self.dataset_sample_counts.keys():
            alphas = self.layer_source_attention[layer_idx].get(dataset_id, [])
            if not alphas:
                continue
            
            # Concatenate all batches: [B*Nbatches, Lprev, T]
            alpha_concat = torch.cat(alphas, dim=0)  # [B_total, Lprev, T]
            
            # Average over batch and sequence
            depth_attention = alpha_concat.mean(dim=(0, 2))  # [Lprev]
            source_layer_ids = self._get_source_layer_ids(layer_idx, depth_attention.shape[0])
            
            # Also get per-position statistics
            depth_per_pos = alpha_concat.mean(dim=0)  # [Lprev, T] - averaged over batch
            
            stats['per_dataset'][dataset_id] = {
                'mean_depth_attention': depth_attention.numpy().tolist(),
                'source_layer_ids': source_layer_ids,
                'depth_attention_std': alpha_concat.std(dim=0).mean(dim=-1).numpy().tolist(),
                'layer_ranking': self._get_layer_ranking(depth_attention, layer_idx=layer_idx),
                'num_samples': alpha_concat.shape[0],
            }
        
        # Aggregate across all datasets
        if stats['per_dataset']:
            all_alphas = []
            for dataset_id, dataset_alphas in self.layer_source_attention[layer_idx].items():
                all_alphas.extend(dataset_alphas)
            if all_alphas:
                alpha_all = torch.cat(all_alphas, dim=0)
                depth_attention_all = alpha_all.mean(dim=(0, 2))
                source_layer_ids_all = self._get_source_layer_ids(layer_idx, depth_attention_all.shape[0])
                stats['aggregated'] = {
                    'mean_depth_attention': depth_attention_all.numpy().tolist(),
                    'source_layer_ids': source_layer_ids_all,
                    'depth_attention_std': alpha_all.std(dim=0).mean(dim=-1).numpy().tolist(),
                    'layer_ranking': self._get_layer_ranking(depth_attention_all, layer_idx=layer_idx),
                    'num_total_samples': alpha_all.shape[0],
                }
        
        return stats
    
    def _analyze_layer_gates(self, layer_idx: int) -> Dict:
        """Analyze gate activation for a layer across datasets."""
        stats = {
            'layer_idx': layer_idx,
            'per_dataset': {},
        }
        
        for dataset_id in self.dataset_sample_counts.keys():
            gate_values = self.layer_gate_values[layer_idx].get(dataset_id, [])
            if not gate_values:
                continue
            
            gate_array = np.array(gate_values)
            stats['per_dataset'][dataset_id] = {
                'mean': float(gate_array.mean()),
                'std': float(gate_array.std()),
                'min': float(gate_array.min()),
                'max': float(gate_array.max()),
                'num_samples': len(gate_values),
            }
        
        # Aggregate across datasets
        all_gates = []
        for dataset_gates in self.layer_gate_values[layer_idx].values():
            all_gates.extend(dataset_gates)
        if all_gates:
            gate_array = np.array(all_gates)
            stats['aggregated'] = {
                'mean': float(gate_array.mean()),
                'std': float(gate_array.std()),
                'min': float(gate_array.min()),
                'max': float(gate_array.max()),
                'num_total_samples': len(all_gates),
            }
        
        return stats
    
    def _analyze_dataset(self, dataset_id: str) -> Dict:
        """Analyze adapter usage patterns for a specific dataset."""
        stats = {
            'dataset_id': dataset_id,
            'dataset_name': self.dataset_names.get(dataset_id, dataset_id),
            'num_samples': self.dataset_sample_counts[dataset_id],
            'layer_analysis': {},
        }
        
        for layer_idx in range(self.num_layers):
            layer_alphas = self.layer_source_attention[layer_idx].get(dataset_id, [])
            layer_gates = self.layer_gate_values[layer_idx].get(dataset_id, [])
            layer_norms = self.layer_adapter_output_norm[layer_idx].get(dataset_id, [])
            
            if layer_alphas:
                alpha_concat = torch.cat(layer_alphas, dim=0)
                depth_attention = alpha_concat.mean(dim=(0, 2))
                source_layer_ids = self._get_source_layer_ids(layer_idx, depth_attention.shape[0])
                
                stats['layer_analysis'][layer_idx] = {
                    'mean_depth_attention': depth_attention.numpy().tolist(),
                    'source_layer_ids': source_layer_ids,
                    'mean_gate': np.mean(layer_gates) if layer_gates else None,
                    'mean_output_norm': np.mean(layer_norms) if layer_norms else None,
                    'layer_ranking': self._get_layer_ranking(depth_attention, layer_idx=layer_idx),
                }
        
        return stats
    
    def _compute_attention_flow(self) -> Dict:
        """Compute aggregated attention flow across layers."""
        # Include one extra source row for embedding (-1) at row 0.
        # Rows 1..num_layers correspond to layer outputs 0..num_layers-1.
        flow_matrix = np.zeros((self.num_layers + 1, self.num_layers))
        
        for layer_idx in range(self.num_layers):
            alphas_all = []
            for dataset_alphas in self.layer_source_attention[layer_idx].values():
                alphas_all.extend(dataset_alphas)
            
            if alphas_all:
                alpha_concat = torch.cat(alphas_all, dim=0)
                depth_attention = alpha_concat.mean(dim=(0, 2))  # [Lprev]
                source_layer_ids = self._get_source_layer_ids(layer_idx, depth_attention.shape[0])

                # Map source ids to matrix rows, including embedding row.
                for local_idx, weight in enumerate(depth_attention):
                    source_layer_idx = source_layer_ids[local_idx]
                    row_idx = 0 if source_layer_idx == -1 else source_layer_idx + 1
                    if 0 <= row_idx < (self.num_layers + 1):
                        flow_matrix[row_idx, layer_idx] = weight.item()

        source_labels = ['Embedding'] + [f'Layer {i}' for i in range(self.num_layers)]
        
        return {
            'attention_flow_matrix': flow_matrix.tolist(),
            'source_labels': source_labels,
            'description': 'flow_matrix[i,j] is the avg attention weight when attending layer j uses source i (row 0 is embedding, rows 1.. are layer outputs 0..)'
        }
    
    def _get_source_layer_ids(self, layer_idx: int, num_sources: int) -> List[int]:
        """
        Map local source positions in alpha (0..num_sources-1) to logical source ids.

        Source id convention:
        - -1 => embedding/post-positioning state
        - k >= 0 => output of transformer layer k
        """
        # State indices include embedding at 0, then layer outputs as 1..layer_idx.
        start_state_idx = max(0, (layer_idx + 1) - num_sources)
        source_state_indices = list(range(start_state_idx, start_state_idx + num_sources))

        source_layer_ids = []
        for state_idx in source_state_indices:
            if state_idx == 0:
                source_layer_ids.append(-1)
            else:
                source_layer_ids.append(state_idx - 1)
        return source_layer_ids

    def _get_layer_ranking(self, attention_weights: torch.Tensor, layer_idx: int = None) -> List[Tuple[int, float]]:
        """
        Return a ranking of source ids by attention weight.

        Ranking uses logical source ids:
        - -1 => embedding/post-positioning state
        - k >= 0 => output of transformer layer k
        """
        weights_np = attention_weights.numpy()
        if layer_idx is not None:
            source_ids = self._get_source_layer_ids(layer_idx, len(weights_np))
            items = [(source_ids[local_idx], weight) for local_idx, weight in enumerate(weights_np)]
        else:
            # Fallback: if layer context is missing, keep positional ids.
            items = list(enumerate(weights_np))

        ranking = sorted(items, key=lambda x: x[1], reverse=True)
        return [(int(idx), float(weight)) for idx, weight in ranking]
    
    def get_most_attended_layers(self, layer_idx: int, dataset_id: str = None, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Get the top-k most attended layers for a given layer.
        
        Respects the lookback constraint: if lookback=8 is set, only layers within
        the valid lookback window are considered.
        
        Args:
            layer_idx: Index of the layer to analyze
            dataset_id: Specific dataset to analyze (if None, uses aggregate)
            top_k: Number of top layers to return
        
        Returns:
            List of (layer_index, attention_weight) tuples
        """
        alphas = self.layer_source_attention[layer_idx].get(dataset_id or 'all', [])
        if not alphas and dataset_id is None:
            # Aggregate across all datasets
            alphas = []
            for dataset_alphas in self.layer_source_attention[layer_idx].values():
                alphas.extend(dataset_alphas)
        
        if not alphas:
            return []
        
        alpha_concat = torch.cat(alphas, dim=0)
        depth_attention = alpha_concat.mean(dim=(0, 2))  # [Lprev]
        ranking = self._get_layer_ranking(depth_attention, layer_idx=layer_idx)
        return ranking[:top_k]
    
    def save_analysis(self, output_path: str):
        """Save analysis results to JSON file."""
        if not self.aggregated_stats:
            self.compute_aggregated_stats()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        stats_json = self._make_json_serializable(self.aggregated_stats)
        
        with open(output_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"Analysis saved to {output_path}")
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        if not self.aggregated_stats:
            self.compute_aggregated_stats()
        
        print("\n" + "="*80)
        print("ATTENTION RESIDUAL ADAPTER ANALYSIS SUMMARY")
        print("="*80)
        
        # Dataset overview
        print("\nDataset Overview:")
        print("-" * 80)
        for dataset_id, count in self.dataset_sample_counts.items():
            dataset_name = self.dataset_names.get(dataset_id, dataset_id)
            print(f"  {dataset_name}: {count} samples")
        
        # Per-layer depth attention
        print("\nPer-Layer Depth Attention (which earlier layers are attended to):")
        print("-" * 80)
        for layer_idx, layer_stats in self.aggregated_stats['layer_depth_stats'].items():
            if layer_stats['aggregated']:
                ranking = layer_stats['aggregated']['layer_ranking'][:3]
                pretty_ranking = [
                    ("Embedding" if src == -1 else f"Layer {src}", weight)
                    for src, weight in ranking
                ]
                print(f"  Layer {layer_idx}:")
                print(f"    Most attended sources: {pretty_ranking}")
                depth_attn = layer_stats['aggregated']['mean_depth_attention']
                print(f"    Attention distribution: {[f'{w:.3f}' for w in depth_attn[:5]]}...")
        
        # Gate activation
        print("\nGate Activation Statistics:")
        print("-" * 80)
        for layer_idx in range(min(5, self.num_layers)):  # Show first 5 layers
            gate_stat = self.aggregated_stats['gate_stats'][layer_idx]
            if gate_stat['aggregated']:
                agg = gate_stat['aggregated']
                print(f"  Layer {layer_idx}: mean={agg['mean']:.6f}, std={agg['std']:.6f}, "
                      f"range=[{agg['min']:.6f}, {agg['max']:.6f}]")
        
        # Cross-dataset comparison
        print("\nCross-Dataset Comparison:")
        print("-" * 80)
        for dataset_id, dataset_stats in self.aggregated_stats['dataset_comparison'].items():
            dataset_name = dataset_stats['dataset_name']
            print(f"  {dataset_name}:")
            for layer_idx in range(min(3, self.num_layers)):
                if layer_idx in dataset_stats['layer_analysis']:
                    layer_info = dataset_stats['layer_analysis'][layer_idx]
                    ranking = layer_info['layer_ranking'][:2] if layer_info['layer_ranking'] else []
                    pretty_ranking = [
                        ("Embedding" if src == -1 else f"Layer {src}", weight)
                        for src, weight in ranking
                    ]
                    print(f"    Layer {layer_idx} most attended: {pretty_ranking}")
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


class AttnResHook:
    """Context manager for automatically capturing AttnRes statistics during forward passes."""
    
    def __init__(self, model, analyzer: AttnResAnalyzer, dataset_id: str = "default"):
        """
        Initialize hook for capturing adapter outputs.
        
        Args:
            model: The wrapped model with adapters (Qwen3ForCausalLMWithAttnRes)
            analyzer: AttnResAnalyzer instance to record data
            dataset_id: Identifier for the dataset being analyzed
        """
        self.model = model
        self.analyzer = analyzer
        self.dataset_id = dataset_id
        self.hooks = []
    
    def __enter__(self):
        """Register hooks on all adapter forward passes."""
        for layer_idx, adapter in enumerate(self.model.adapters):
            hook_fn = self._make_adapter_hook(layer_idx)
            handle = adapter.register_forward_hook(hook_fn)
            self.hooks.append(handle)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
    
    def _make_adapter_hook(self, layer_idx: int):
        """Create a forward hook for an adapter."""
        def hook(module, input, output):
            # output from DepthAttentionAdapter.forward is (out, alpha)
            if isinstance(output, tuple) and len(output) == 2:
                out, alpha = output
                # Get gate value
                gate_value = module.gate
                # Record statistics
                self.analyzer.record_forward_pass(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    gate_value=gate_value,
                    adapter_output=out,
                    dataset_id=self.dataset_id,
                    batch_size=input[0].shape[0],  # input[0] is h_base
                )
        return hook


def analyze_attnres_on_dataset(
    model,
    dataloader,
    dataset_id: str = "default",
    num_layers: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> AttnResAnalyzer:
    """
    Run inference on a dataset and collect adapter usage statistics.
    
    Args:
        model: Wrapped model with adapters
        dataloader: DataLoader yielding batches with 'input_ids' and optionally 'attention_mask'
        dataset_id: Name for this dataset
        num_layers: Number of layers (if None, inferred from model)
        device: Device to run on
    
    Returns:
        AttnResAnalyzer with recorded statistics
    """
    if num_layers is None:
        num_layers = len(model.model.layers)
    
    analyzer = AttnResAnalyzer(num_layers=num_layers)
    analyzer.set_dataset_name(dataset_id, dataset_id)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad(), AttnResHook(model, analyzer, dataset_id=dataset_id):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass (hooks capture statistics)
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_depth_weights=False,
            )
    
    return analyzer
