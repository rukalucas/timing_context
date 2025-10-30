"""Analysis utilities for neural network hidden state analysis and psychometrics."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union
from pathlib import Path


def perform_pca(hidden_states, n_components=3):
    """
    Perform PCA on hidden states.

    Args:
        hidden_states: torch.Tensor of shape (batch, hidden_size, time) or (batch*time, hidden_size)
        n_components: Number of principal components to compute

    Returns:
        pca_data: np.ndarray of shape (batch, time, n_components) or (batch*time, n_components)
        explained_variance_ratio: np.ndarray of shape (n_components,)
        pcs: np.ndarray of shape (hidden_size, n_components) - principal component vectors
        mean: np.ndarray of shape (hidden_size,) - mean used for centering
    """
    # Handle different input shapes
    if hidden_states.dim() == 3:
        # (batch, hidden_size, time) -> (batch*time, hidden_size)
        B, H, T = hidden_states.shape
        hidden_flat = hidden_states.permute(0, 2, 1).reshape(-1, H)
        reshape_needed = True
    else:
        # Already flat
        hidden_flat = hidden_states
        reshape_needed = False
        H = hidden_flat.shape[1]

    # Convert to numpy if needed
    if isinstance(hidden_flat, torch.Tensor):
        hidden_flat = hidden_flat.cpu().numpy()

    # Center the data
    mean = hidden_flat.mean(axis=0)
    hidden_centered = hidden_flat - mean

    # Compute covariance matrix
    cov = (hidden_centered.T @ hidden_centered) / (hidden_centered.shape[0] - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project onto principal components
    pcs = eigenvectors[:, :n_components]
    pca_data = hidden_centered @ pcs

    # Compute explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    # Reshape back if needed
    if reshape_needed:
        pca_data = pca_data.reshape(B, T, n_components)

    return pca_data, explained_variance_ratio, pcs, mean


def perform_period_pca(hidden_states, batch, period_names, task, n_components=3):
    """
    Perform PCA on specific periods of trials (e.g., ITI, rule_report, decision).

    Args:
        hidden_states: torch.Tensor of shape (num_trials, hidden_size, time)
        batch: dict containing trial metadata including 'trial_length'
        period_names: str or list of str - period names to analyze
        task: Task object with dt attribute
        n_components: Number of principal components

    Returns:
        dict with keys as period names, values are dicts containing:
            - pca_data: np.ndarray of shape (num_trials, max_period_length, n_components)
            - explained_variance_ratio: np.ndarray of shape (n_components,)
            - period_lengths: list of int - actual length of each trial's period
            - period_info: list of dicts with 'start' and 'end' indices for each trial
            - pcs: np.ndarray of shape (hidden_size, n_components) - principal component vectors
            - hidden_flat: np.ndarray - flattened hidden states for this period
    """
    # Ensure period_names is a list
    if isinstance(period_names, str):
        period_names = [period_names]

    # Convert hidden_states to numpy
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu()

    num_trials = hidden_states.shape[0]
    H = hidden_states.shape[1]

    # Extract period boundaries from batch
    # This assumes the task has methods or the batch has information about periods
    period_info_all = _extract_period_boundaries(batch, task, num_trials)

    results = {}

    for period_name in period_names:
        # Get period info for this period
        period_info = period_info_all[period_name]

        # Determine max period length
        period_lengths = [info['end'] - info['start'] for info in period_info]
        max_period_length = max(period_lengths)

        # Extract and align hidden states for this period
        aligned_hidden = np.zeros((num_trials, max_period_length, H))

        for i, info in enumerate(period_info):
            start, end = info['start'], info['end']
            period_len = end - start

            # Extract hidden states for this period: (H, T_period)
            period_hidden = hidden_states[i, :, start:end].numpy()

            # Store in aligned array: (trial, time, hidden)
            aligned_hidden[i, :period_len, :] = period_hidden.T

        # Flatten for PCA (only use actual data, not padding)
        hidden_flat_list = []
        for i in range(num_trials):
            period_len = period_lengths[i]
            hidden_flat_list.append(aligned_hidden[i, :period_len, :])
        hidden_flat = np.vstack(hidden_flat_list)

        # Perform PCA
        mean = hidden_flat.mean(axis=0)
        hidden_centered = hidden_flat - mean
        cov = (hidden_centered.T @ hidden_centered) / (hidden_centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project all data (including padding)
        pcs = eigenvectors[:, :n_components]
        aligned_hidden_flat = aligned_hidden.reshape(-1, H)
        aligned_hidden_centered = aligned_hidden_flat - mean
        aligned_hidden_pca = aligned_hidden_centered @ pcs
        aligned_hidden_pca = aligned_hidden_pca.reshape(num_trials, max_period_length, n_components)

        explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

        results[period_name] = {
            'pca_data': aligned_hidden_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'period_lengths': period_lengths,
            'period_info': period_info,
            'pcs': pcs,
            'hidden_flat': hidden_flat  # Store for cross-period variance computation
        }

    return results


def plot_cross_period_variance(period_pca_results, figsize=(8, 7), save_path=None):
    """
    Plot heatmap showing how much variance in one period is explained by another period's PCs.

    Args:
        period_pca_results: dict from perform_period_pca containing results for all periods
        figsize: tuple, figure size
        save_path: str or Path, optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    period_names = list(period_pca_results.keys())
    n_periods = len(period_names)

    # Detect number of components from the first period
    first_period = period_names[0]
    n_components = period_pca_results[first_period]['pcs'].shape[1]

    # Create variance explained matrix
    variance_matrix = np.zeros((n_periods, n_periods))

    for i, fit_period in enumerate(period_names):
        fit_pcs = period_pca_results[fit_period]['pcs']  # (H, n_components)

        for j, test_period in enumerate(period_names):
            test_data = period_pca_results[test_period]['hidden_flat']  # (N, H)

            # Center test data
            test_centered = test_data - test_data.mean(axis=0)

            # Project onto fit period's PCs
            test_projected = test_centered @ fit_pcs  # (N, n_components)

            # Compute variance explained
            projected_variance = np.sum(np.var(test_projected, axis=0))
            total_variance = np.sum(np.var(test_centered, axis=0))
            variance_explained = projected_variance / total_variance

            variance_matrix[i, j] = variance_explained * 100  # Convert to percentage

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(variance_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_periods))
    ax.set_yticks(np.arange(n_periods))
    ax.set_xticklabels(period_names)
    ax.set_yticklabels(period_names)

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add value annotations
    for i in range(n_periods):
        for j in range(n_periods):
            text = ax.text(j, i, f'{variance_matrix[i, j]:.1f}',
                          ha='center', va='center', color='white' if variance_matrix[i, j] < 50 else 'black',
                          fontsize=11)

    # Labels and title
    ax.set_xlabel('Test', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fit', fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Period Variance Explained by Top {n_components} PCs (%)', fontsize=13, pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Variance Explained (%)', fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cross-period variance plot to {save_path}")

    return fig, ax


def _extract_period_boundaries(batch, task, num_trials):
    """
    Extract period boundaries for both sequence and non-sequence tasks.

    For sequence tasks: Returns 'iti', 'rule_report', 'timing', 'decision'
    For non-sequence tasks: Returns 'rule_report', 'timing', 'decision' (no ITI)

    Returns:
        dict with period names as keys, values are lists of dicts with 'start'/'end'
    """
    # Detect task type
    is_sequence_task = hasattr(task, 'inter_trial_interval')

    # Initialize period info based on task type
    if is_sequence_task:
        period_info = {
            'iti': [],
            'rule_report': [],
            'timing': [],
            'decision': []
        }
        iti_steps = int(task.inter_trial_interval / task.dt)
    else:
        period_info = {
            'rule_report': [],
            'timing': [],
            'decision': []
        }

    # Get task parameters (common to both task types)
    pulse_width_steps = int(task.pulse_width / task.dt)
    response_duration = int(task.response_period / task.dt)
    rule_delay_max = int(task.fixation_delay_range[1] / task.dt)
    rule_report_duration = int(task.rule_report_period / task.dt)
    timing_delay_max = int(task.fixation_delay_range[1] / task.dt)

    # For each trial, compute period boundaries
    for i in range(num_trials):
        # Extract trial length (handling both 1D and 2D arrays)
        trial_length = batch['trial_length'][0][i] if batch['trial_length'].ndim > 1 else batch['trial_length'][i]

        # Get measured time for this trial
        t_m = batch['t_m'][0][i] if batch['t_m'].ndim > 1 else batch['t_m'][i]
        t_m_steps = int(t_m / task.dt)

        if is_sequence_task:
            # Sequence task: Has ITI at start
            iti_start = 0
            iti_end = iti_steps

            # Rule report period: after ITI
            rule_report_start = iti_end
            rule_report_end = rule_report_start + rule_delay_max + rule_report_duration
            rule_report_end = min(trial_length, rule_report_end)

            period_info['iti'].append({'start': iti_start, 'end': iti_end})
        else:
            # Non-sequence task: No ITI, rule report starts from beginning
            rule_report_start = 0
            rule_report_end = rule_delay_max + rule_report_duration
            rule_report_end = min(trial_length, rule_report_end)

        # Timing period: from first pulse to end of second pulse (same for both task types)
        timing_start = rule_report_end + timing_delay_max
        timing_duration = t_m_steps + pulse_width_steps
        timing_end = timing_start + timing_duration
        timing_end = min(trial_length, timing_end)

        # Decision period: last part of trial (same for both task types)
        decision_start = trial_length - response_duration
        decision_end = trial_length

        # Add to period info
        period_info['rule_report'].append({'start': rule_report_start, 'end': rule_report_end})
        period_info['timing'].append({'start': timing_start, 'end': timing_end})
        period_info['decision'].append({'start': decision_start, 'end': decision_end})

    return period_info


def generate_data(task, model, num_trials=None, device='cpu'):
    """
    Generate multiple trials and collect all data.

    Handles both sequence tasks (with multiple trials per sequence and reward feedback)
    and non-sequence tasks (independent trials).

    Args:
        task: Task instance
        model: Trained model instance
        num_trials: Number of trials to generate (default: task.trials_per_sequence for sequence tasks, 40 otherwise)
        device: Device to run model on

    Returns:
        dict containing:
            - 'inputs': torch.Tensor of shape (num_trials, C, T) - all trial inputs
            - 'outputs': torch.Tensor of shape (num_trials, C_out, T) - model outputs
            - 'hidden_states': torch.Tensor of shape (num_trials, H, T) - hidden states
            - 'batch': dict with all batch metadata (trial_length, rule, t_m, t_s, etc.)
    """
    model.eval()

    # Check if this is a sequence task
    is_sequence_task = hasattr(task, 'trials_per_sequence') and task.trials_per_sequence > 1

    if is_sequence_task:
        # Sequence task: Generate trials with reward feedback
        num_trials = num_trials or task.trials_per_sequence

        # Generate batch with single sequence
        batch = task.generate_batch(batch_size=1)
        inputs = batch['inputs'].to(device)  # (1, N, T, C)

        B, N, T_max, C = inputs.shape
        assert B == 1, "Only batch_size=1 supported"

        # Get eval_mask for correctness checking
        eval_mask = batch['eval_mask'].to(device)  # (1, N, T, 4)
        targets = batch['targets'].to(device)  # (1, N, T, C_out)

        # Run trials sequentially with reward updates
        hidden = None
        all_outputs = []
        all_hidden_states = []

        with torch.no_grad():
            for n in range(N):
                # Get trial inputs: [1, T, C]
                trial_inputs = inputs[:, n, :, :].clone()

                # Initialize hidden state for this trial if needed
                if hidden is None:
                    hidden = torch.zeros(1, model.hidden_size, device=device)

                # Process timestep by timestep
                trial_outputs_list = []
                trial_hidden_list = []

                for t in range(T_max):
                    input_t = trial_inputs[:, t, :]  # [1, C]
                    output_t, hidden = model(input_t, hidden)  # [1, C_out], [1, H]
                    trial_outputs_list.append(output_t)
                    trial_hidden_list.append(hidden)

                # Stack to [1, T, C_out] and [1, T, H]
                trial_outputs = torch.stack(trial_outputs_list, dim=1)
                trial_hidden_states = torch.stack(trial_hidden_list, dim=1)

                all_outputs.append(trial_outputs)
                all_hidden_states.append(trial_hidden_states)

                if n < N - 1:
                    # Update reward signal for next trial
                    if hasattr(task, 'is_trial_correct_batch') and hasattr(task, 'set_reward_signal_batch'):
                        # Transpose for is_trial_correct_batch: expects [B, C_out, T]
                        trial_outputs_t = trial_outputs.transpose(1, 2)  # [1, C_out, T]
                        trial_targets = targets[:, n, :, :].transpose(1, 2)  # [1, C_out, T]
                        trial_eval_mask = eval_mask[:, n, :, :].transpose(1, 2)  # [1, 4, T]

                        is_correct_batch = task.is_trial_correct_batch(
                            trial_outputs_t,
                            trial_targets,
                            trial_eval_mask
                        )

                        # Update reward in next trial inputs: [1, T, C]
                        task.set_reward_signal_batch(
                            inputs[:, n+1, :, :],
                            is_correct_batch
                        )

        # Stack all outputs: [N, T, C_out] and [N, T, H]
        outputs = torch.cat(all_outputs, dim=0)  # (N, T, C_out)
        hidden_states = torch.cat(all_hidden_states, dim=0)  # (N, T, H)

        # Transpose to [N, C_out, T] and [N, H, T] for analysis compatibility
        outputs = outputs.transpose(1, 2)  # (N, C_out, T)
        hidden_states = hidden_states.transpose(1, 2)  # (N, H, T)
        inputs_flat = inputs[0].transpose(0, 1)  # (N, C, T)

        # Flatten batch metadata
        batch_flat = {}
        for key, value in batch.items():
            if key == 'metadata' and isinstance(value, dict):
                # Flatten nested metadata dict
                for meta_key, meta_value in value.items():
                    batch_flat[meta_key] = meta_value
            elif isinstance(value, torch.Tensor):
                batch_flat[key] = value[0] if value.ndim > 1 else value
            elif isinstance(value, np.ndarray):
                batch_flat[key] = value[0] if value.ndim > 1 else value
            else:
                batch_flat[key] = value

        return {
            'inputs': inputs_flat,
            'outputs': outputs,
            'hidden_states': hidden_states,
            'batch': batch_flat
        }

    else:
        # Non-sequence task: Generate independent trials
        num_trials = num_trials or 40  # Default to 40 trials

        # Generate batch with multiple independent trials
        batch = task.generate_batch(batch_size=num_trials)
        inputs = batch['inputs'].to(device)  # (B, T, C)

        B, T, C = inputs.shape

        # Run all trials through model timestep by timestep
        with torch.no_grad():
            # Initialize hidden state
            hidden = torch.zeros(B, model.hidden_size, device=device)

            # Process timestep by timestep
            outputs_list = []
            hidden_states_list = []

            for t in range(T):
                input_t = inputs[:, t, :]  # [B, C]
                output_t, hidden = model(input_t, hidden)  # [B, C_out], [B, H]
                outputs_list.append(output_t)
                hidden_states_list.append(hidden)

            # Stack to [B, T, C_out] and [B, T, H]
            outputs = torch.stack(outputs_list, dim=1)  # (B, T, C_out)
            hidden_states = torch.stack(hidden_states_list, dim=1)  # (B, T, H)

            # Transpose to [B, C_out, T] and [B, H, T] for analysis compatibility
            outputs = outputs.transpose(1, 2)  # (B, C_out, T)
            hidden_states = hidden_states.transpose(1, 2)  # (B, H, T)
            inputs = inputs.transpose(1, 2)  # (B, C, T)

        # Flatten metadata if nested (for backward compatibility)
        if 'metadata' in batch:
            metadata = batch.pop('metadata')
            batch.update(metadata)

        # Return data in [B, C, T] format for analysis
        return {
            'inputs': inputs,
            'outputs': outputs,
            'hidden_states': hidden_states,
            'batch': batch
        }


def get_metadata(batch, outputs=None, task=None):
    """
    Extract metadata from batch for visualization.

    Args:
        batch: dict containing batch data
        outputs: Optional model outputs for computing decisions
        task: Optional task object (required if outputs provided)

    Returns:
        dict with metadata including rule, has_instruction, is_switch, etc.
    """
    # Helper function to safely extract and flatten batch data
    def _extract_field(field_name):
        if field_name not in batch:
            return None
        value = batch[field_name]
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return value[0] if value.ndim > 1 else value
        return value

    # Extract basic metadata
    metadata = {
        'rule': _extract_field('rule'),
        't_m': _extract_field('t_m'),
        't_s': _extract_field('t_s'),
        'stim_direction': _extract_field('stim_direction')
    }

    # Add sequence-specific fields if they exist
    if 'has_instruction' in batch:
        metadata['has_instruction'] = _extract_field('has_instruction')
    if 'is_switch' in batch:
        metadata['is_switch'] = _extract_field('is_switch')

    # Compute decisions if outputs provided (decision = model's predicted direction)
    if outputs is not None and task is not None:
        trial_lengths = _extract_field('trial_length')
        if trial_lengths is not None:
            N = len(trial_lengths)
            decisions = []

            for i in range(N):
                trial_length = trial_lengths[i]
                eval_start = trial_length - int(300 / task.dt)

                decision_pred = outputs[i, 0, eval_start:trial_length].mean().item()

                # Decision is the predicted direction: +1 for right, -1 for left
                decisions.append(1 if decision_pred > 0 else -1)

            metadata['decision'] = np.array(decisions)

    return metadata


def plot_batch_trials(batch, outputs=None, targets=None, task=None,
                      num_trials=5, save_dir=None):
    """
    Plot trials from a batch with inputs, outputs, and targets.

    Handles both sequence tasks (4D tensors with reward) and non-sequence tasks (3D tensors).

    Args:
        batch: dict containing batch data
        outputs: Optional model outputs
        targets: Optional target outputs
        task: Task object with dt attribute
        num_trials: Number of trials to plot
        save_dir: Optional directory to save plots

    Returns:
        list of figure objects
    """
    # Helper function to extract batch fields
    def _get_field(field_name, index):
        if field_name not in batch:
            return None
        value = batch[field_name]
        if isinstance(value, (torch.Tensor, np.ndarray)):
            # Handle 2D arrays (sequence) or 1D arrays (non-sequence)
            if value.ndim > 1:
                return value[0][index]
            else:
                return value[index]
        return None

    trial_lengths = batch['trial_length'][0] if batch['trial_length'].ndim > 1 else batch['trial_length']
    num_trials_to_plot = min(num_trials, len(trial_lengths))
    total_trials = len(trial_lengths)

    inputs = batch['inputs'][0] if batch['inputs'].ndim > 3 else batch['inputs']
    if targets is not None:
        targets = batch['targets'][0] if batch['targets'].ndim > 3 else batch['targets']

    # Check number of input channels
    num_input_channels = inputs.shape[1]
    has_reward = num_input_channels > 5  # Channel 5 is reward in sequence tasks

    # Extract metadata for block structure
    rules = batch['rule'][0] if batch['rule'].ndim > 1 else batch['rule']
    has_instruction = batch['has_instruction'][0] if 'has_instruction' in batch and batch['has_instruction'].ndim > 1 else batch.get('has_instruction', None)

    figures = []

    for i in range(num_trials_to_plot):
        trial_length = trial_lengths[i]
        time_ms = np.arange(trial_length) * task.dt

        # Create figure with block structure at top
        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.25, 1, 1, 1], hspace=0.3)

        # Subplot 0: Block structure overview
        ax_block = fig.add_subplot(gs[0])
        trial_indices = np.arange(total_trials)
        colors = ['C0' if r == 1 else 'C1' for r in rules]

        # Plot all trials with appropriate markers
        if has_instruction is not None:
            for j in range(total_trials):
                if has_instruction[j]:
                    ax_block.scatter(j, rules[j], c=colors[j], s=100, marker='o',
                                   edgecolors='black', linewidths=1.5, alpha=0.6)
                else:
                    ax_block.scatter(j, rules[j], c=colors[j], s=100, marker='x',
                                   linewidths=2, alpha=0.6)
        else:
            # No instruction info available
            ax_block.scatter(trial_indices, rules, c=colors, s=100, alpha=0.6,
                           edgecolors='black', linewidths=1)

        # Highlight current trial
        ax_block.scatter(i, rules[i], s=400, facecolors='none',
                       edgecolors='lime', linewidths=3, zorder=10)

        ax_block.set_ylabel('Rule', fontsize=10)
        ax_block.set_yticks([-1, 1])
        ax_block.set_yticklabels(['Rule 2', 'Rule 1'], fontsize=9)
        if has_instruction is not None:
            ax_block.set_title(f'Batch Overview (○=instructed, ×=uninstructed, trial {i+1}/{total_trials} highlighted)',
                             fontsize=10)
        else:
            ax_block.set_title(f'Batch Overview (trial {i+1}/{total_trials} highlighted)', fontsize=10)
        ax_block.grid(True, alpha=0.3, axis='y')
        ax_block.set_xlim(-1, total_trials)
        ax_block.set_ylim(-1.5, 1.5)
        ax_block.set_xticks([])  # Hide x-axis ticks for cleaner look

        axes = [fig.add_subplot(gs[j]) for j in range(1, 4)]

        # Top: Inputs
        ax = axes[0]
        ax.plot(time_ms, inputs[i, 0, :trial_length], label='Timing stimulus', linewidth=1.5)
        ax.plot(time_ms, inputs[i, 1, :trial_length], label='Direction cue', linewidth=1.5)
        ax.plot(time_ms, inputs[i, 2, :trial_length], label='H fixation', linewidth=1.5, alpha=0.7)
        ax.plot(time_ms, inputs[i, 3, :trial_length], label='V fixation', linewidth=1.5, alpha=0.7)
        ax.plot(time_ms, inputs[i, 4, :trial_length], label='Rule cue', linewidth=1.5)
        # Only plot reward channel if it exists
        if has_reward:
            ax.plot(time_ms, inputs[i, 5, :trial_length], label='Reward', linewidth=1.5)
        ax.set_ylabel('Input value')

        # Build title with safe field extraction
        t_s = _get_field('t_s', i)
        t_m = _get_field('t_m', i)
        rule = _get_field('rule', i)
        ax.set_title(f'Trial {i+1}: t_s={t_s:.0f}ms, t_m={t_m:.0f}ms, rule={rule:+.0f}')
        ax.legend(loc='upper right', ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Middle: Rule outputs
        ax = axes[1]
        if outputs is not None:
            ax.plot(time_ms, outputs[i, 2, :trial_length], label='Rule report (model)', linewidth=2, color='C0')
        if targets is not None:
            ax.plot(time_ms, targets[i, 2, :trial_length], label='Rule report (target)',
                   linewidth=2, linestyle='--', color='C0', alpha=0.7)
        if outputs is not None:
            ax.plot(time_ms, outputs[i, 3, :trial_length], label='V fixation (model)', linewidth=2, color='C1')
        if targets is not None:
            ax.plot(time_ms, targets[i, 3, :trial_length], label='V fixation (target)',
                   linewidth=2, linestyle='--', color='C1', alpha=0.7)
        ax.set_ylabel('Output value')
        ax.set_title('Rule Report Outputs')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Bottom: Decision outputs
        ax = axes[2]
        if outputs is not None:
            ax.plot(time_ms, outputs[i, 0, :trial_length], label='Decision (model)', linewidth=2, color='C0')
        if targets is not None:
            ax.plot(time_ms, targets[i, 0, :trial_length], label='Decision (target)',
                   linewidth=2, linestyle='--', color='C0', alpha=0.7)
        if outputs is not None:
            ax.plot(time_ms, outputs[i, 1, :trial_length], label='H fixation (model)', linewidth=2, color='C1')
        if targets is not None:
            ax.plot(time_ms, targets[i, 1, :trial_length], label='H fixation (target)',
                   linewidth=2, linestyle='--', color='C1', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Output value')
        ax.set_title('Decision Outputs')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        if save_dir is not None:
            save_path = Path(save_dir) / f'trial_{i+1}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trial {i+1} to {save_path}")

        figures.append(fig)

    return figures


def _process_single_trial(model, inputs_tensor, device='cpu'):
    """
    Helper function to process a single trial through the time-step RNN.

    Args:
        model: RNN model
        inputs_tensor: [1, C, T] or [C, T] tensor
        device: device to run on

    Returns:
        outputs: [1, C_out, T] tensor
    """
    # Ensure batch dimension
    if inputs_tensor.ndim == 2:
        inputs_tensor = inputs_tensor.unsqueeze(0)

    # Transpose to [1, T, C]
    inputs_t = inputs_tensor.transpose(1, 2)  # [1, T, C]

    B, T, C = inputs_t.shape

    # Initialize hidden state
    hidden = torch.zeros(B, model.hidden_size, device=device)

    # Process timestep by timestep
    outputs_list = []
    for t in range(T):
        input_t = inputs_t[:, t, :]  # [1, C]
        output_t, hidden = model(input_t, hidden)  # [1, C_out], [1, H]
        outputs_list.append(output_t)

    # Stack to [1, T, C_out] and transpose to [1, C_out, T]
    outputs = torch.stack(outputs_list, dim=1).transpose(1, 2)

    return outputs


def compute_psychometric_curves(task, model, num_trials_per_interval=100, rules=None):
    """
    Compute psychometric curves showing P(Pro saccade) vs interval duration.

    Args:
        task: Task instance (should have discrete_eval=True)
        model: Trained model instance
        num_trials_per_interval: Number of trials to run per interval value
        rules: None for single-rule tasks, or list of rules (e.g., [1, -1]) for multi-rule tasks

    Returns:
        dict containing:
            - 'intervals': np.ndarray of interval values tested
            - 'accuracies': np.ndarray or dict of accuracies (if rules, dict with rule keys)
            - 'pro_probs': np.ndarray or dict of P(Pro) values
    """
    model.eval()
    intervals = task.eval_intervals

    if rules is None:
        # Single rule or no rule
        accuracies = []
        pro_probs = []

        with torch.no_grad():
            for delta_t in intervals:
                correct = 0
                pro_count = 0
                total = 0

                for _ in range(num_trials_per_interval):
                    trial = task.generate_trial(delta_t=delta_t)
                    inputs = torch.from_numpy(trial['inputs'])  # [C, T]
                    targets = torch.from_numpy(trial['targets']).unsqueeze(0)  # [1, C_out, T]

                    outputs = _process_single_trial(model, inputs)

                    T = trial['trial_length']
                    eval_start = T - int(300 / task.dt)
                    decision_pred = outputs[0, 0, eval_start:T].mean().item()
                    decision_target = targets[0, 0, eval_start:T].mean().item()

                    if (decision_pred > 0) == (decision_target > 0):
                        correct += 1

                    # Check if Pro choice (decision matches direction)
                    is_pro = ((decision_pred > 0 and trial['stim_direction'] > 0) or
                             (decision_pred < 0 and trial['stim_direction'] < 0))
                    if is_pro:
                        pro_count += 1

                    total += 1

                accuracies.append(correct / total)
                pro_probs.append(pro_count / total)

        return {
            'intervals': np.array(intervals),
            'accuracies': np.array(accuracies),
            'pro_probs': np.array(pro_probs)
        }

    else:
        # Multi-rule task
        results = {
            'intervals': np.array(intervals),
            'accuracies': {},
            'pro_probs': {}
        }

        with torch.no_grad():
            for rule in rules:
                rule_accuracies = []
                rule_pro_probs = []

                for delta_t in intervals:
                    correct = 0
                    pro_count = 0
                    total = 0

                    for _ in range(num_trials_per_interval):
                        trial = task.generate_trial(delta_t=delta_t, rule=rule)
                        inputs = torch.from_numpy(trial['inputs'])  # [C, T]
                        targets = torch.from_numpy(trial['targets']).unsqueeze(0)  # [1, C_out, T]

                        outputs = _process_single_trial(model, inputs)

                        T = trial['trial_length']
                        eval_start = T - int(300 / task.dt)
                        decision_pred = outputs[0, 0, eval_start:T].mean().item()
                        decision_target = targets[0, 0, eval_start:T].mean().item()

                        if (decision_pred > 0) == (decision_target > 0):
                            correct += 1

                        is_pro = ((decision_pred > 0 and trial['stim_direction'] > 0) or
                                 (decision_pred < 0 and trial['stim_direction'] < 0))
                        if is_pro:
                            pro_count += 1

                        total += 1

                    rule_accuracies.append(correct / total)
                    rule_pro_probs.append(pro_count / total)

                results['accuracies'][rule] = np.array(rule_accuracies)
                results['pro_probs'][rule] = np.array(rule_pro_probs)

        return results


def visualize_pca(pca_data, explained_variance, trial_lengths, metadata=None,
                  plot_3d=False, num_trials=20, color_by='rule',
                  figsize=(10, 8), save_path=None):
    """
    Create static PCA visualization (2D or 3D).

    Args:
        pca_data: np.ndarray of shape (num_trials, max_time, n_components)
        explained_variance: np.ndarray of explained variance ratios
        trial_lengths: list or array of actual trial lengths
        metadata: dict with trial metadata (rules, decisions, stim_direction, etc.)
        plot_3d: bool, whether to plot in 3D
        num_trials: Number of trials to plot
        color_by: str - what to color by
                       Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                       Continuous (with colorbar): 't_m', 't_s'
        figsize: tuple of figure size
        save_path: Optional path to save figure

    Returns:
        fig, ax
    """
    if plot_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Set up coloring
    color_map, labels = _get_color_scheme(color_by, metadata)

    # Limit number of trials
    num_trials = min(num_trials, len(trial_lengths))

    # Check if using continuous colormap
    use_continuous = color_map == 'continuous'

    if use_continuous and metadata is not None and color_by in metadata:
        # Set up continuous colormap
        import matplotlib.cm as cm
        cmap = cm.viridis
        color_values = metadata[color_by][:num_trials]
        vmin, vmax = color_values.min(), color_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot each trial
    for i in range(num_trials):
        T_trial = trial_lengths[i]

        # Get color for this trial
        if use_continuous and metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = scalar_map.to_rgba(color_val)
            label = None
        elif metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = color_map.get(color_val, 'gray')
            # Determine label (only show once per unique value)
            if color_val in labels and labels[color_val] not in [item.get_label() for item in ax.get_children() if hasattr(item, 'get_label')]:
                label = labels[color_val]
            else:
                label = None
        else:
            color = f'C{i % 10}'
            label = f'Trial {i+1}' if i == 0 else None

        # Plot trajectory
        if plot_3d:
            ax.plot(pca_data[i, :T_trial, 0],
                   pca_data[i, :T_trial, 1],
                   pca_data[i, :T_trial, 2],
                   color=color, alpha=0.6, linewidth=1.5, label=label)

            # Mark start
            ax.scatter(pca_data[i, 0, 0], pca_data[i, 0, 1], pca_data[i, 0, 2],
                      color=color, s=50, marker='*', alpha=0.8, zorder=5)

            # Mark end
            ax.scatter(pca_data[i, T_trial-1, 0], pca_data[i, T_trial-1, 1], pca_data[i, T_trial-1, 2],
                      color=color, s=50, marker='x', alpha=0.8, zorder=5)
        else:
            ax.plot(pca_data[i, :T_trial, 0],
                   pca_data[i, :T_trial, 1],
                   color=color, alpha=0.6, linewidth=1.5, label=label)

            # Mark start and end
            ax.scatter(pca_data[i, 0, 0], pca_data[i, 0, 1],
                      color=color, s=50, marker='*', alpha=0.8, zorder=5)
            ax.scatter(pca_data[i, T_trial-1, 0], pca_data[i, T_trial-1, 1],
                      color=color, s=50, marker='x', alpha=0.8, zorder=5)

    # Labels
    if plot_3d:
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} var)')
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%} var)')
    else:
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} var)')

    # Title
    title = f'Hidden State Trajectories in PC Space ({"3D" if plot_3d else "2D"})'
    ax.set_title(title)

    # Add legend or colorbar
    if use_continuous and metadata is not None and color_by in metadata:
        # Add colorbar
        cbar = plt.colorbar(scalar_map, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(labels, rotation=270, labelpad=20)
    else:
        ax.legend(loc='best')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig, ax


def animate_pca(pca_data, explained_variance, period_lengths, metadata,
                period_name='', plot_simultaneously=True, plot_3d=True, color_by='rule',
                num_trials=20, interval=50, figsize=(14, 10), save_path=None):
    """
    Create animated PCA visualization (2D or 3D).

    Args:
        pca_data: np.ndarray of shape (num_trials, max_time, n_components)
        explained_variance: np.ndarray of explained variance ratios
        period_lengths: list of actual period/trial lengths
        metadata: dict with trial metadata
        period_name: str, name of period being animated (for title)
        plot_simultaneously: bool, whether all trials evolve together
        plot_3d: bool, whether to animate in 3D (default True)
        color_by: str - what to color by
                       Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                       Continuous (with colorbar): 't_m', 't_s'
        num_trials: Number of trials to animate
        interval: Animation interval in ms
        figsize: tuple of figure size
        save_path: Optional path to save animation (as .mp4 or .gif)

    Returns:
        FuncAnimation object
    """
    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    # Set up color scheme
    color_map, labels = _get_color_scheme(color_by, metadata)

    # Limit number of trials
    num_trials = min(num_trials, len(period_lengths))

    # Check if using continuous colormap
    use_continuous = color_map == 'continuous'

    if use_continuous and metadata is not None and color_by in metadata:
        # Set up continuous colormap
        import matplotlib.cm as cm
        cmap = cm.viridis
        color_values = metadata[color_by][:num_trials]
        vmin, vmax = color_values.min(), color_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 20, height_ratios=[0.1, 1])

    # Time display panel
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(0, 1)
    ax_time.axis('off')

    # Main plot (2D or 3D)
    if plot_3d:
        ax_main = fig.add_subplot(gs[1, :], projection='3d')
        ax_main.set_xlim(pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5)
        ax_main.set_ylim(pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5)
        ax_main.set_zlim(pca_data[:, :, 2].min() - 0.5, pca_data[:, :, 2].max() + 0.5)
        ax_main.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax_main.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax_main.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
    else:
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.set_xlim(pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5)
        ax_main.set_ylim(pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5)
        ax_main.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax_main.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax_main.grid(True, alpha=0.3)

    title = f'RNN Trajectories ({"3D" if plot_3d else "2D"}) - All Trials Evolving Together' if plot_simultaneously else f'RNN Trajectories ({"3D" if plot_3d else "2D"}) - Sequential'
    if period_name:
        title = f'{period_name.upper()} Period - ' + title
    ax_main.set_title(title)

    # Time text
    time_text = ax_time.text(0.5, 0.5, '', ha='center', va='center',
                            fontsize=16, fontweight='bold')

    # Add legend or colorbar
    if use_continuous and metadata is not None and color_by in metadata:
        # Add colorbar
        cbar = plt.colorbar(scalar_map, ax=ax_main, pad=0.1, shrink=0.6)
        cbar.set_label(labels, rotation=270, labelpad=20)
    else:
        # Add discrete legend
        legend_patches = []
        for val, label in labels.items():
            if val in color_map:
                legend_patches.append(mpatches.Patch(color=color_map[val], label=label, alpha=0.7))
        if legend_patches:
            ax_main.legend(handles=legend_patches, loc='upper right')

    # Initial view angle
    initial_elev = 20
    initial_azim = 45

    # Create line objects for each trial
    lines = []
    current_points = []
    end_markers = []

    for i in range(num_trials):
        # Get color for this trial
        if use_continuous and metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = scalar_map.to_rgba(color_val)
        elif metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = color_map.get(color_val, 'gray')
        else:
            color = f'C{i % 10}'

        # Get trial end position
        period_len = period_lengths[i]

        if plot_3d:
            line, = ax_main.plot([], [], [], color=color, alpha=0.7, linewidth=2)
            lines.append(line)
            point = ax_main.scatter([], [], [], color=color, s=100, marker='o',
                                   edgecolors='black', linewidths=2, zorder=10)
            current_points.append(point)
            # Add static end marker (x)
            end_marker = ax_main.scatter([pca_data[i, period_len-1, 0]],
                                        [pca_data[i, period_len-1, 1]],
                                        [pca_data[i, period_len-1, 2]],
                                        color=color, s=60, marker='x',
                                        linewidths=2, alpha=0.8, zorder=5)
            end_markers.append(end_marker)
        else:
            line, = ax_main.plot([], [], color=color, alpha=0.7, linewidth=2)
            lines.append(line)
            point, = ax_main.plot([], [], 'o', color=color, markersize=10,
                                 markeredgecolor='black', markeredgewidth=2, zorder=10)
            current_points.append(point)
            # Add static end marker (x)
            end_marker = ax_main.scatter([pca_data[i, period_len-1, 0]],
                                        [pca_data[i, period_len-1, 1]],
                                        color=color, s=60, marker='x',
                                        linewidths=2, alpha=0.8, zorder=5)
            end_markers.append(end_marker)

    # Find maximum length
    max_length = max(period_lengths[:num_trials])

    def init():
        for line in lines:
            line.set_data([], [])
            if hasattr(line, 'set_3d_properties'):
                line.set_3d_properties([])
        for point in current_points:
            if hasattr(point, '_offsets3d'):
                point._offsets3d = ([], [], [])
            else:
                point.set_data([], [])
        time_text.set_text('')
        if hasattr(ax_main, 'view_init'):
            ax_main.view_init(elev=initial_elev, azim=initial_azim)
        return lines + current_points + [time_text]

    def animate(frame):
        # Update each trial's trajectory up to current time
        for i in range(num_trials):
            period_len = period_lengths[i]

            if frame < period_len:
                # Trial is still ongoing
                lines[i].set_data(pca_data[i, :frame+1, 0], pca_data[i, :frame+1, 1])

                if hasattr(lines[i], 'set_3d_properties'):
                    # 3D line
                    lines[i].set_3d_properties(pca_data[i, :frame+1, 2])
                    current_points[i]._offsets3d = ([pca_data[i, frame, 0]],
                                                   [pca_data[i, frame, 1]],
                                                   [pca_data[i, frame, 2]])
                else:
                    # 2D line
                    current_points[i].set_data([pca_data[i, frame, 0]], [pca_data[i, frame, 1]])
            elif frame >= period_len:
                # Trial has ended
                lines[i].set_data(pca_data[i, :period_len, 0], pca_data[i, :period_len, 1])

                if hasattr(lines[i], 'set_3d_properties'):
                    # 3D line
                    lines[i].set_3d_properties(pca_data[i, :period_len, 2])
                    current_points[i]._offsets3d = ([], [], [])
                else:
                    # 2D line
                    current_points[i].set_data([], [])

        # Update time display (assuming 10ms per step)
        current_time_ms = frame * 10
        if period_name:
            time_text.set_text(f'{period_name.upper()}: {current_time_ms} ms from period start')
        else:
            time_text.set_text(f'Time: {current_time_ms} ms')

        # Rotate view slightly (only for 3D)
        if hasattr(ax_main, 'view_init'):
            ax_main.view_init(elev=initial_elev, azim=initial_azim + frame * 0.2)

        return lines + current_points + [time_text]

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=max_length,
                        interval=interval, blit=False, repeat=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if str(save_path).endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif str(save_path).endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=20)

        print(f"Saved animation to {save_path}")

    return anim


def _get_color_scheme(color_by, metadata):
    """
    Get color mapping and labels for visualization.

    Returns:
        color_map: dict mapping values to colors (or 'continuous' for continuous colormaps)
        labels: dict mapping values to label strings (or colorbar label for continuous)
    """
    if color_by == 'rule':
        color_map = {1: 'C0', -1: 'C1'}
        labels = {1: 'Rule 1', -1: 'Rule 2'}
    elif color_by == 'decision':
        color_map = {1: 'purple', -1: 'orange'}
        labels = {1: 'Right', -1: 'Left'}
    elif color_by == 'stim_direction':
        color_map = {1: 'purple', -1: 'orange'}
        labels = {1: 'Right', -1: 'Left'}
    elif color_by == 't_m':
        color_map = 'continuous'
        labels = 'Measured interval t_m (ms)'
    elif color_by == 't_s':
        color_map = 'continuous'
        labels = 'True interval t_s (ms)'
    elif color_by == 'instructed':
        color_map = {True: 'blue', False: 'gray', 1: 'blue', 0: 'gray'}
        labels = {True: 'Instructed', False: 'Uninstructed', 1: 'Instructed', 0: 'Uninstructed'}
    elif color_by == 'switch':
        color_map = {True: 'red', False: 'blue', 1: 'red', 0: 'blue'}
        labels = {True: 'Switch', False: 'No Switch', 1: 'Switch', 0: 'No Switch'}
    elif color_by == 'reward':
        color_map = {1: 'gold', 0: 'gray', True: 'gold', False: 'gray'}
        labels = {1: 'Rewarded', 0: 'Not Rewarded', True: 'Rewarded', False: 'Not Rewarded'}
    else:
        # Default coloring
        color_map = {}
        labels = {}

    return color_map, labels
