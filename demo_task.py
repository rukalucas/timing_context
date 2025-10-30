"""Vibe-coded. Demo the Timing Decision Task and Instructed Timing Task in ./tasks.py."""

from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
import torch

from tasks import InstructedTimingTask, SequenceInstructedTask, InferredTask, TransitionTask
from models.rnn import RNN
from trainers.utils import BaseTrainer

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_trials', 1, 'Number of task trials to plot.')
flags.DEFINE_bool('discrete', False, 'Use discrete evaluation intervals.')
flags.DEFINE_float('rule_cue_prob', 0.5, 'Rule cue probability for transition task (0.0=inferred, 1.0=fully instructed).')

def demo_instructed_timing_task(num_trials: int, discrete: bool):
    """Demo the InstructedTimingTask with new batch structure."""

    task = InstructedTimingTask(
        dt=10.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        discrete_eval=discrete,
    )

    for _ in range(num_trials):
        # Generate batch - returns list with 1 trial dict for single-trial tasks
        batch = task.generate_batch(batch_size=1)
        trial_dict = batch[0]  # First (and only) trial in sequence

        inputs = trial_dict['inputs'][0].numpy()  # [T, 5]
        targets = trial_dict['targets'][0].numpy()  # [T, 2]
        eval_mask = trial_dict['eval_mask'][0].numpy()  # [T, 2]
        loss_mask = trial_dict['loss_mask'][0].numpy()  # [T, 2]

        rule = trial_dict['metadata']['rule'][0]
        t_s = trial_dict['metadata']['t_s'][0]
        t_m = trial_dict['metadata']['t_m'][0]
        stim_direction = trial_dict['metadata']['stim_direction'][0]
        decision = trial_dict['metadata']['decision'][0]

        rule_name = "Rule 1 (short→pro, long→anti)" if rule == 1 else "Rule 2 (short→anti, long→pro)"

        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        time_steps = np.arange(inputs.shape[0]) * task.dt

        # First subplot: Rule cue and Vertical cue
        ax = axes[0]
        ax.plot(time_steps, inputs[:, 2], label='Rule Cue', alpha=0.7, linewidth=2)
        ax.plot(time_steps, inputs[:, 3], label='Vertical Cue', alpha=0.7, linewidth=2)
        ax.set_ylabel('Input Value')
        ax.set_title(f"InstructedTimingTask: {rule_name}, t_s={t_s:.0f}ms, t_m={t_m:.0f}ms, stim_dir={stim_direction:+d}, Dec={decision:+d}")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Second subplot: Center fixation, Horizontal cue, and Reward cue
        ax = axes[1]
        ax.plot(time_steps, inputs[:, 0], label='Center Fixation', alpha=0.7, linewidth=2)
        ax.plot(time_steps, inputs[:, 1], label='Horizontal Cue', alpha=0.7, linewidth=2)
        ax.plot(time_steps, inputs[:, 4], label='Reward Cue', alpha=0.7, linewidth=2)
        ax.set_ylabel('Input Value')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Third subplot: Vertical eye position (rule report output)
        ax = axes[2]
        for i in range(len(time_steps)):
            if loss_mask[i, 1] == 0:
                ax.axvspan(time_steps[i], time_steps[i] + task.dt, color='gray', alpha=0.3, linewidth=0)
        ax.plot(time_steps, targets[:, 1], 'b-', label='Vertical Eye Position (target)', linewidth=2)
        if eval_mask[:, 1].any():
            eval_idx = np.where(eval_mask[:, 1] > 0)[0]
            ax.axvspan(time_steps[eval_idx[0]], time_steps[eval_idx[-1]], color='green', alpha=0.15, linewidth=0, label='Eval region')
        ax.set_ylabel('Target Value')
        ax.set_title('Vertical Eye Position / Rule Report (gray = not trained, green = evaluated)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Fourth subplot: Horizontal eye position (decision output)
        ax = axes[3]
        for i in range(len(time_steps)):
            if loss_mask[i, 0] == 0:
                ax.axvspan(time_steps[i], time_steps[i] + task.dt, color='gray', alpha=0.3, linewidth=0)
        ax.plot(time_steps, targets[:, 0], 'b-', label='Horizontal Eye Position (target)', linewidth=2)
        if eval_mask[:, 0].any():
            eval_idx = np.where(eval_mask[:, 0] > 0)[0]
            ax.axvspan(time_steps[eval_idx[0]], time_steps[eval_idx[-1]], color='red', alpha=0.15, linewidth=0, label='Eval region')
        ax.set_ylabel('Target Value')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Horizontal Eye Position / Timing Decision (gray = not trained, red = evaluated)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    plt.show()


def demo_sequence_instructed_task(num_sequences: int, discrete: bool):
    """Demo the SequenceInstructedTask with interactive navigation."""

    task = SequenceInstructedTask(
        dt=10.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        rule_cue_prob=0.7,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from first trial to build sequence structure
        rules = np.array([batch[i]['metadata']['rule'][0] for i in range(num_trials)])
        has_instruction = np.array([batch[i]['metadata']['has_instruction'][0] for i in range(num_trials)])
        block_ids = np.array([batch[i]['metadata']['block_id'][0] for i in range(num_trials)])
        is_switch = np.array([batch[i]['metadata']['is_switch'][0] for i in range(num_trials)])
        trial_lengths = np.array([batch[i]['trial_lengths'][0].item() for i in range(num_trials)])

        # State for interactive navigation
        state = {'current_trial': 0}

        # Create figure
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('Use ← → arrow keys to navigate trials', fontsize=12, y=0.995)

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            fig.clear()
            gs = fig.add_gridspec(4, 1, hspace=0.35, top=0.96, height_ratios=[1, 3, 3, 3])

            # Subplot 1: Block structure overview with current trial highlighted
            ax1 = fig.add_subplot(gs[0])
            trial_indices = np.arange(num_trials)
            colors = ['blue' if r == 1 else 'red' for r in rules]
            ax1.scatter(trial_indices, rules, c=colors, s=100, alpha=0.6, edgecolors='black')
            for i, instr in enumerate(has_instruction):
                if instr:
                    marker = 'o'
                    ax1.scatter(i, rules[i], marker=marker, s=150, c=colors[i], edgecolors='black', linewidths=2)
                else:
                    marker = 'x'
                    ax1.scatter(i, rules[i], marker=marker, s=150, c=colors[i], linewidths=2)

            # Highlight current trial
            ax1.scatter(trial_idx, rules[trial_idx], s=400, facecolors='none',
                       edgecolors='lime', linewidths=3, zorder=10)

            ax1.set_ylabel('Rule')
            ax1.set_yticks([-1, 1])
            ax1.set_yticklabels(['Rule 2', 'Rule 1'])
            ax1.set_title('Sequence Overview (○=instructed, ×=uninstructed, green circle=current)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1, num_trials)
            ax1.set_ylim(-2.0, 2.0)

            # Extract trial-specific data from batch list structure
            trial_inputs = batch[trial_idx]['inputs'][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]['targets'][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]['loss_mask'][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]['eval_mask'][0].numpy()  # [T, 2]

            T = trial_inputs.shape[0]
            time_steps = np.arange(T) * task.dt
            trial_len = trial_lengths[trial_idx]

            # Subplot 2: Horizontal inputs (center fixation, horizontal cue, reward)
            ax2 = fig.add_subplot(gs[1])
            ax2.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax2.plot(time_steps, trial_inputs[:, 0], 'b-', linewidth=2, label='Center Fixation', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 1], 'orange', linewidth=2, label='Horizontal Cue', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 4], 'cyan', linewidth=1.5, label='Reward Cue', alpha=0.6)

            ax2.set_ylabel('Input Value')
            ax2.set_title(f'Trial {trial_idx} - Horizontal Inputs')
            ax2.legend(loc='upper right', fontsize=8, ncol=4)
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Rule cue and Vertical cue
            ax3 = fig.add_subplot(gs[2])
            ax3.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax3.plot(time_steps, trial_inputs[:, 2], 'purple', linewidth=2, label='Rule Cue', alpha=0.7)
            ax3.plot(time_steps, trial_inputs[:, 3], 'magenta', linewidth=1.5, label='Vertical Cue', alpha=0.6)

            ax3.set_ylabel('Input Value')
            ax3.set_title('Vertical & Rule Inputs')
            ax3.legend(loc='upper right', fontsize=8, ncol=3)
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Targets with eval mask
            ax4 = fig.add_subplot(gs[3])

            # Show loss mask regions and eval mask regions
            for t in range(len(time_steps)):
                # Check if all masks are zero (padding)
                if trial_loss_mask[t, 0] == 0 and trial_loss_mask[t, 1] == 0:
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='darkgray', alpha=0.4, linewidth=0)
                # Show eval regions
                if trial_eval_mask[t, 1] > 0:  # Vertical eval (rule)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='green', alpha=0.15, linewidth=0)
                if trial_eval_mask[t, 0] > 0:  # Horizontal eval (decision)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='red', alpha=0.15, linewidth=0)

            ax4.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5)
            ax4.plot(time_steps, trial_targets[:, 1], 'g-', linewidth=2.5, label='Vertical Eye Position (rule)', alpha=0.8)
            ax4.plot(time_steps, trial_targets[:, 0], 'r-', linewidth=2.5, label='Horizontal Eye Position (decision)', alpha=0.8)

            ax4.set_ylabel('Target Value')
            ax4.set_xlabel('Time (ms)')
            ax4.set_title('Targets (dark gray=padding, green=rule eval, red=decision eval)')
            ax4.legend(loc='upper right', fontsize=8, ncol=3)
            ax4.grid(True, alpha=0.3)

            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == 'right':
                state['current_trial'] = min(state['current_trial'] + 1, num_trials - 1)
                plot_trial(state['current_trial'])
            elif event.key == 'left':
                state['current_trial'] = max(state['current_trial'] - 1, 0)
                plot_trial(state['current_trial'])

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_inferred_task(num_sequences: int, discrete: bool):
    """Demo the InferredTask with interactive navigation."""

    task = InferredTask(
        dt=10.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from trials
        rules = np.array([batch[i]['metadata']['rule'][0] for i in range(num_trials)])
        is_switch = np.array([batch[i]['metadata']['is_switch'][0] for i in range(num_trials)])
        trial_lengths = np.array([batch[i]['trial_lengths'][0].item() for i in range(num_trials)])

        # State for interactive navigation
        state = {'current_trial': 0}

        # Create figure
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('Use ← → arrow keys to navigate trials', fontsize=12, y=0.995)

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            fig.clear()
            gs = fig.add_gridspec(4, 1, hspace=0.35, top=0.96, height_ratios=[1, 3, 3, 3])

            # Subplot 1: Block structure overview with current trial highlighted
            ax1 = fig.add_subplot(gs[0])
            trial_indices = np.arange(num_trials)
            colors = ['blue' if r == 1 else 'red' for r in rules]
            # Mark switches with different markers
            for i, switch in enumerate(is_switch):
                marker = 'x' if switch else 'o'
                ax1.scatter(i, rules[i], c=colors[i], s=150, marker=marker, alpha=0.6, linewidths=2)

            # Highlight current trial
            ax1.scatter(trial_idx, rules[trial_idx], s=400, facecolors='none',
                       edgecolors='lime', linewidths=3, zorder=10)

            ax1.set_ylabel('Rule')
            ax1.set_yticks([-1, 1])
            ax1.set_yticklabels(['Rule 2', 'Rule 1'])
            ax1.set_title('Sequence Overview (×=switch, ○=stay, green circle=current)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1, num_trials)
            ax1.set_ylim(-2.0, 2.0)

            # Extract trial-specific data from batch list structure
            trial_inputs = batch[trial_idx]['inputs'][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]['targets'][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]['loss_mask'][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]['eval_mask'][0].numpy()  # [T, 2]

            T = trial_inputs.shape[0]
            time_steps = np.arange(T) * task.dt
            trial_len = trial_lengths[trial_idx]

            # Subplot 2: Inputs (center fixation, horizontal cue, reward)
            ax2 = fig.add_subplot(gs[1])
            ax2.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax2.plot(time_steps, trial_inputs[:, 0], 'b-', linewidth=2, label='Center Fixation', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 1], 'orange', linewidth=2, label='Horizontal Cue', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 4], 'cyan', linewidth=1.5, label='Reward', alpha=0.6)

            ax2.set_ylabel('Input Value')
            ax2.set_title(f'Trial {trial_idx} - Inputs (InferredTask: infer rule from reward)')
            ax2.legend(loc='upper right', fontsize=8, ncol=4)
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Vertical cue and Rule cue (always 0 for inferred)
            ax3 = fig.add_subplot(gs[2])
            ax3.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax3.plot(time_steps, trial_inputs[:, 2], 'purple', linewidth=2, label='Rule Cue (always 0)', alpha=0.7)
            ax3.plot(time_steps, trial_inputs[:, 3], 'magenta', linewidth=1.5, label='Vertical Cue', alpha=0.6)

            ax3.set_ylabel('Input Value')
            ax3.set_title('Vertical & Rule Inputs (Rule inferred from reward)')
            ax3.legend(loc='upper right', fontsize=8, ncol=3)
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Targets with eval mask
            ax4 = fig.add_subplot(gs[3])

            # Show loss mask regions and eval mask regions
            for t in range(len(time_steps)):
                # Check if all masks are zero (padding)
                if trial_loss_mask[t, 0] == 0 and trial_loss_mask[t, 1] == 0:
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='darkgray', alpha=0.4, linewidth=0)
                # Show eval regions
                if trial_eval_mask[t, 1] > 0:  # Vertical eval (rule)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='green', alpha=0.15, linewidth=0)
                if trial_eval_mask[t, 0] > 0:  # Horizontal eval (decision)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='red', alpha=0.15, linewidth=0)

            ax4.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5)
            ax4.plot(time_steps, trial_targets[:, 1], 'g-', linewidth=2.5, label='Vertical Eye Position (rule)', alpha=0.8)
            ax4.plot(time_steps, trial_targets[:, 0], 'r-', linewidth=2.5, label='Horizontal Eye Position (decision)', alpha=0.8)

            ax4.set_ylabel('Target Value')
            ax4.set_xlabel('Time (ms)')
            ax4.set_title('Targets (dark gray=padding, green=rule eval, red=decision eval)')
            ax4.legend(loc='upper right', fontsize=8, ncol=3)
            ax4.grid(True, alpha=0.3)

            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == 'right':
                state['current_trial'] = min(state['current_trial'] + 1, num_trials - 1)
                plot_trial(state['current_trial'])
            elif event.key == 'left':
                state['current_trial'] = max(state['current_trial'] - 1, 0)
                plot_trial(state['current_trial'])

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_transition_task(num_sequences: int, discrete: bool, rule_cue_prob: float):
    """Demo the TransitionTask with interactive navigation.

    TransitionTask allows adjustable rule_cue_prob to interpolate between
    fully instructed (rule_cue_prob=1.0) and fully inferred (rule_cue_prob=0.0).
    """

    task = TransitionTask(
        dt=10.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        rule_cue_prob=rule_cue_prob,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from trials
        rules = np.array([batch[i]['metadata']['rule'][0] for i in range(num_trials)])
        has_instruction = np.array([batch[i]['metadata']['has_instruction'][0] for i in range(num_trials)])
        is_switch = np.array([batch[i]['metadata']['is_switch'][0] for i in range(num_trials)])
        trial_lengths = np.array([batch[i]['trial_lengths'][0].item() for i in range(num_trials)])

        # State for interactive navigation
        state = {'current_trial': 0}

        # Create figure
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f'TransitionTask (rule_cue_prob={rule_cue_prob:.2f}) - Use ← → arrow keys to navigate trials',
                     fontsize=12, y=0.995)

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            fig.clear()
            gs = fig.add_gridspec(4, 1, hspace=0.35, top=0.96, height_ratios=[1, 3, 3, 3])

            # Subplot 1: Block structure overview with current trial highlighted
            ax1 = fig.add_subplot(gs[0])
            trial_indices = np.arange(num_trials)
            colors = ['blue' if r == 1 else 'red' for r in rules]

            # Mark instructed vs uninstructed, and switches
            for i in range(num_trials):
                if has_instruction[i]:
                    marker = 'o'
                    size = 150
                    alpha = 0.6
                else:
                    marker = 'x'
                    size = 150
                    alpha = 0.6
                ax1.scatter(i, rules[i], c=colors[i], s=size, marker=marker, alpha=alpha, linewidths=2, edgecolors='black')

            # Highlight current trial
            ax1.scatter(trial_idx, rules[trial_idx], s=400, facecolors='none',
                       edgecolors='lime', linewidths=3, zorder=10)

            ax1.set_ylabel('Rule')
            ax1.set_yticks([-1, 1])
            ax1.set_yticklabels(['Rule 2', 'Rule 1'])
            ax1.set_title(f'Sequence Overview (○=instructed, ×=uninstructed, green circle=current)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1, num_trials)
            ax1.set_ylim(-2.0, 2.0)

            # Extract trial-specific data from batch list structure
            trial_inputs = batch[trial_idx]['inputs'][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]['targets'][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]['loss_mask'][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]['eval_mask'][0].numpy()  # [T, 2]

            T = trial_inputs.shape[0]
            time_steps = np.arange(T) * task.dt
            trial_len = trial_lengths[trial_idx]

            # Subplot 2: Horizontal inputs (center fixation, horizontal cue, reward)
            ax2 = fig.add_subplot(gs[1])
            ax2.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax2.plot(time_steps, trial_inputs[:, 0], 'b-', linewidth=2, label='Center Fixation', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 1], 'orange', linewidth=2, label='Horizontal Cue', alpha=0.7)
            ax2.plot(time_steps, trial_inputs[:, 4], 'cyan', linewidth=1.5, label='Reward Cue', alpha=0.6)

            ax2.set_ylabel('Input Value')
            ax2.set_title(f'Trial {trial_idx} - Horizontal Inputs (has_instruction={has_instruction[trial_idx]})')
            ax2.legend(loc='upper right', fontsize=8, ncol=4)
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Rule cue and Vertical cue
            ax3 = fig.add_subplot(gs[2])
            ax3.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Trial end')
            ax3.plot(time_steps, trial_inputs[:, 2], 'purple', linewidth=2, label='Rule Cue', alpha=0.7)
            ax3.plot(time_steps, trial_inputs[:, 3], 'magenta', linewidth=1.5, label='Vertical Cue', alpha=0.6)

            ax3.set_ylabel('Input Value')
            ax3.set_title('Vertical & Rule Inputs')
            ax3.legend(loc='upper right', fontsize=8, ncol=3)
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Targets with eval mask
            ax4 = fig.add_subplot(gs[3])

            # Show loss mask regions and eval mask regions
            for t in range(len(time_steps)):
                # Check if all masks are zero (padding)
                if trial_loss_mask[t, 0] == 0 and trial_loss_mask[t, 1] == 0:
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='darkgray', alpha=0.4, linewidth=0)
                # Show eval regions
                if trial_eval_mask[t, 1] > 0:  # Vertical eval (rule)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='green', alpha=0.15, linewidth=0)
                if trial_eval_mask[t, 0] > 0:  # Horizontal eval (decision)
                    ax4.axvspan(time_steps[t], time_steps[t] + task.dt, color='red', alpha=0.15, linewidth=0)

            ax4.axvline(trial_len * task.dt, color='red', linestyle=':', linewidth=2, alpha=0.5)
            ax4.plot(time_steps, trial_targets[:, 1], 'g-', linewidth=2.5, label='Vertical Eye Position (rule)', alpha=0.8)
            ax4.plot(time_steps, trial_targets[:, 0], 'r-', linewidth=2.5, label='Horizontal Eye Position (decision)', alpha=0.8)

            ax4.set_ylabel('Target Value')
            ax4.set_xlabel('Time (ms)')
            ax4.set_title('Targets (dark gray=padding, green=rule eval, red=decision eval)')
            ax4.legend(loc='upper right', fontsize=8, ncol=3)
            ax4.grid(True, alpha=0.3)

            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == 'right':
                state['current_trial'] = min(state['current_trial'] + 1, num_trials - 1)
                plot_trial(state['current_trial'])
            elif event.key == 'left':
                state['current_trial'] = max(state['current_trial'] - 1, 0)
                plot_trial(state['current_trial'])

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_rnn_processing():
    """Demonstrate timestep-by-timestep RNN processing with ITI and reward feedback.

    Uses the same forward pass logic as in training (via BaseTrainer._forward_pass)
    to ensure consistency.

    Shows:
    - Single timestep forward pass: [B, 5] -> [B, 2]
    - Hidden state evolution across multiple trials with ITI
    - Reward feedback processing during ITI (as in training)
    """
    print("\n" + "="*70)
    print("RNN TIMESTEP-BY-TIMESTEP PROCESSING DEMO")
    print("="*70 + "\n")

    # Create sequence task with 3 trials and RNN model
    task = SequenceInstructedTask(dt=10.0, input_noise_std=0.1, trials_per_sequence=3,
                                   inter_trial_interval=1000.0, reward_duration=300.0)
    model = RNN(input_size=5, hidden_size=128, output_size=2, dt=10.0, tau=100.0)
    model.eval()

    # Generate a batch with 1 sequence
    batch = task.generate_batch(batch_size=1)
    num_trials = len(batch)

    # Convert batch to float32
    for trial_dict in batch:
        for key in ['inputs', 'targets', 'loss_mask', 'eval_mask']:
            if key in trial_dict:
                trial_dict[key] = trial_dict[key].float()

    H = model.hidden_size
    print(f"Sequence with {num_trials} trials, hidden_size={H}\n")

    # Create trainer instance to use its forward pass method
    trainer = BaseTrainer(model=model, log_dir='logs/temp')

    # Use trainer's forward pass with return_hidden=True to get everything in one pass
    with torch.no_grad():
        (all_outputs, all_inputs_list, all_targets_list, all_masks, all_hidden,
         iti_regions, trial_boundaries, all_iti_inputs, all_iti_targets) = \
            trainer._forward_pass(batch, task, include_iti=True, return_hidden=True)

    # Interleave trial and ITI data for plotting
    # all_inputs_list contains only trial inputs, all_iti_inputs contains ITI inputs
    # Need to interleave them based on trial_boundaries
    combined_inputs = []
    combined_targets = []
    combined_loss_masks = []

    for trial_idx in range(num_trials):
        combined_inputs.append(all_inputs_list[trial_idx])
        combined_targets.append(all_targets_list[trial_idx])
        combined_loss_masks.append(batch[trial_idx]['loss_mask'])
        # Add ITI after each trial except the last
        if trial_idx < len(all_iti_inputs):
            combined_inputs.append(all_iti_inputs[trial_idx])
            combined_targets.append(all_iti_targets[trial_idx])
            # ITI has zero loss mask
            iti_len = all_iti_inputs[trial_idx].shape[1]
            iti_loss_mask = torch.zeros(1, iti_len, 2)
            combined_loss_masks.append(iti_loss_mask)

    # Concatenate and convert to numpy
    all_inputs_concat = torch.cat(combined_inputs, dim=1)[0].numpy()  # [T_total, 5]
    all_targets_concat = torch.cat(combined_targets, dim=1)[0].numpy()  # [T_total, 2]
    all_loss_masks_concat = torch.cat(combined_loss_masks, dim=1)[0].numpy()  # [T_total, 2]
    all_hidden_states = torch.stack(all_hidden, dim=0)[:, 0, :].numpy()  # [T_total, H]

    T_total = all_inputs_concat.shape[0]
    time_ms = np.arange(T_total) * task.dt

    # Identify regions where loss_mask is 0 for both outputs (padded regions)
    zero_mask_regions = []
    in_zero_region = False
    region_start = None
    for t in range(T_total):
        is_zero = (all_loss_masks_concat[t, 0] == 0) and (all_loss_masks_concat[t, 1] == 0)
        if is_zero and not in_zero_region:
            region_start = t
            in_zero_region = True
        elif not is_zero and in_zero_region:
            zero_mask_regions.append((region_start, t))
            in_zero_region = False
    # Close final region if still open
    if in_zero_region:
        zero_mask_regions.append((region_start, T_total))

    # Select first 10 hidden dimensions to plot
    hidden_dims_to_plot = list(range(10))  # First 10 dimensions

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle(f'RNN Processing Across {num_trials} Trials with ITI', fontsize=14, fontweight='bold')

    # Mark regions where loss_mask=0 (ITI + padding) on all plots
    for region_start, region_end in zero_mask_regions:
        for ax in axes:
            # Compute time coordinates directly to handle region_end = T_total
            start_time = region_start * task.dt
            end_time = region_end * task.dt
            ax.axvspan(start_time, end_time, color='lightgray', alpha=0.3, linewidth=0)

    # Mark trial boundaries
    for boundary in trial_boundaries[1:-1]:
        for ax in axes:
            ax.axvline(time_ms[boundary], color='black', linestyle=':', alpha=0.4, linewidth=1)

    # Plot 1: Inputs
    ax = axes[0]
    ax.plot(time_ms, all_inputs_concat[:, 0], label='Center Fixation', linewidth=1.5, alpha=0.7)
    ax.plot(time_ms, all_inputs_concat[:, 1], label='Horizontal Cue', linewidth=1.5, alpha=0.7)
    ax.plot(time_ms, all_inputs_concat[:, 2], label='Rule Cue', linewidth=1.5, alpha=0.6)
    ax.plot(time_ms, all_inputs_concat[:, 3], label='Vertical Cue', linewidth=1.5, alpha=0.6)
    ax.plot(time_ms, all_inputs_concat[:, 4], label='Reward Cue', linewidth=2, alpha=0.8)
    ax.set_ylabel('Input Value')
    ax.set_title('Inputs (5 channels, gray=loss_mask=0)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Outputs vs Targets
    # Recompute outputs from hidden states for plotting (already have them)
    # For simplicity, just show targets since we processed everything above
    ax = axes[1]
    ax.plot(time_ms, all_targets_concat[:, 0], 'b-', linewidth=2, label='Horizontal Eye (target)', alpha=0.7)
    ax.plot(time_ms, all_targets_concat[:, 1], 'r-', linewidth=2, label='Vertical Eye (target)', alpha=0.7)
    ax.set_ylabel('Target Value')
    ax.set_title('Target Outputs', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Hidden state dimensions (first 10)
    ax = axes[2]
    cmap = plt.cm.tab10
    for i, dim_idx in enumerate(hidden_dims_to_plot):
        ax.plot(time_ms, all_hidden_states[:, dim_idx], label=f'h{dim_idx}',
                linewidth=1.2, alpha=0.7, color=cmap(i))
    ax.set_ylabel('Hidden Value')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Hidden State Evolution (first 10 dimensions)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Clean up temporary trainer
    trainer.close()

    plt.show()


def main(argv):
    """Plot stimulus and desired response for task trials."""
    if len(argv) < 2:
        print("Usage: python demo_task.py <task_type> [options]")
        print("Available tasks: instructed, sequence_instructed, inferred, transition, rnn_demo")
        return

    task_type = argv[1]

    if task_type == 'instructed':
        print(f"Demonstrating InstructedTimingTask with {FLAGS.num_trials} trials...")
        demo_instructed_timing_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == 'sequence_instructed':
        print(f"Demonstrating SequenceInstructedTask with {FLAGS.num_trials} sequences...")
        demo_sequence_instructed_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == 'inferred':
        print(f"Demonstrating InferredTask with {FLAGS.num_trials} sequences...")
        demo_inferred_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == 'transition':
        print(f"Demonstrating TransitionTask with {FLAGS.num_trials} sequences (rule_cue_prob={FLAGS.rule_cue_prob})...")
        demo_transition_task(FLAGS.num_trials, FLAGS.discrete, FLAGS.rule_cue_prob)
    elif task_type == 'rnn_demo':
        demo_rnn_processing()
    else:
        print(f"Unknown task type: {task_type}")
        print("Available tasks: instructed, sequence_instructed, inferred, transition, rnn_demo")


if __name__ == '__main__':
    app.run(main)
