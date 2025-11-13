"""Profiling script for analyzing training performance.

Usage:
    python prof.py configs/instructed.yaml
    python prof.py configs/instructed.yaml --steps 5
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torch.profiler import profile, ProfilerActivity

from main import create_model, create_trainer, set_random_seed


def main():
    parser = argparse.ArgumentParser(description='Profile training performance')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--steps', type=int, default=3, help='Number of steps to profile (default: 3)')
    parser.add_argument('--save-trace', action='store_true', help='Save Chrome trace to logs/profiler/trace.json')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    conf = OmegaConf.load(args.config)

    # Disable wandb for profiling (zero overhead)
    import os
    os.environ['WANDB_MODE'] = 'disabled'

    # Ensure wandb config exists (required by trainer, but will be disabled)
    if 'wandb' not in conf:
        conf['wandb'] = {'project': 'profiling'}

    # Override total_steps to match profiling steps
    conf['training']['total_steps'] = args.steps

    # Set random seed
    random_seed = conf.get('random_seed', 42)
    set_random_seed(random_seed)

    # Create model and trainer
    print("Creating model and trainer...")
    model = create_model(conf)
    trainer = create_trainer(conf, model, log_dir='logs/profiler')

    print(f"\nProfiling {args.steps} training steps...")
    print(f"Tasks: {trainer.task_names}")
    print(f"Batch size: {trainer.batch_size}\n")

    # Set up profiler
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True
    ) as prof:

        # Profile the actual train() method
        trainer.train()
        prof.step()

    print("\nProfiling complete!\n")

    # Print summary table
    print("="*80)
    print("CPU Time Summary (sorted by total CPU time)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Optionally save trace
    if args.save_trace:
        profiler_dir = Path('logs/profiler')
        profiler_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profiler_dir / 'trace.json'
        prof.export_chrome_trace(str(trace_path))
        print(f"\nChrome trace saved to: {trace_path}")
        print("To view: Open chrome://tracing and load the trace file.")


if __name__ == '__main__':
    main()
