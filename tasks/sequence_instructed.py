"""Instructed Timing Task with sequence of trials and block structure."""

from .utils import BaseTask


class SequenceInstructedTask(BaseTask):
    """
    Sequence-based instructed timing task with block structure. BPTT through sequence of trials.

    Trial Structure:
    1. Rule report epoch (variable delay + 700ms response) - reports current rule
    2. Timing decision epoch (variable delay + pulses + 700ms response) - timing-based decision
    3. ITI - 1000ms with reward feedback, handled during training

    Block Structure:
    - Block length = block_min + geometric(1/block_mean)
    - First 4 trials of block: always instructed
    - Remaining trials: instructed with probability rule_cue_prob
    - Rules alternate between blocks
    """

    def __init__(self, **kwargs):
        """Initialize sequence instructed timing task."""
        super().__init__(**kwargs)
        self.name = "Sequence Instructed"
