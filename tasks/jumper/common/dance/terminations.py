"""Termination mechanisms for finite reference clips."""

from __future__ import annotations


def clip_finished(env, command_name: str, last_frame_margin: int):
    """End before the motion command wraps and resets physical state."""
    if last_frame_margin < 1:
        raise ValueError("Clip end margin must be at least one frame")
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - last_frame_margin
