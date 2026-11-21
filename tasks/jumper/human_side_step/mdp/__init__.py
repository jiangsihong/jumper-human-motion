"""Terms specific to the finite human side-step performance."""

from __future__ import annotations


def clip_finished(env, command_name: str):
    """Finish before MotionCommand can wrap and teleport to another frame."""
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - 2
