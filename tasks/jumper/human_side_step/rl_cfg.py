"""PPO settings for one short clip and an initial 256-environment GPU batch."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Conservative starting settings; a smoke run is not convergence evidence."""
    return jumper_ppo_baseline(
        experiment_name="jumper.human_side_step",
        # This choreography is asymmetric: mirrored data would erase the motion.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        # Smaller than the long-dance model for the initial 8192-sample batch.
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        init_std=0.3,
        entropy_coef=0.001,
        learning_rate=3.0e-4,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=32,
        max_iterations=2000,
    )


def runner_cls() -> type:
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
