"""Track the measured v2 side-step without reducing its expressive amplitude.

Measured on the source clip, 698 frames at 50 Hz: root X/Y range 74/110 mm,
root height 85-103 mm, joint amplitude RMS .225 rad, joint-speed RMS .722 rad/s,
and body linear component RMS .044 m/s. These are reference statistics, not
trained-policy results. All weights below are explicit starting choices.

The actor uses simulated root position error and linear velocity. This task is
simulation-only; deployment needs a compatible real state estimator and contract.
There are no velocity-command axes or operator bindings: the clip is the command.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg

from ..common.dance.env import dance_env_cfg
from .mdp import clip_finished

MEDIA = Path(__file__).resolve().parent / "media"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build one finite-clip tracking task, with all tuning owned here."""
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        # Six-second sampled windows; the full source is 13.96 seconds.
        episode_s=6.0,
        rsi_pose_range={
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.002, 0.002),
            "roll": (-0.025, 0.025),
            "pitch": (-0.025, 0.025),
            "yaw": (-0.025, 0.025),
        },
        rsi_velocity_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
        rsi_joint_range=(-0.025, 0.025),
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # .10 rad is .44 of reference amplitude; .5 rad/s is .69 of its speed.
        joint_pos_std=0.10,
        joint_pos_weight=1.5,
        joint_vel_std=0.5,
        joint_vel_weight=0.75,
        # Lower smoothness costs than the long dance: do not buy a stationary
        # solution by charging away the deliberate stepping movement.
        action_rate_weight=-0.02,
        action_acc_weight=-0.01,
        power_weight=-0.005,
        torque_headroom_weight=-0.5,
        joint_limit_weight=-10.0,
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        anchor_height_error=0.035,
        anchor_tilt_error=0.3,
        # This compares with the moving reference foot, not with the floor.
        support_foot_error=0.045,
        push_interval_s=(5.0, 10.0),
        push_velocity_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
    )
    # The inherited .3 m root tolerance rewards standing through a .11 m step.
    # Read-only source probe: frozen first pose averages .954 at .3, .181 at .035.
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.035
    cfg.rewards["motion_global_root_pos"].weight = 1.5
    cfg.rewards["motion_body_pos"].params["std"] = 0.035
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.08
    cfg.rewards["motion_body_ang_vel"].params["std"] = 1.0
    # Explicitly shrink the humanoid skeleton's 25/50/50 mm COM offsets to a
    # small initial robustness band. These 2 mm values are not hardware measurements.
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.002, 0.002),
        1: (-0.002, 0.002),
        2: (-0.002, 0.002),
    }
    for group in ("actor", "critic"):
        cfg.observations[group].terms["motion_anchor_pos_b"] = ObservationTermCfg(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=UniformNoiseCfg(n_min=-0.003, n_max=0.003) if group == "actor" else None,
        )
    cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    cfg.terminations["clip_finished"] = TerminationTermCfg(
        func=clip_finished,
        time_out=True,
        params={"command_name": "motion"},
    )
    cfg.viewer.azimuth = 115.0
    cfg.viewer.elevation = -20.0
    cfg.viewer.distance = 0.78
    return cfg
