"""Track the measured 02_kick_step performance with its full expressive amplitude.

Reference statistics: 458 frames at 50 Hz, 9.16 seconds; joint amplitude RMS .170 rad and speed RMS .418 rad/s. Right-rear local X/Z ranges are 163/61 mm.
These are source measurements, not learned-policy results. All weights below
are starting choices. Two or more feet are within 5 mm of standing site height
on every reference frame; this is not a contact-force or stability certificate.
The actor uses simulated position error/linear velocity: hardware deployment
requires a compatible state estimator. No operator-command axes are used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg

from ..common.dance.env import dance_env_cfg
from ..common.dance.terminations import clip_finished
from ..common.dance.support import SupportSchedule

MEDIA = Path(__file__).resolve().parent / "media"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build one finite-clip tracking task, with all tuning owned here."""
    with np.load(MEDIA / "support/schedule.npz") as data:
        schedule = SupportSchedule(
            stance=data["stance"],
            dt=float(data["dt"]),
            height_tolerance=0.005,
            max_site_drop=0.0035,
            min_support_feet=2,
        )
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        support_schedule=schedule,
        # Four-second windows; evaluate the full clip separately from its start.
        episode_s=4.0,
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
        # Position tolerance is about .44 of the source amplitude; velocity
        # tolerance is about two thirds of its measured RMS speed.
        joint_pos_std=0.075,
        joint_pos_weight=1.5,
        joint_vel_std=0.30,
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
    # Centimeter-scale tracking tolerances rather than humanoid .3 m values.
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.025
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
        params={"command_name": "motion", "last_frame_margin": 2},
    )
    # RR is the principal kicking leg: score its full path directly so a
    # moving front arm cannot hide a stationary kick in the whole-body average.
    cfg.rewards["right_rear_kick"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.025, "body_names": ("RR_foot_tip_link",)},
    )
    cfg.viewer.azimuth = 115.0
    cfg.viewer.elevation = -20.0
    cfg.viewer.distance = 0.78
    return cfg
