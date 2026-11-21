"""Learn one mapped human performance using simulation-only reference tracking."""

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.human_arm_body_sway",
    assets=JUMPER_ASSETS,
    description="imitate the mapped 03_arm_body_sway performance with changing foot support",
    tags=("imitation", "jumper", "human", "flat"),
)
