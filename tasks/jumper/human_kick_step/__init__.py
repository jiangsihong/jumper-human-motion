"""Learn one mapped human performance using simulation-only reference tracking."""

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.human_kick_step",
    assets=JUMPER_ASSETS,
    description="imitate the mapped 02_kick_step performance with changing foot support",
    tags=("imitation", "jumper", "human", "flat"),
)
