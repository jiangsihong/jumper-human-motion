"""Learn the user's mapped side-step clip; simulation-only motion tracking."""

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.human_side_step",
    assets=JUMPER_ASSETS,
    description="imitate the mapped human side-step with two arms and four stepping legs",
    tags=("imitation", "jumper", "human", "flat"),
)
