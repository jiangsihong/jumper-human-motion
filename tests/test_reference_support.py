"""Pin failures where stepping data could otherwise bypass ground consistency.

A swinging foot must be accepted without accepting an airborne stance foot or a
wrong ground plane. Schedule changes must not silently reuse an earlier cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from tasks.jumper.common.dance import motion
from tasks.jumper.common.dance.support import SupportSchedule


def schedule(mask=None, tolerance=0.005):
    if mask is None:
        mask = np.array([[True, True, True, False], [True, True, False, True]])
    return SupportSchedule(mask, 0.02, tolerance, 0.0035, 2)


def test_swing_is_allowed_but_not_a_floating_or_underground_stance():
    home = 0.009
    z = home + np.array([[0.0, 0.0, 0.0, 0.06], [0.0, 0.0, 0.06, 0.0]])
    check = schedule()
    assert check.validate(z, home, 0.02)["min_support_feet"] == 3
    # Control: the old fixed-support check must reject this genuine swing.
    with pytest.raises(ValueError, match="out of plane"):
        motion._check_support_feet(z, home, "measured")
    floating = z + 0.02
    with pytest.raises(ValueError, match="stance foot"):
        check.validate(floating, home, 0.02)
    underground = z.copy()
    underground[0, 0] -= 0.004
    with pytest.raises(ValueError, match="home-height tolerance"):
        check.validate(underground, home, 0.02)


def test_wrong_support_labels_or_timeline_cannot_silently_pass():
    z = 0.009 + np.array([[0.0, 0.0, 0.0, 0.06], [0.0, 0.0, 0.06, 0.0]])
    check = schedule()
    assert check.validate(z, 0.009, 0.02)
    wrong = check.stance.copy()
    wrong[0] = [True, False, False, True]
    with pytest.raises(ValueError, match="stance foot"):
        schedule(wrong).validate(z, 0.009, 0.02)
    too_few = check.stance.copy()
    too_few[0] = [True, False, False, False]
    with pytest.raises(ValueError, match="too few"):
        schedule(too_few).validate(z, 0.009, 0.02)
    with pytest.raises(ValueError, match="duration"):
        check.validate(np.tile(z, (4, 1)), 0.009, 0.02)


def test_support_policy_changes_invalidate_conversion_cache(tmp_path, monkeypatch):
    source = tmp_path / "clip.npz"
    source.write_bytes(b"constant clip payload; only the support policy changes")
    monkeypatch.setattr(motion, "_model_digest", lambda model: "constant-model")
    check = schedule()
    base = motion._fingerprint(source, 0.02, "measured", object(), check)
    assert base == motion._fingerprint(source, 0.02, "measured", object(), schedule())
    assert base != motion._fingerprint(
        source, 0.02, "measured", object(), schedule(tolerance=0.004)
    )
    changed = check.stance.copy()
    changed[0, 1] = False
    assert base != motion._fingerprint(source, 0.02, "measured", object(), schedule(changed))
    assert base != motion._fingerprint(source, 0.02, "measured", object())
