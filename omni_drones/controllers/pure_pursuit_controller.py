from __future__ import annotations

from .intercept_baseline_common import GuidanceBaseline


class PurePursuitController(GuidanceBaseline):
    """Pure pursuit: steer straight at the evader's current position."""

    def __init__(self, ctbr, task_cfg):
        super().__init__(ctbr)
