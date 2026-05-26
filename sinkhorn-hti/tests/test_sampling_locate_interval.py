"""Boundary cases for ``sampling.locate_interval``."""

from __future__ import annotations

import math

import pytest

from sinkhorn_hti.sampling import locate_interval


def test_locate_interval_boundaries():
    anchors = (0.0, 0.5, 1.0)

    k, s = locate_interval(0.0, anchors)
    assert (k, s) == (0, 0.0)

    k, s = locate_interval(1.0, anchors)
    assert (k, s) == (1, 1.0)

    k, s = locate_interval(0.25, anchors)
    assert k == 0 and math.isclose(s, 0.5)

    k, s = locate_interval(0.75, anchors)
    assert k == 1 and math.isclose(s, 0.5)

    k, s = locate_interval(-0.1, anchors)
    assert (k, s) == (0, 0.0)

    k, s = locate_interval(1.1, anchors)
    assert (k, s) == (1, 1.0)

    # Exact interior anchor: bisect_right lands at the next sub-interval.
    k, s = locate_interval(0.5, anchors)
    assert k == 1 and math.isclose(s, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
