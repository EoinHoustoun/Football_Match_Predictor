"""Collision-free label placement for the bankroll charts.

Plotly annotations have no layout engine: two chips at the same x simply draw
over each other. These helpers work out the pixel offsets up front.
"""
from __future__ import annotations


def spread_label_shifts(ys: list[float], y_lo: float, y_hi: float,
                        plot_px: float, min_gap_px: float = 44) -> list[float]:
    """Vertical pixel shifts (plotly `yshift`, positive = up) that keep labels
    anchored at `ys` at least `min_gap_px` apart, moving each as little as
    possible. Shifts are returned in the order of `ys`.
    """
    if not ys:
        return []
    span = (y_hi - y_lo) or 1.0
    px = [(y - y_lo) / span * plot_px for y in ys]
    order = sorted(range(len(ys)), key=lambda i: px[i])
    placed = [px[i] for i in order]

    # Push upward from the bottom so neighbours clear each other...
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap_px)
    # ...then centre the group on where the labels wanted to be, so a cluster
    # spreads both ways instead of only drifting up.
    drift = sum(placed[k] - px[order[k]] for k in range(len(placed))) / len(placed)
    placed = [p - drift for p in placed]
    for k in range(1, len(placed)):  # re-check after centring
        placed[k] = max(placed[k], placed[k - 1] + min_gap_px)

    shifts = [0.0] * len(ys)
    for k, i in enumerate(order):
        shifts[i] = placed[k] - px[i]
    return shifts


def now_badge_offset(rising: bool) -> tuple[int, int]:
    """Arrow offset (ax, ay) for the NOW badge when pending projection lines
    occupy the space to its right. The badge moves up-left or down-left, to
    the side the settled line is NOT coming from."""
    return (-70, -48) if rising else (-70, 48)
