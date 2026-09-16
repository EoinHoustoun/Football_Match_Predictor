"""The bankroll chart's end-of-line chips ("If all win", "Expected", "If all
lose") share one x position, so when two outcomes are close in money they
drew on top of each other. `spread_label_shifts` pushes them apart vertically.
"""
from chart_labels import now_badge_offset, spread_label_shifts


def _pixel_positions(ys, shifts, y_lo, y_hi, plot_px):
    per_px = (y_hi - y_lo) / plot_px
    return [y / per_px + s for y, s in zip(ys, shifts)]


def test_far_apart_labels_are_not_moved():
    ys = [60_000, 45_000, 30_000]
    assert all(abs(s) < 1e-9 for s in spread_label_shifts(ys, 20_000, 70_000, 440, min_gap_px=44))


def test_close_labels_end_at_least_min_gap_apart():
    ys = [60_000, 44_600, 44_000]
    shifts = spread_label_shifts(ys, 20_000, 70_000, 440, min_gap_px=44)
    px = sorted(_pixel_positions(ys, shifts, 20_000, 70_000, 440))
    assert all(b - a >= 44 - 1e-6 for a, b in zip(px, px[1:]))


def test_order_is_preserved_and_input_order_kept():
    ys = [44_000, 60_000, 44_600]  # unsorted input
    shifts = spread_label_shifts(ys, 20_000, 70_000, 440, min_gap_px=44)
    px = _pixel_positions(ys, shifts, 20_000, 70_000, 440)
    assert px[1] > px[2] > px[0]


def test_identical_values_still_separate():
    ys = [50_000, 50_000, 50_000]
    shifts = spread_label_shifts(ys, 20_000, 70_000, 440, min_gap_px=44)
    px = sorted(_pixel_positions(ys, shifts, 20_000, 70_000, 440))
    assert px[2] - px[0] >= 88 - 1e-6


def test_now_badge_sits_on_the_side_away_from_the_incoming_line():
    # Rising into NOW: the line arrives from below-left, so badge goes above.
    ax, ay = now_badge_offset(rising=True)
    assert ax < 0 and ay < 0  # plotly: negative ay is above the point
    ax, ay = now_badge_offset(rising=False)
    assert ax < 0 and ay > 0
