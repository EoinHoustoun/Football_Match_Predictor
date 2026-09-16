"""Kelly fraction sweep (50% to 100%; 20-45% was run ad hoc and peaks at 40-45%) on the gameweek backtest with live
calibration (isotonic, refit each gameweek on the prior 40 weeks).

Main's live settings otherwise (50% gameweek cap, 5 pending bets). Reports per
season, worst season, deepest drawdown, and the three seasons COMPOUNDED,
since the live bankroll rolls forward from one season to the next. Three
seasons select nothing on their own: read it for the shape of the curve, not
the single best cell.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402
import validate_draw_calibration as c  # noqa: E402


def main() -> int:
    pack = pickle.loads(c.PACK.read_bytes())
    hist = pd.concat([pack[s][2] for s in pack if s != "2026-27"],
                     ignore_index=True).sort_values("Date")
    fn = c.make_fn(hist, "isotonic", 280)
    live = v.live_main_config()
    rows = []
    print(f"{'Kelly':>6} " + " ".join(f"{s:>9}" for s in c.TEST_SEASONS)
          + f" {'worst':>8} {'maxDD':>6} {'compounded x':>13} {'bets':>5}")
    for k in np.round(np.arange(0.50, 1.0001, 0.05), 2):
        prof, dd, bets = [], [], 0
        for s in c.TEST_SEASONS:
            dfs, ftr, bt, _ = pack[s]
            log, summ = pf.ev_backtest_simulate(
                bt, dfs, df_features=ftr, initial_bankroll=v.INITIAL,
                enable_simultaneous_correction=True, odds_source="Max",
                detect_source="PS", calibrators=None, calibrator_fn=fn,
                **c.LIVE_KW, **{**live, "kelly_frac": float(k)})
            prof.append(summ.get("profit", 0.0))
            dd.append(v._max_drawdown_pct(log) if not log.empty else 0.0)
            bets += summ.get("n_bets", 0)
        mult = float(np.prod([1 + p / v.INITIAL for p in prof]))
        rows.append({"kelly": float(k), "per_season": dict(zip(c.TEST_SEASONS, prof)),
                     "worst": min(prof), "max_dd": round(max(dd), 1),
                     "compounded_multiple": round(mult, 3), "bets": bets})
        print(f"{k:>6.2f} " + " ".join(f"{p:>+9,.0f}" for p in prof)
              + f" {min(prof):>+8,.0f} {max(dd):>5.1f}% {mult:>12.2f}x {bets:>5}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"sweep_kelly_fraction_{ts}.json"
    path.write_text(json.dumps({"config": {k: sorted(x) if isinstance(x, set) else x
                                           for k, x in live.items()},
                                "rows": rows}, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
