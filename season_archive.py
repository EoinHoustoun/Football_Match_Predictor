"""Season archiving for the F_PRED paper-trading portfolios.

Closing a season is the one moment where the portfolio-safety rule is easiest
to break: the only existing path to a fresh bankroll is the 🗑️ Reset button,
which deletes every bet. This module makes the reset non-destructive by
refusing to run until the outgoing season is safely archived.

Layout::

    data/seasons/2025-26/
        portfolio.json        verbatim copy of the live main portfolio
        portfolio_two.json    verbatim copy of the live research portfolio
        manifest.json         closing summary (no API keys)

Lives outside ``portfolio.py`` deliberately — that module is already 98k and
this is a self-contained concern with its own failure modes.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import portfolio as pf

SEASONS_DIR = pf.DATA_DIR / "seasons"

MAIN_NAME     = "portfolio.json"
TWO_NAME      = "portfolio_two.json"
MANIFEST_NAME = "manifest.json"

# Settings keys that must never reach an archived manifest.
_SECRET_KEYS = ("odds_api_key",)


class ArchiveError(RuntimeError):
    """Raised when an archive operation would lose or overwrite live data."""


# ── Paths ─────────────────────────────────────────────────────────────────────

def _season_dir(season: str) -> Path:
    return SEASONS_DIR / season


def is_archived(season: str) -> bool:
    """True when `season` has a complete archive on disk."""
    d = _season_dir(season)
    return (d / MANIFEST_NAME).exists() and (d / MAIN_NAME).exists()


def list_archived_seasons() -> list[str]:
    """Archived season labels, newest first."""
    if not SEASONS_DIR.exists():
        return []
    return sorted(
        (d.name for d in SEASONS_DIR.iterdir() if d.is_dir() and is_archived(d.name)),
        reverse=True,
    )


# ── Summarising ───────────────────────────────────────────────────────────────

def _settings_label(settings: dict) -> str | None:
    """Main tags its config as `main_settings_label`, Mock Two as `v2_*`."""
    for key in ("main_settings_label", "v2_settings_label", "settings_label"):
        if settings.get(key):
            return settings[key]
    return None


def _summarise(p: dict) -> dict:
    """Closing numbers for one portfolio. Never includes secrets."""
    stats = pf.portfolio_stats(p)
    clv   = pf.clv_summary(p)
    return {
        "initial_bankroll": p.get("initial_bankroll"),
        "final_bankroll":   p.get("bankroll"),
        "profit":           stats["profit"],
        "roi_pct":          stats["roi"],
        "n_bets":           len(p.get("bets", [])),
        "n_settled":        stats["n_settled"],
        "win_rate":         stats["win_rate"],
        "median_clv":       clv.get("median_clv"),
        "n_clv_tagged":     clv.get("n", 0),
        "settings_label":   _settings_label(p.get("settings", {})),
    }


# ── Archiving ─────────────────────────────────────────────────────────────────

def _snapshot(live_file: Path, dest: Path, loaded: dict) -> None:
    """Copy the live file verbatim, then verify byte identity.

    Falls back to serialising the loaded state when no file exists yet, which
    happens for Mock Two before its first save.
    """
    if live_file.exists():
        shutil.copy2(live_file, dest)
        if dest.read_bytes() != live_file.read_bytes():
            raise ArchiveError(f"Archived copy of {live_file} does not match the original")
    else:
        dest.write_text(json.dumps(loaded, indent=2, default=str))


def archive_season(season: str, *, overwrite: bool = False) -> dict:
    """Snapshot both live portfolios into `data/seasons/<season>/`.

    Returns the manifest. Refuses to clobber an existing archive unless
    `overwrite=True`.
    """
    dest = _season_dir(season)
    if is_archived(season) and not overwrite:
        raise ArchiveError(
            f"{season} is already archived at {dest}. "
            f"Pass overwrite=True only if you mean to replace it."
        )
    dest.mkdir(parents=True, exist_ok=True)

    main = pf.load_portfolio()
    two  = pf.load_portfolio_two()

    _snapshot(pf.PORTFOLIO_FILE, dest / MAIN_NAME, main)
    _snapshot(pf.MOCK2_PORTFOLIO_FILE, dest / TWO_NAME, two)

    manifest = {
        "season":     season,
        "closed_at":  datetime.now(timezone.utc).isoformat(),
        "portfolios": {"main": _summarise(main), "mock_two": _summarise(two)},
    }
    _assert_no_secrets(manifest)
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _assert_no_secrets(manifest: dict) -> None:
    raw = json.dumps(manifest, default=str)
    for portfolio in (pf.load_portfolio(), pf.load_portfolio_two()):
        for key in _SECRET_KEYS:
            secret = portfolio.get("settings", {}).get(key)
            if secret and secret in raw:
                raise ArchiveError(f"Manifest would leak {key}")


def load_archived_season(season: str) -> dict:
    """Read an archived season back as `{manifest, main, mock_two}`."""
    d = _season_dir(season)
    if not is_archived(season):
        raise ArchiveError(f"No archive for {season} at {d}")

    two_file = d / TWO_NAME
    return {
        "manifest": json.loads((d / MANIFEST_NAME).read_text()),
        "main":     json.loads((d / MAIN_NAME).read_text()),
        "mock_two": json.loads(two_file.read_text()) if two_file.exists() else None,
    }


# ── Season picker (drives the Season Review tab) ──────────────────────────────

def live_season(default: str | None = None) -> str | None:
    """The season the live portfolios are currently on.

    Portfolios written before the archive existed carry no season tag, so the
    caller passes the dataset's latest season as the fallback.
    """
    return pf.load_portfolio().get("season") or default


def season_options(live: str | None) -> list[str]:
    """Seasons offered in the review picker: the live one first, then archives."""
    archived = list_archived_seasons()
    if live is None:
        return archived
    return [live] + [s for s in archived if s != live]


def _empty_like(reference: dict | None) -> dict:
    """A zeroed portfolio, for archives predating Mock Two."""
    initial = (reference or {}).get("initial_bankroll", 0.0)
    return {"initial_bankroll": initial, "bankroll": initial,
            "bets": [], "settings": {}}


def load_season_view(season: str, *, live_season: str | None) -> dict:
    """Portfolios for `season`, from the live files or the archive.

    Returns `{season, is_live, main, mock_two, manifest}`. `manifest` is None
    for the live season, which has not closed yet.
    """
    if season == live_season:
        return {
            "season":   season,
            "is_live":  True,
            "main":     pf.load_portfolio(),
            "mock_two": pf.load_portfolio_two(),
            "manifest": None,
        }

    archived = load_archived_season(season)
    return {
        "season":   season,
        "is_live":  False,
        "main":     archived["main"],
        "mock_two": archived["mock_two"] or _empty_like(archived["main"]),
        "manifest": archived["manifest"],
    }


# ── Bankroll chart scopes ─────────────────────────────────────────────────────

def previous_season(live: str | None) -> str | None:
    """The newest archived season that is not the live one."""
    return next((s for s in list_archived_seasons() if s != live), None)


def all_time_portfolio(archived: list[tuple[str, dict]],
                       live: tuple[str, dict]) -> tuple[dict, list[tuple[int, str]]]:
    """Join archived seasons (oldest first) and the live one into one portfolio
    whose `bankroll_history` runs continuously from the first opening bankroll.

    Only the live season contributes pending bets. When a season opened on a
    different bankroll from the one the previous season closed on, a settled
    "carry-over" row bridges the gap so the line still ends on the real live
    bankroll. Returns `(portfolio, marks)`, where marks are
    `(settled_index, season_label)` for where each season's line begins.
    """
    def _settled(p):
        return sorted((b for b in p.get("bets", [])
                       if b.get("status") in ("won", "lost") and b.get("settled_at")),
                      key=lambda b: b["settled_at"])

    seasons = list(archived) + [live]
    bets: list[dict] = []
    marks: list[tuple[int, str]] = []
    running = seasons[0][1].get("initial_bankroll", 0.0)
    for i, (label, p) in enumerate(seasons):
        season_bets = _settled(p)
        gap = round(p.get("initial_bankroll", running) - running, 2)
        if i > 0 and abs(gap) >= 0.01:
            # bankroll_history orders by settled_at, so stamp the bridge just
            # after the previous season's last settlement.
            last = bets[-1]["settled_at"] if bets else ""
            bets.append({"id": f"carry-{label}", "home": "Bankroll", "away": "carry-over",
                         "selection": label, "status": "won" if gap > 0 else "lost",
                         "profit": gap, "stake": 0.0, "odds": 1.0,
                         "settled_at": last + "~"})
            running += gap
        marks.append((len(bets), label))
        bets.extend(season_bets)
        running = round(running + sum(b["profit"] for b in season_bets), 2)
    bets.extend(b for b in live[1].get("bets", []) if b.get("status") == "pending")
    merged = {**live[1], "initial_bankroll": seasons[0][1].get("initial_bankroll", 0.0),
              "bets": bets}
    return merged, marks


# ── Starting the next season ──────────────────────────────────────────────────

def _backup(live_file: Path, tag: str) -> Path:
    """Timestamped backup beside the live file, per the portfolio-safety rule."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = live_file.parent / f"{live_file.stem}.backup.{ts}.{tag}.json"
    if live_file.exists():
        shutil.copy2(live_file, dest)
    else:
        dest.write_text("{}")
    return dest


def _fresh(existing: dict, season: str, initial_bankroll: float,
           overrides: dict | None) -> dict:
    """A new season's portfolio: same settings, empty book, reset bankroll."""
    settings = dict(existing.get("settings", {}))
    settings.update(overrides or {})
    return {
        "season":           season,
        "initial_bankroll": float(initial_bankroll),
        "bankroll":         float(initial_bankroll),
        "bets":             [],
        "settings":         settings,
    }


def set_opening_bankroll(
    main: float | None = None,
    mock_two: float | None = None,
    *,
    rolled_from: str | None = None,
) -> dict:
    """Set a line's opening bankroll while its book is still empty.

    Compounding a season means carrying the closing bankroll forward rather
    than restarting at a fixed stake, so the opening figure has to be settable
    after `reset_for_new_season` has run. Refuses once any bet exists, settled
    or pending, because moving the opening figure under a live book would
    silently rewrite every P&L and ROI number derived from it.
    """
    targets = [(name, amount) for name, amount in
               (("main", main), ("mock_two", mock_two)) if amount is not None]
    if not targets:
        raise ValueError("Give an opening bankroll for at least one line")
    for name, amount in targets:
        if amount <= 0:
            raise ValueError(f"{name} opening bankroll must be positive, got {amount}")

    loaders = {"main": (pf.load_portfolio, pf.save_portfolio, pf.PORTFOLIO_FILE),
               "mock_two": (pf.load_portfolio_two, pf.save_portfolio_two,
                            pf.MOCK2_PORTFOLIO_FILE)}

    # Check every target before writing any of them.
    state = {}
    for name, amount in targets:
        load, _, _ = loaders[name]
        p = load()
        if p.get("bets"):
            raise ArchiveError(
                f"Refusing to change the {name} opening bankroll: "
                f"{len(p['bets'])} bet(s) already on the book. "
                f"Every P&L figure is measured from the opening figure."
            )
        state[name] = p

    backups, updated = {}, {}
    for name, amount in targets:
        _, save, path = loaders[name]
        backups[name] = _backup(path, "preOpeningBankroll")
        p = state[name]
        p["initial_bankroll"] = float(amount)
        p["bankroll"] = float(amount)
        if rolled_from:
            p["rolled_from"] = rolled_from
        save(p)
        updated[name] = p

    return {"backups": backups, "portfolios": updated}


def reset_for_new_season(
    season: str,
    *,
    closing_season: str,
    initial_bankroll: float,
    main_settings: dict | None = None,
    two_settings: dict | None = None,
) -> dict:
    """Start `season` fresh on both lines, once `closing_season` is archived.

    Settings carry over from the outgoing season (so the Odds API key and every
    validated gate survive), with `main_settings` / `two_settings` applied on
    top. Returns the backup paths and the two fresh portfolios.
    """
    if not is_archived(closing_season):
        raise ArchiveError(
            f"Refusing to reset: {closing_season} is not archived. "
            f"Run archive_season({closing_season!r}) first."
        )

    main_before = pf.load_portfolio()
    two_before  = pf.load_portfolio_two()

    if main_before.get("season") == season:
        raise ArchiveError(
            f"Refusing to reset: the live portfolio is already on {season}. "
            f"Resetting again would discard bets placed since."
        )

    backups = {
        "main":     _backup(pf.PORTFOLIO_FILE, f"preSeason{season}"),
        "mock_two": _backup(pf.MOCK2_PORTFOLIO_FILE, f"preSeason{season}"),
    }

    main_fresh = _fresh(main_before, season, initial_bankroll, main_settings)
    two_fresh  = _fresh(two_before,  season, initial_bankroll, two_settings)
    pf.save_portfolio(main_fresh)
    pf.save_portfolio_two(two_fresh)

    return {"season": season, "backups": backups,
            "main": main_fresh, "mock_two": two_fresh}
