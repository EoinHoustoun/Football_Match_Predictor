"""Phone notifications for F_PRED via ntfy (https://ntfy.sh), free and account-free.

Every activity-log event passes through `notify_event`; the ones worth a phone
buzz (a bet placed, a bet settled, a cancelled bet, a settings change, an
error) are sent to the ntfy topic in data/notify.json. Subscribe to that topic
in the ntfy app and they arrive as push notifications.

Rules:
  * Never raises and never blocks for long: a 4-second timeout, and any
    failure is swallowed, because a notification must never break a bet run.
  * Each event is sent once. The hourly runner and the app can both log the
    same settlement, so a fingerprint of every sent message is kept in
    data/notify_sent.json and repeats are dropped.
  * The same error is sent at most once every six hours.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "data" / "notify.json"
SENT = ROOT / "data" / "notify_sent.json"
ERROR_COOLDOWN_S = 6 * 3600
_MARKETS = {"H": "Home win", "D": "Draw", "A": "Away win",
            "over25": "Over 2.5", "under25": "Under 2.5"}
_LINES = {"main": "Main", "mt": "Mock Two"}


def load_config(create: bool = False) -> dict:
    """{"server", "topic", "enabled"}; NTFY_TOPIC in the environment wins."""
    cfg = {}
    try:
        cfg = json.loads(CONFIG.read_text())
    except Exception:
        cfg = {}
    if create and not cfg.get("topic"):
        cfg = {"server": "https://ntfy.sh", "enabled": True,
               # Unguessable: anyone who knows an ntfy topic can read it.
               "topic": "fpred-" + secrets.token_hex(8)}
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=1))
    if os.environ.get("NTFY_TOPIC"):
        cfg["topic"] = os.environ["NTFY_TOPIC"]
    cfg.setdefault("server", "https://ntfy.sh")
    cfg.setdefault("enabled", bool(cfg.get("topic")))
    return cfg


def _money(x) -> str:
    x = float(x or 0)
    s = f"£{abs(x):,.0f}"
    return ("−" + s) if x < 0 else ("+" + s if x > 0 else s)


def format_event(e: dict) -> dict | None:
    """Title, body, tags, priority and a dedupe key for one log event, or None."""
    t = e.get("type")
    line = _LINES.get(e.get("portfolio"), e.get("portfolio") or "")
    match = str(e.get("match", "")).replace(" vs ", " v ")
    mkt = _MARKETS.get(e.get("market"), e.get("market") or "")
    if t == "auto_bet_placed":
        ev = e.get("ev")
        return {"title": f"{line}: {mkt} backed",
                "body": f"{match} @{float(e.get('odds') or 0):.2f} · stake £{float(e.get('stake') or 0):,.0f}"
                        + (f" · EV {float(ev)*100:+.0f}%" if ev is not None else ""),
                "tags": "robot", "priority": 3,
                "key": f"placed|{e.get('portfolio')}|{match}|{e.get('market')}"}
    if t == "settled":
        won = e.get("result") == "won"
        return {"title": f"{line}: {'WON' if won else 'lost'} {_money(e.get('profit'))}",
                "body": f"{mkt} · {match}",
                "tags": "white_check_mark" if won else "x", "priority": 4 if won else 3,
                "key": f"settled|{e.get('portfolio')}|{match}|{e.get('market')}|{e.get('result')}"}
    if t == "bet_cancelled":
        return {"title": f"{line}: bet cancelled", "body": f"{mkt} · {match} · £{float(e.get('stake') or 0):,.0f}",
                "tags": "wastebasket", "priority": 3,
                "key": f"cancel|{e.get('portfolio')}|{match}|{e.get('market')}|{e.get('ts')}"}
    if t == "settings_changed":
        try:
            ch = json.loads(e.get("changes") or "{}")
        except Exception:
            ch = {}
        body = "; ".join(f"{k}: {v[0]} to {v[1]}" for k, v in ch.items())[:300]
        return {"title": f"{line}: live settings changed", "body": body or "see the change log",
                "tags": "gear", "priority": 4, "key": f"settings|{e.get('ts')}|{body}"}
    if t in ("error", "fatal"):
        msg = str(e.get("error") or e.get("message") or "")[:240]
        stage = e.get("stage") or t
        bucket = int(time.time() // ERROR_COOLDOWN_S)
        return {"title": f"F_PRED {'crashed' if t == 'fatal' else 'error'}: {stage}", "body": msg,
                "tags": "warning", "priority": 5 if t == "fatal" else 4,
                "key": f"err|{stage}|{msg[:80]}|{bucket}"}
    return None


def _load_sent() -> list:
    try:
        return json.loads(SENT.read_text())
    except Exception:
        return []


def _post(cfg: dict, msg: dict) -> bool:
    url = f"{cfg['server'].rstrip('/')}/{cfg['topic']}"
    req = urllib.request.Request(url, data=msg["body"].encode("utf-8"), method="POST", headers={
        "Title": msg["title"].encode("utf-8").decode("latin-1", "ignore"),
        "Tags": msg["tags"], "Priority": str(msg["priority"]),
        "Click": cfg.get("click", ""),
    })
    with urllib.request.urlopen(req, timeout=4) as r:
        return 200 <= r.status < 300


def notify_event(e: dict, sender=None) -> bool:
    """Send one event if it matters and has not been sent. Never raises."""
    try:
        # Never buzz the phone from the test suite; tests inject a sender.
        if sender is None and os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        cfg = load_config()
        if not cfg.get("enabled") or not cfg.get("topic"):
            return False
        msg = format_event(e)
        if not msg:
            return False
        fp = hashlib.sha1(msg["key"].encode()).hexdigest()[:16]
        sent = _load_sent()
        if fp in sent:
            return False
        ok = (sender or _post)(cfg, msg)
        if ok:
            sent = (sent + [fp])[-500:]
            SENT.parent.mkdir(parents=True, exist_ok=True)
            tmp = SENT.with_suffix(".tmp")
            tmp.write_text(json.dumps(sent))
            tmp.replace(SENT)
        return bool(ok)
    except Exception:
        return False


def send_test(sender=None) -> bool:
    """A hello message, to check the phone subscription works."""
    cfg = load_config(create=True)
    try:
        return bool((sender or _post)(cfg, {
            "title": "F_PRED notifications are on",
            "body": "You'll get a buzz when a bet is placed or settles, and if a run fails.",
            "tags": "soccer", "priority": 3, "key": f"test|{time.time()}"}))
    except Exception:
        return False
