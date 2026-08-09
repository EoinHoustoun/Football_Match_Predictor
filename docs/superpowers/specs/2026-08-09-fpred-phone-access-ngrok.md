# F_PRED on the phone: ngrok + login

Written 2026-08-09. **Nothing here is built.** Agreed route, to be implemented
in a later session.

## Goal

Open F_PRED from a phone on any network — mobile data, hotel Wi-Fi, the office —
and have it look and behave like an installed app.

## Where we are

Streamlit already serves on the LAN:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The Mac is `192.168.1.100`, so `http://192.168.1.100:8501` works on home Wi-Fi,
and Safari → Share → Add to Home Screen gives a home-screen icon that opens
full-screen with no browser chrome.

It fails everywhere else, because `192.168.x.x` is a private address that does
not route off the local network. That is the whole problem being solved.

## The blocker: there is no login

`app.py` has **no authentication of any kind**. The settings panel can edit the
Odds API key, every betting gate, and the auto-bet switch. An ngrok URL is
public by construction, and public URLs get scanned.

**Login ships before the tunnel does.** Not "we'll add it after", not "just to
test for five minutes".

## Build order

### 1. Login gate (~1 hour)

- Password in `.streamlit/secrets.toml`. Already gitignored — confirm before
  writing anything into it.
- Read via `st.secrets["app_password"]`.
- Gate at the very top of `main()`, before any render, storing the result in
  `st.session_state`. It has to cover the entire app: the settings panel is
  reachable from Main, Mock Two and Pre-Flight, so gating one tab is not enough.
- Compare with `hmac.compare_digest`, not `==`.
- Tests: wrong password renders nothing but the prompt; right password renders
  the app; session survives a rerun; a missing secret fails closed rather than
  open.

### 2. ngrok (~15 minutes)

- Free tier issues a **new random URL on every restart**, which breaks a
  home-screen icon. A reserved domain is a paid feature. Decide whether to pay
  or to accept re-adding the icon after each restart.
- Run it under launchd alongside the existing `com.eoinhoustoun.fpred.autobet`
  so it comes back after a reboot. Reuse the two lessons from that job:
  **pin the interpreter** (launchd does not inherit the login shell's PATH) and
  set **`ProcessType: Standard`** (background QoS throttles hard).
- Point it at `http://localhost:8501`.

### 3. Add to Home Screen

On the ngrok URL, not the LAN one.

### 4. Cold start (~2 hours, optional)

A cold load is **73 seconds** (peak 164 MB). Every visit after an idle period
pays it, which is the difference between a usable phone app and an unusable one.
Persist the fitted models to disk and load them instead of refitting; the same
change took the FPL app's board from 6.2s to 15ms.

## Constraints that do not go away

- **The Mac must be awake.** ngrok tunnels to this machine; a closed lid serves
  nothing. Set never-sleep on power.
- The portfolio and the hourly runner stay on the Mac. That is deliberate —
  hosting in the cloud would split live betting state across two places.

## Alternatives considered

| Option | Away access | Mac can sleep | Login needed | Effort |
|---|---|---|---|---|
| LAN only (today) | no | no | no | done |
| **ngrok + login** | **yes** | **no** | **yes** | **~1.5 h** |
| Tailscale | yes | no | no | ~15 min |
| Streamlit Cloud | yes | yes | yes | ~1 day |
| Paid host + disk | yes | yes | yes | ~1 day + $5/mo |

Tailscale is the safer answer — a private mesh between Eoin's own devices, no
public URL, no login strictly required. It was recommended and not chosen. Keep
it as the fallback if the login work stalls.

Streamlit Community Cloud is free and Mac-independent but needs Supabase for
portfolio persistence, has no scheduler to replace launchd, and its free tier
sleeps, so every visit pays the 73s cold start.

## Mobile layout, still unaddressed

Dense tables and Pre-Flight's five-column fixture rows will need horizontal
scrolling on a phone. Wrap heavy tables in `st.expander`, or split the portfolio
panels into `st.tabs` (Stats / Bets / Settings). Worth doing only after Eoin has
used it on the phone and said what actually irritates him, rather than guessing.

## Security checklist before the tunnel goes up

- [ ] Login gate merged, tested, covering every entry point
- [ ] `.streamlit/secrets.toml` confirmed gitignored and not staged
- [ ] Odds API key still absent from tracked files and git history
      (verified clean on 2026-08-09; re-check after any push)
- [ ] Auto-bet switch reachable only behind the login
