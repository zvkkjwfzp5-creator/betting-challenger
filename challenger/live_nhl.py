"""Daily NHL model numbers from current results.

History to June 2022 comes from the back-test archive; everything since is
read from the NHL's public schedule feed and kept in `live/nhl_results.csv`
so each run only fetches what is new. Settings (pooling, evolution rate, tie
inflation, overtime step, combination weights) are the ones the back-test
chose; nothing is re-tuned here.

Output is information for a manual decision. The back-test found no edge over
the market, so a model number that differs from the market is not a reason to
bet by itself.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data_us import load_sbr
from .dynamic import FilterParams, run_filter
from .mathx import sigmoid
from .predictive import rate_draws, standard_draws
from .scoregrid import ScoreGrid

FEED = "https://api-web.nhle.com/v1/schedule/{}"
RESULTS = Path("live/nhl_results.csv")
FIRST_LIVE_DATE = "2022-10-01"
# feed name -> name used in the archive (the Utah club is the relocated Coyotes roster)
NAME_FIX = {"Utah Hockey Club": "Coyotes", "Mammoth": "Coyotes", "Hockey Club": "Coyotes"}


def _get(day: str) -> dict:
    # identify honestly as a script; the feed rejects requests with no client name
    req = urllib.request.Request(FEED.format(day), headers={"User-Agent": "betting-challenger/0.1 (python)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())


def parse_week(payload: dict) -> list[dict]:
    """Rows for every regular-season or playoff game in one feed response."""
    rows = []
    for d in payload.get("gameWeek", []):
        for g in d.get("games", []):
            if g.get("gameType") not in (2, 3):
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            final = g.get("gameState") in ("OFF", "FINAL") and "score" in h and "score" in a
            last = (g.get("gameOutcome") or {}).get("lastPeriodType")
            fh, fa = (h["score"], a["score"]) if final else (None, None)
            # regulation score: an OT/SO game was level after three periods
            hg, ag = (fh, fa) if last == "REG" else ((min(fh, fa),) * 2 if final else (None, None))
            name = lambda t: NAME_FIX.get(t["commonName"]["default"], t["commonName"]["default"])
            rows.append({"game_id": g["id"], "season": int(str(g["season"])[:4]), "date": d["date"],
                         "start_utc": g["startTimeUTC"], "home": name(h), "away": name(a),
                         "fh": fh, "fa": fa, "hg": hg, "ag": ag, "final": final})
    return rows


def update_results(path: Path = RESULTS, today: str | None = None, fetch=_get) -> pd.DataFrame:
    """Bring the stored results up to date and return them with this week's fixtures."""
    today = today or date.today().isoformat()
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = old[old.final] if len(old) else old
    start = FIRST_LIVE_DATE if done.empty else (pd.Timestamp(done.date.max()) - timedelta(days=7)).date().isoformat()
    rows, day = [], start
    while day and day <= (pd.Timestamp(today) + timedelta(days=7)).date().isoformat():
        payload = fetch(day)
        rows += parse_week(payload)
        nxt = payload.get("nextStartDate")
        day = nxt if nxt and nxt > day else None
    new = pd.DataFrame(rows)
    allr = pd.concat([old, new]).drop_duplicates("game_id", keep="last").sort_values(["date", "game_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    allr[allr.final].to_csv(path, index=False)
    return allr.reset_index(drop=True)


def settings(report: str = "reports/nhl.json") -> dict:
    o = json.loads(Path(report).read_text())
    s = o["meta"]["settings_by_season"]
    last = s[max(s, key=int)]
    kv = dict(x.split("=") for x in last["filter"].split("|"))
    market = next(iter(o["markets"].values()))
    pool = market["pool_by_season"]
    return {"filter": {"tau": float(kv["tau"]), "q_year": float(kv["q"]), "season_jump": float(kv["jump"]),
                       "promo_delta": float(kv["promo"])},
            "tie_inflation": last["tie_inflation"], "ot": last["ot"], "pool": pool[max(pool, key=int)]["theta"]}


def predict(day: str, cfg: dict, results: pd.DataFrame | None = None, draws: int = 2000) -> pd.DataFrame:
    """Model probabilities for the games on `day` (local date of the feed)."""
    s = settings()
    c = cfg["nhl"]
    hist = load_sbr(c["path"], "nhl")[["season", "date", "home", "away", "hg", "ag"]]
    live = update_results(today=day) if results is None else results
    played = live[live.final & (live.date < day)].copy()
    played["date"] = pd.to_datetime(played.date)
    fixtures = live[live.date == day].copy()
    if fixtures.empty:
        return fixtures
    fx = fixtures.assign(date=pd.Timestamp(day), hg=0, ag=0)        # placeholder scores, never learned from:
    df = pd.concat([hist, played[hist.columns], fx[hist.columns]], ignore_index=True)   # same-day games are
    df["league"] = "NHL"                                           # predicted before any same-day update
    df[["hg", "ag"]] = df[["hg", "ag"]].astype(int)
    df["match_id"] = np.arange(len(df))
    params = FilterParams(mu0=c["mu0"], h0=c["h0"], q_league_year=c["q_league_year"], **s["filter"])
    p = run_filter(df, params).to_numpy()[-len(fx):]
    z = standard_draws(draws, cfg["predictive"]["seed"])
    grid = ScoreGrid(12)
    lh, la = rate_draws(p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], z)
    p3 = grid.p_1x2(grid.pmf(lh, la, 0.0, s["tie_inflation"]))
    ph = p3[..., 0] + p3[..., 1] * sigmoid(s["ot"][0] + s["ot"][1] * (np.log(lh) - np.log(la)))
    out = fixtures[["game_id", "start_utc", "away", "home"]].reset_index(drop=True)
    order = np.argsort(pd.to_datetime(out.start_utc).to_numpy(), kind="stable")
    out, ph, p3, lh, la = out.iloc[order].reset_index(drop=True), ph[order], p3[order], lh[order], la[order]
    out["start_mt"] = pd.to_datetime(out.start_utc).dt.tz_convert("America/Edmonton").dt.strftime("%-I:%M %p")
    out["model_home"] = ph.mean(axis=1)
    out["lo90"], out["hi90"] = np.quantile(ph, 0.05, axis=1), np.quantile(ph, 0.95, axis=1)
    out["reg_draw"] = p3[..., 1].mean(axis=1)
    out["exp_goals"] = (lh + la).mean(axis=1)
    out["games_this_season"] = [int(((played.home == h) | (played.away == h)).where(played.season == played.season.max(), False).sum())
                                for h in out.home]
    out.attrs["pool"] = s["pool"]
    out.attrs["results_to"] = str(played.date.max().date())
    return out


def combine_with_market(model_home: float, home_price: float, away_price: float, theta: list[float]) -> dict:
    """Market no-vig probability and the back-test's market + model combination."""
    qh, qa = 1 / home_price, 1 / away_price
    mk = qh / (qh + qa)
    zh = theta[0] * np.log(mk) + theta[1] * np.log(model_home)
    za = theta[0] * np.log(1 - mk) + theta[1] * np.log(1 - model_home) + theta[2]
    comb = float(1 / (1 + np.exp(za - zh)))
    return {"market_home": mk, "combined_home": comb, "ev_home": comb * home_price - 1, "ev_away": (1 - comb) * away_price - 1}


def table(out: pd.DataFrame, prices: dict | None = None) -> str:
    if out.empty:
        return "No NHL games on that date."
    L = [f"NHL model numbers (results to {out.attrs['results_to']}). Home-win probability incl. overtime.",
         "Information only: the back-test found no edge over the market price.", "",
         "| Start (MT) | Game | Model home % | 90% range | Market home % | Market + model % | Reg. draw % | Exp. goals |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        key = f"{r.away}@{r.home}"
        show = lambda t: "Utah" if t == "Coyotes" else t
        mk = cb = "n/a"
        if prices and key in prices:
            c = combine_with_market(r.model_home, *prices[key], out.attrs["pool"])
            mk, cb = f"{c['market_home']:.1%}", f"{c['combined_home']:.1%}"
        L.append(f"| {r.start_mt} | {show(r.away)} @ {show(r.home)} | {r.model_home:.1%} | {r.lo90:.1%} to {r.hi90:.1%} | {mk} | {cb} | "
                 f"{r.reg_draw:.1%} | {r.exp_goals:.1f} |")
    return "\n".join(L)
