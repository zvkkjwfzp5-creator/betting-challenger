"""Daily NBA model numbers from current results.

History to June 2022 comes from the back-test archive; everything since is
read from ESPN's public scoreboard feed (one request per month) and kept in
`live/nba_results.csv`. Settings are the ones the back-test chose.

Information for a manual decision only. In the back-test this model was less
accurate than the opening line on spreads, totals and win probability.
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .data_us import load_sbr
from .gaussian import GaussParams, run_gauss_filter
from .mathx import ndtr
from .nba import line_probs

FEED = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}&limit=1000"
RESULTS = Path("live/nba_results.csv")
FIRST_LIVE_MONTH = "2022-10"
NAME_FIX = {"76ers": "Seventysixers", "Trail Blazers": "Trailblazers"}
TEAMS = {"Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers", "Grizzlies", "Hawks", "Heat", "Hornets", "Jazz",
         "Kings", "Knicks", "Lakers", "Magic", "Mavericks", "Nets", "Nuggets", "Pacers", "Pelicans", "Pistons",
         "Raptors", "Rockets", "Seventysixers", "Spurs", "Suns", "Thunder", "Timberwolves", "Trailblazers",
         "Warriors", "Wizards"}


def _get(month: str) -> dict:
    req = urllib.request.Request(FEED.format(month.replace("-", "")),
                                 headers={"User-Agent": "betting-challenger/0.1 (python)", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def parse_month(payload: dict) -> list[dict]:
    """Regular-season, play-in and playoff games between the 30 NBA teams."""
    rows = []
    for e in payload.get("events", []):
        if e["season"]["type"] not in (2, 3, 5):
            continue
        comp = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
        if set(comp) != {"home", "away"}:
            continue
        name = lambda c: NAME_FIX.get(c["team"]["name"], c["team"]["name"])
        h, a = name(comp["home"]), name(comp["away"])
        if h not in TEAMS or a not in TEAMS:
            continue
        final = bool(e["status"]["type"]["completed"])
        start = pd.Timestamp(e["date"])
        rows.append({"game_id": int(e["id"]), "season": int(e["season"]["year"]) - 1,
                     "date": start.tz_convert("America/New_York").date().isoformat(), "start_utc": start.isoformat(),
                     "home": h, "away": a,
                     "hg": int(comp["home"]["score"]) if final else None,
                     "ag": int(comp["away"]["score"]) if final else None, "final": final})
    return rows


def _months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def update_results(path: Path = RESULTS, today: str | None = None, fetch=_get) -> pd.DataFrame:
    today = today or date.today().isoformat()
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = old[old.final] if len(old) else old
    start = FIRST_LIVE_MONTH if done.empty else done.date.max()[:7]      # refetch the latest stored month
    rows = []
    for m in _months(start, (pd.Timestamp(today) + pd.Timedelta(days=2)).strftime("%Y-%m")):
        if m[5:] in ("07", "08", "09"):
            continue                                                     # no NBA games
        rows += parse_month(fetch(m))
    allr = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("game_id", keep="last").sort_values(["date", "game_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    allr[allr.final].to_csv(path, index=False)
    return allr.reset_index(drop=True)


def add_b2b(df: pd.DataFrame) -> pd.DataFrame:
    """Flag teams that played the previous calendar day."""
    long = pd.concat([df[["match_id", "date"]].assign(team=df.home), df[["match_id", "date"]].assign(team=df.away)])
    long = long.sort_values(["team", "date"])
    long["b2b"] = (long.groupby("team").date.diff().dt.days == 1).astype(float)
    b = long.set_index(["match_id", "team"]).b2b
    df["b2b_h"] = b.reindex(pd.MultiIndex.from_arrays([df.match_id, df.home])).to_numpy()
    df["b2b_a"] = b.reindex(pd.MultiIndex.from_arrays([df.match_id, df.away])).to_numpy()
    return df


def settings(report: str = "reports/nba.json") -> dict:
    s = json.loads(Path(report).read_text())["meta"]["settings_by_season"]
    last = s[max(s, key=int)]
    kv = dict(x.split("=") for x in last["filter"].split("|"))
    return {"params": GaussParams(tau=float(kv["tau"]), q_year=float(kv["q"]), season_jump=float(kv["jump"]),
                                  use_b2b=kv["b2b"] == "1"),
            "margin_sd": last["margin_sd"], "total_sd": last["total_sd"]}


def predict(day: str, cfg: dict, results: pd.DataFrame | None = None) -> pd.DataFrame:
    s = settings()
    cols = ["season", "date", "home", "away", "hg", "ag"]
    hist = load_sbr(cfg["nba"]["path"], "nba")[cols]
    live = update_results(today=day) if results is None else results
    played = live[live.final & (live.date < day)].copy()
    played["date"] = pd.to_datetime(played.date)
    fixtures = live[live.date == day].copy()
    if fixtures.empty:
        return fixtures
    fx = fixtures.assign(date=pd.Timestamp(day), hg=0, ag=0)   # placeholders: the day is predicted before any update
    df = pd.concat([hist, played[cols], fx[cols]], ignore_index=True)
    df[["hg", "ag"]] = df[["hg", "ag"]].astype(int)
    df["match_id"] = np.arange(len(df))
    df = add_b2b(df)
    p, _ = run_gauss_filter(df, s["params"])
    p, tail = p[-len(fx):], df.tail(len(fx))
    out = fixtures[["game_id", "start_utc", "away", "home"]].reset_index(drop=True)
    out["start_mt"] = pd.to_datetime(out.start_utc).dt.tz_convert("America/Edmonton").dt.strftime("%-I:%M %p")
    out["margin"], out["total"] = p[:, 0], p[:, 1]
    out["margin_se"], out["total_se"] = np.sqrt(p[:, 2]), np.sqrt(p[:, 3])
    z = np.random.default_rng(cfg["predictive"]["seed"]).standard_normal(4000)
    md = p[:, [0]] + out.margin_se.to_numpy()[:, None] * z[None, :]
    out["home_win"] = (1.0 - ndtr(-md / s["margin_sd"])).mean(axis=1)
    out["b2b"] = [("home " if h else "") + ("away" if a else "") for h, a in zip(tail.b2b_h, tail.b2b_a)]
    order = np.argsort(pd.to_datetime(out.start_utc).to_numpy(), kind="stable")
    out = out.iloc[order].reset_index(drop=True)
    out.attrs.update(results_to=str(played.date.max().date()), margin_sd=s["margin_sd"], total_sd=s["total_sd"])
    return out


def with_lines(r, home_spread: float, total_line: float, margin_sd: float, total_sd: float, z: np.ndarray) -> dict:
    """Model probability of the home side covering and of the over, given no push."""
    md = np.array([r.margin + r.margin_se * z])
    td = np.array([r.total + r.total_se * z])
    cw, cp = line_probs(md, margin_sd, np.array([-home_spread]))
    ow, op = line_probs(td, total_sd, np.array([total_line]))
    return {"cover": float((cw / (1 - cp)).mean()), "over": float((ow / (1 - op)).mean())}


def table(out: pd.DataFrame, lines: dict | None = None) -> str:
    if out.empty:
        return "No NBA games on that date."
    z = np.random.default_rng(1).standard_normal(4000)
    L = [f"NBA model numbers (results to {out.attrs['results_to']}).",
         "Information only: in the back-test this model was less accurate than the opening line and showed no edge.", "",
         "| Start (MT) | Game | Model home margin | 90% range | Model total | Home win % | Line (home spread / total) | "
         "Home cover % | Over % | Back-to-back |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        ln = cv = ov = "n/a"
        key = f"{r.away}@{r.home}"
        if lines and key in lines:
            sp, tl = lines[key]
            c = with_lines(r, sp, tl, out.attrs["margin_sd"], out.attrs["total_sd"], z)
            ln, cv, ov = f"{sp:+g} / {tl:g}", f"{c['cover']:.1%}", f"{c['over']:.1%}"
        show = lambda t: {"Seventysixers": "76ers", "Trailblazers": "Trail Blazers"}.get(t, t)
        L.append(f"| {r.start_mt} | {show(r.away)} @ {show(r.home)} | {r.margin:+.1f} | "
                 f"{r.margin - 1.645 * r.margin_se:+.1f} to {r.margin + 1.645 * r.margin_se:+.1f} | {r.total:.1f} | "
                 f"{r.home_win:.1%} | {ln} | {cv} | {ov} | {r.b2b or '-'} |")
    return "\n".join(L)
