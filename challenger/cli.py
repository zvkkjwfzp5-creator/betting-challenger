"""Command line: `python -m challenger.cli backtest` and `... price`."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np


def _cfg(path):
    return tomllib.load(open(path, "rb"))


def cmd_backtest(a):
    from .backtest import run_backtest, write_reports
    from .report import backtest_markdown, sport_markdown
    if a.sport != "soccer":
        import importlib
        from .twoway import strip_private
        out = strip_private(importlib.import_module(f"challenger.{a.sport}").run(_cfg(a.config), verbose=not a.quiet))
        Path("reports").mkdir(exist_ok=True)
        Path(f"reports/{a.sport}.json").write_text(json.dumps(out, indent=1, default=str))
        md = sport_markdown(out)
        Path(f"reports/REPORT_{a.sport.upper()}.md").write_text(md)
        print(md)
        return
    out = run_backtest(_cfg(a.config), verbose=not a.quiet)
    write_reports(out)
    md = backtest_markdown(out)
    Path("reports/REPORT_SOCCER.md").write_text(md)
    print(md)


def cmd_report(a):
    from .report import backtest_markdown, summary_markdown
    Path("reports/REPORT_SOCCER.md").write_text(backtest_markdown(json.loads(Path("reports/backtest.json").read_text())))
    md = summary_markdown()
    Path("reports/SUMMARY.md").write_text(md)
    print(md)


def cmd_live(a):
    """Today's model numbers from current results (NHL or NBA)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    day = a.date or datetime.now(ZoneInfo("America/Edmonton")).date().isoformat()
    if a.sport == "nba":
        from . import live_nba
        lines = {k: tuple(v) for k, v in json.loads(a.lines).items()} if a.lines else None
        print(live_nba.table(live_nba.predict(day, _cfg(a.config)), lines))
        return
    from . import live_nhl
    prices = {k: tuple(v) for k, v in json.loads(a.prices).items()} if a.prices else None
    print(live_nhl.table(live_nhl.predict(day, _cfg(a.config)), prices))


def cmd_price(a):
    """Price one fixture. Output is conditional on the model's assumptions and
    never places or sizes a bet: the decision stays manual."""
    from .data import load_matches
    from .dynamic import FilterParams, run_filter
    from .market import DEVIG
    from .predictive import Scenario, rate_draws, standard_draws
    from .scoregrid import ScoreGrid

    cfg = _cfg(a.config)
    df = load_matches(cfg["data"]["path"], cfg["data"]["leagues"], cfg["data"].get("url"))
    bt = json.loads(Path("reports/backtest.json").read_text()) if Path("reports/backtest.json").exists() else None
    if bt:
        last = max(bt["meta"]["filter_setting_by_season"], key=int)
        params = FilterParams.from_key(bt["meta"]["filter_setting_by_season"][last])
        rho = float(bt["meta"]["rho_by_season"][last])
    else:
        params, rho = FilterParams(), -0.05
    lg = df[df.league == a.league]
    _, filters = run_filter(lg, params, return_filters=True)
    f = filters[a.league]
    for t in (a.home, a.away):
        if t not in f.prev_teams:
            raise SystemExit(f"{t!r} is not in the current {a.league} season. Known: {sorted(f.prev_teams)}")
    eta, S, _ = f.predict(a.home, a.away)
    grid = ScoreGrid(cfg["score"]["max_goals"])
    z = standard_draws(a.draws, cfg["predictive"]["seed"])
    scen = [Scenario("base")] + [Scenario(**s) for s in json.loads(a.scenarios or "[]")]

    def price(s: Scenario):
        lh, la = rate_draws([eta[0] + s.d_home], [eta[1] + s.d_away], [S[0, 0]], [S[1, 1]], [S[0, 1]], z)
        return grid.pmf(lh[0], la[0], rho)          # (draws, G, G)

    M = price(scen[0])
    lines = [f"{a.home} v {a.away} ({a.league}); data to {lg.date.max():%Y-%m-%d}; {a.draws} parameter draws.",
             "Conditional on the model's assumptions; line-ups and injuries are not in the data."]
    rows = []
    if a.odds_1x2:
        o = [float(x) for x in a.odds_1x2.split(",")]
        mk = DEVIG["multiplicative"](np.array([o]))[0]
        for j, nm in enumerate(["home", "draw", "away"]):
            mask = [grid.m_home, grid.m_draw, grid.m_away][j]
            rows.append((f"1X2 {nm}", o[j], lambda MM, mask=mask, oj=o[j]: grid.expected_return(MM, mask * oj - 1.0),
                         lambda MM, mask=mask: grid.p_event(MM, mask), mk[j]))
    if a.odds_ou:
        o = [float(x) for x in a.odds_ou.split(",")]
        for nm, oj in zip(["over", "under"], o):
            pay = grid.total_payoff(a.ou_line, oj, nm)
            rows.append((f"{nm} {a.ou_line}", oj, lambda MM, pay=pay: grid.expected_return(MM, pay),
                         lambda MM, pay=pay: grid.p_event(MM, (pay > 0).astype(float)), None))
    if a.odds_ah:
        o = [float(x) for x in a.odds_ah.split(",")]
        for nm, oj in zip(["home", "away"], o):
            pay = grid.ah_payoff(a.ah_line, oj, nm)
            rows.append((f"AH {nm} {a.ah_line:+g} (not validated)", oj, lambda MM, pay=pay: grid.expected_return(MM, pay),
                         lambda MM, pay=pay: grid.p_event(MM, (pay > 0).astype(float)), None))
    for name, odds, evf, pf, mk in rows:
        ev = evf(M)
        p = pf(M)
        txt = (f"- {name} @ {odds}: P(win) {p.mean():.1%} (fair {1 / p.mean():.2f}"
               + (f", market no-vig {mk:.1%}" if mk is not None else "") + f"); "
               f"expected return {ev.mean():+.1%}, 90% interval {np.quantile(ev, .05):+.1%} to {np.quantile(ev, .95):+.1%}; "
               f"P(expected return > 0) {np.mean(ev > 0):.0%}; "
               f"sampling error ±{ev.std(ddof=1) / np.sqrt(len(ev)):.2%}")
        # price sensitivity: expected return if the price shortens 5%; break-even price
        short = evf(M).mean() - 0.05 * odds * pf(M).mean()
        txt += f"; if the price shortens 5%: about {short:+.1%}"
        for s in scen[1:]:
            e2 = evf(price(s)).mean()
            tag = f"weight {s.weight}" if s.weight is not None else "sensitivity only, no weight"
            txt += f"; scenario '{s.name}' ({tag}): {e2:+.1%}"
        lines.append(txt)
    if bt:
        from .report import verdict
        v = verdict(bt)
        if not any(v["betting_rule_warranted"].values()):
            lines.append("Status: information only. Out of sample, no model here beat the market price and no "
                         "selection rule showed a positive return, so a positive number above is not evidence of an edge.")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(prog="challenger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--config", default="config.toml")
    b.add_argument("--quiet", action="store_true")
    b.add_argument("--sport", default="soccer", choices=["soccer", "nhl", "nba", "ufc"])
    b.set_defaults(fn=cmd_backtest)
    lv = sub.add_parser("live")
    lv.add_argument("--config", default="config.toml")
    lv.add_argument("--sport", default="nhl", choices=["nhl", "nba"])
    lv.add_argument("--lines", help='NBA: JSON of (home spread, total), e.g. {"Knicks@Pacers": [-3.5, 228.5]}')
    lv.add_argument("--date", help="YYYY-MM-DD in Mountain Time; default today")
    lv.add_argument("--prices", help='NHL: JSON of decimal prices, e.g. {"Rangers@Canucks": [1.80, 2.05]} (home, away)')
    lv.set_defaults(fn=cmd_live)
    r = sub.add_parser("report")
    r.set_defaults(fn=cmd_report)
    p = sub.add_parser("price")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--league", required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--odds-1x2", help="home,draw,away decimal prices")
    p.add_argument("--odds-ou", help="over,under decimal prices")
    p.add_argument("--ou-line", type=float, default=2.5)
    p.add_argument("--odds-ah", help="home,away decimal prices")
    p.add_argument("--ah-line", type=float, default=0.0)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--scenarios", help='JSON list, e.g. [{"name":"striker out","d_home":-0.12}]')
    p.set_defaults(fn=cmd_price)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
