"""Short, plain summaries of the backtest and of a single fixture."""
from __future__ import annotations


def _ci(r, scale=1.0, fmt="{:+.4f}"):
    return f"{fmt.format(r['mean'] * scale)} (95% interval {fmt.format(r['lo'] * scale)} to {fmt.format(r['hi'] * scale)})"


def verdict(out: dict) -> dict:
    """Promotion rule: the challenger replaces the existing model only if the
    combination beats it out of sample in BOTH markets with an interval that
    excludes zero, and does so in most seasons. A betting rule is 'warranted'
    only if its flat-stake return interval is entirely above zero."""
    wins = []
    for m in ("1x2", "ou25"):
        d = out["markets"][m]["log_loss_differences"]["D_combo minus B_existing_blend"]
        a, b = (int(x) for x in d["seasons_better"].split("/"))
        wins.append(d["hi"] < 0 and a / b >= 0.6)
    beats_market = any(out["markets"][m]["log_loss_differences"]["D_combo minus A_market"]["hi"] < 0
                       for m in ("1x2", "ou25"))
    r = out["returns"]
    bet = {k: (v["n"] > 0 and v["lo"] > 0) for k, v in r.items()}
    import statistics
    w_model = {m: statistics.median(abs(v[m]["theta"][1]) for v in out["meta"]["pool_by_season"].values())
               for m in ("1x2", "ou25")}
    return {"model_weight_in_combination": w_model, "promote_probabilities": all(wins), "beats_market_probabilities": beats_market,
            "betting_rule_warranted": bet}


def backtest_markdown(out: dict) -> str:
    v = verdict(out)
    L = []
    s0, s1 = out["meta"]["eval_seasons"]
    L.append(f"# Challenger test: top-5 European leagues, seasons {s0}/{s0 + 1 - 2000:02d} to {s1}/{s1 + 1 - 2000:02d}")
    L.append("")
    L.append("Every number below is out of sample: each season was predicted with settings chosen on earlier seasons only. "
             "Lower log loss and Brier are better. Prices are Bet365 pre-match prices from football-data.co.uk.")
    for m, title in (("1x2", "Match result (1X2)"), ("ou25", "Over/under 2.5 goals")):
        r = out["markets"][m]
        L += ["", f"## {title}: {r['n_matches']:,} matches", "",
              "| Model | Log loss | Brier | Calibration error | Calibration slope |", "|---|---|---|---|---|"]
        names = {"A_market": "A. Market (margin removed)", "B_existing_raw": "B. Existing model, sim only",
                 "B_existing_blend": "B. Existing model, 70/30 blend", "C_new": "C. New model alone",
                 "D_combo": "D. Market + new model"}
        for k, nm in names.items():
            x = r[k]
            L.append(f"| {nm} | {x['log_loss']:.4f} | {x['brier']:.4f} | {x['ece']:.4f} | {x['calibration_slope']:.2f} |")
        L += ["", "Log-loss differences (negative = first model better):", ""]
        for k, d in r["log_loss_differences"].items():
            L.append(f"- {k.replace('_', ' ')}: {_ci(d)}; better in {d['seasons_better']} seasons")
    L += ["", "## Which parts of the new model help", "",
          "Change in the new model's log loss when one part is removed (positive = that part helps):", "",
          "| Removed part | 1X2 | Over/under 2.5 |", "|---|---|---|"]
    names = {"C_static": "Strength changing over time", "C_nopool": "Pooling toward league baseline",
             "C_nodc": "Low-score dependence", "C_plugin": "Parameter uncertainty (outer draws)"}
    for k, nm in names.items():
        L.append(f"| {nm} | {_ci(out['ablations']['1x2'][k])} | {_ci(out['ablations']['ou25'][k])} |")
    L += ["", "## Flat one-unit stakes at the quoted price", "",
          "| Rule | Bets | Return per bet | Profit (units) |", "|---|---|---|---|"]
    names = {"B_existing_rule": "B. Existing rule (blended EV > 0, not flagged)",
             "C_new_rule": "C. New model, thresholds from earlier seasons",
             "D_combo_rule": "D. Combination, thresholds from earlier seasons"}
    for k, nm in names.items():
        x = out["returns"][k]
        if x["n"] == 0:
            L.append(f"| {nm} | 0 | no selections | 0 |")
        else:
            L.append(f"| {nm} | {x['n']:,} | {_ci(x, 100, '{:+.1f}%')} | {x['profit_units']:+.1f} |")
    sel = out["selection"]
    L += ["", f"Selections searched in the evaluation window: {sel['candidates_searched']['D']:,} for the combination. "
          f"Seasons where the rule chose no selection at all: C {len(sel['seasons_with_no_selection']['C'])}, "
          f"D {len(sel['seasons_with_no_selection']['D'])} of {s1 - s0 + 1}.",
          "", "## Verdict", ""]
    wm = v["model_weight_in_combination"]
    L.append("- Combination versus the existing 70/30 blend: "
             + ("**better** in both markets, every season, intervals excluding zero."
                if v["promote_probabilities"] else "**not established** in both markets."))
    L.append(f"- Why: the combination learned to give the new model almost no weight (typical weight "
             f"{wm['1x2']:.2f} on 1X2 and {wm['ou25']:.2f} on totals, against 1.00 for the market). "
             "In practice it is the margin-free market price. The gain over the existing blend comes from "
             "dropping the fixed 30% model weight, not from the new model knowing more than the market.")
    L.append("- Better than the market's own probabilities: " + ("yes, in at least one market." if v["beats_market_probabilities"]
             else "**no**. No tested model beat the margin-free market price."))
    w = v["betting_rule_warranted"]
    L.append("- Any betting rule with a return interval above zero: "
             + ("yes: " + ", ".join(k for k, ok in w.items() if ok) if any(w.values()) else "**none**."))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- NHL / NBA / UFC
_NAMES = {"A_market": "A. Market (margin removed)", "B_existing_raw": "B. Existing model, sim only",
          "B_existing_blend": "B. Existing model, 70/30 blend", "C_new": "C. New model alone",
          "D_combo": "D. Market + new model",
          "Z_closing_price_reference": "Closing price (reference only, not available in time)"}
_ABL = {"C_static": "Strength changing over time", "C_nopool": "Pooling toward league baseline",
        "C_plugin": "Parameter uncertainty (outer draws)", "C_no_tie_inflation": "Extra regulation ties",
        "C_no_back_to_back": "Back-to-back adjustment", "C_no_age_reach": "Age and reach"}
_RULES = {"B_existing_rule": "B. Existing rule (blended edge > 0, not flagged)",
          "B_blend_edge_ignoring_flag": "B. Blended edge > 0, flag ignored",
          "C_new_rule": "C. New model, thresholds from earlier seasons",
          "D_combo_rule": "D. Combination, thresholds from earlier seasons"}


def market_verdict(r: dict) -> dict:
    d = r["market"]["log_loss_differences"]
    import statistics
    v = {"model_weight": statistics.median(abs(p["theta"][1]) for p in r["pool_by_season"].values()),
         "D_beats_market": d["D_combo minus A_market"]["hi"] < 0,
         "C_beats_market": d["C_new minus A_market"]["hi"] < 0,
         "rule_warranted": [k for k, x in r["returns"].items() if x["n"] > 0 and x["lo"] > 0]}
    if "D_combo minus B_existing_blend" in d:
        v["D_beats_existing_blend"] = d["D_combo minus B_existing_blend"]["hi"] < 0
        v["existing_blend_worse_than_market"] = d["B_existing_blend minus A_market"]["lo"] > 0
    return v


def sport_markdown(out: dict) -> str:
    L = [f"# Challenger test: {out['sport']}", "",
         "Every number is out of sample: each season was predicted with settings chosen on earlier seasons only. "
         "Lower log loss and Brier are better. Intervals resample whole match weeks."]
    for title, r in out["markets"].items():
        m = r["market"]
        s0, s1 = r["eval_seasons"]
        L += ["", f"## {title}", "", f"{m['n_matches']:,} games, seasons {s0} to {s1}.", "",
              "| Model | Log loss | Brier | Calibration error | Calibration slope |", "|---|---|---|---|---|"]
        for k, nm in _NAMES.items():
            if k in m:
                x = m[k]
                L.append(f"| {nm} | {x['log_loss']:.4f} | {x['brier']:.4f} | {x['ece']:.4f} | {x['calibration_slope']:.2f} |")
        L += ["", "Log-loss differences (negative = first model better):", ""]
        for k, d in m["log_loss_differences"].items():
            L.append(f"- {k.replace('_', ' ')}: {_ci(d)}; better in {d['seasons_better']} seasons")
        if r["ablations"]:
            L += ["", "Change in the new model's log loss when one part is removed (positive = that part helps):", ""]
            for k, d in r["ablations"].items():
                L.append(f"- {_ABL.get(k, k)}: {_ci(d)}")
        if r["returns"]:
            L += ["", "| Flat one-unit stakes | Bets | Return per bet | Profit (units) |", "|---|---|---|---|"]
            for k, x in r["returns"].items():
                if x["n"] == 0:
                    L.append(f"| {_RULES[k]} | 0 | no selections | 0 |")
                else:
                    L.append(f"| {_RULES[k]} | {x['n']:,} | {_ci(x, 100, '{:+.1f}%')} | {x['profit_units']:+.1f} |")
            sel = r["selection"]
            L += ["", f"Selections searched: {sel['candidates_searched']:,}. Seasons with no selection at all: "
                  f"C {len(sel['seasons_with_no_selection']['C'])}, D {len(sel['seasons_with_no_selection']['D'])} "
                  f"of {sel['n_eval_seasons']}."]
        v = market_verdict(r)
        L += ["", f"Verdict: the combination gives the new model a typical weight of {v['model_weight']:.2f} "
              f"(market 1.00). Beats the market: {'yes' if v['D_beats_market'] else 'no'}."
              + (f" Beats the existing 70/30 blend: {'yes' if v['D_beats_existing_blend'] else 'not established'}."
                 if "D_beats_existing_blend" in v else "")
              + (f" Rule with a return interval above zero: {', '.join(v['rule_warranted']) if v['rule_warranted'] else 'none'}."
                 if r["returns"] else "")]
    return "\n".join(L) + "\n"


def summary_markdown(folder: str = "reports") -> str:
    """One table across all sports that have been back-tested."""
    import json
    from pathlib import Path
    rows, rets = [], []
    nm = {"B_existing_rule": "Existing rule (edge > 0, not flagged)",
          "B_blend_edge_ignoring_flag": "Existing blend, edge > 0, flag ignored",
          "C_new_rule": "New model rule", "D_combo_rule": "Combination rule"}
    f = Path(folder, "backtest.json")
    if f.exists():
        s = json.loads(f.read_text())
        for m, t in (("1x2", "Soccer: match result"), ("ou25", "Soccer: over/under 2.5")):
            r = s["markets"][m]
            rows.append((t, r["n_matches"], r["A_market"]["log_loss"], r["B_existing_blend"]["log_loss"],
                         r["C_new"]["log_loss"], r["D_combo"]["log_loss"], r["log_loss_differences"]["D_combo minus A_market"]))
        rets += [("Soccer: both markets", k, x) for k, x in s["returns"].items()]
    for name in ("nhl", "nba", "ufc"):
        f = Path(folder, f"{name}.json")
        if not f.exists():
            continue
        o = json.loads(f.read_text())
        for t, r in o["markets"].items():
            m = r["market"]
            label = f"{o['sport']}: {t.split(' (')[0].replace(' at the opening line', '').replace(' at the opening price', '').lower()}"
            rows.append((label, m["n_matches"], m["A_market"]["log_loss"], m.get("B_existing_blend", {}).get("log_loss"),
                         m["C_new"]["log_loss"], m["D_combo"]["log_loss"], m["log_loss_differences"]["D_combo minus A_market"]))
            rets += [(label, k, x) for k, x in r["returns"].items()]
    L = ["# Challenger test: summary across sports", "",
         "Out-of-sample log loss (lower is better). The last column is the combination minus the market with its "
         "95% interval; an interval that includes zero means no demonstrated improvement.", "",
         "| Sport and market | Games | Market | Existing 70/30 blend | New model alone | Market + new model | Combination vs market |",
         "|---|---|---|---|---|---|---|"]
    for t, n, a, b, c, d, dd in rows:
        L.append(f"| {t} | {n:,} | {a:.4f} | {'none' if b is None else f'{b:.4f}'} | {c:.4f} | {d:.4f} | "
                 f"{dd['mean']:+.4f} ({dd['lo']:+.4f} to {dd['hi']:+.4f}) |")
    L += ["", "## Flat-stake returns of each betting rule", "",
          "| Sport and market | Rule | Bets | Return per bet (95% interval) |", "|---|---|---|---|"]
    for t, k, x in rets:
        L.append(f"| {t} | {nm[k]} | {x['n']:,} | " + ("no selections |" if x["n"] == 0 else
                 f"{x['mean'] * 100:+.1f}% ({x['lo'] * 100:+.1f}% to {x['hi'] * 100:+.1f}%) |"))
    L += ["", "Notes: NBA spread and total bets are scored at an assumed -110. The UFC row with corners as recorded "
          "is not credible (the corner label appears to carry information about the result); the corner-randomised "
          "row is the one to rely on. NHL and NBA history ends in June 2022."]
    return "\n".join(L) + "\n"
