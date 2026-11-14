import pandas as pd
import pytest

from challenger import live_nhl


def _game(gid, day, home, away, hs=None, as_=None, last=None, state="FUT", gtype=2):
    g = {"id": gid, "season": 20262027, "gameType": gtype, "startTimeUTC": f"{day}T00:00:00Z", "gameState": state,
         "homeTeam": {"commonName": {"default": home}}, "awayTeam": {"commonName": {"default": away}}}
    if hs is not None:
        g["homeTeam"]["score"], g["awayTeam"]["score"] = hs, as_
        g["gameOutcome"] = {"lastPeriodType": last}
    return g


def test_feed_parsing_recovers_regulation_scores_and_skips_preseason():
    week = {"gameWeek": [{"date": "2026-11-10", "games": [
        _game(1, "2026-11-10", "Bruins", "Flyers", 4, 2, "REG", "OFF"),
        _game(2, "2026-11-10", "Oilers", "Utah Hockey Club", 3, 2, "OT", "OFF"),
        _game(3, "2026-11-10", "Kings", "Ducks", 1, 2, "SO", "FINAL"),
        _game(4, "2026-11-10", "Jets", "Wild", state="FUT"),
        _game(5, "2026-11-10", "Stars", "Blues", 5, 0, "REG", "OFF", gtype=1)]}]}
    rows = {r["game_id"]: r for r in live_nhl.parse_week(week)}
    assert 5 not in rows                                         # pre-season is ignored
    assert (rows[1]["hg"], rows[1]["ag"]) == (4, 2)
    assert (rows[2]["hg"], rows[2]["ag"], rows[2]["fh"]) == (2, 2, 3)   # level after regulation
    assert (rows[3]["hg"], rows[3]["ag"]) == (1, 1)
    assert rows[2]["away"] == "Coyotes"                          # relocated club keeps its history
    assert rows[4]["final"] is False and rows[4]["hg"] is None


def test_update_only_stores_finished_games_and_refetches_recent_week(tmp_path):
    calls = []

    def fetch(day):
        calls.append(day)
        return {"nextStartDate": None, "gameWeek": [{"date": "2026-11-10", "games": [
            _game(1, "2026-11-10", "Bruins", "Flyers", 4, 2, "REG", "OFF"), _game(4, "2026-11-10", "Jets", "Wild")]}]}

    f = tmp_path / "r.csv"
    out = live_nhl.update_results(f, today="2026-11-10", fetch=fetch)
    assert len(out) == 2 and list(pd.read_csv(f).game_id) == [1]
    live_nhl.update_results(f, today="2026-11-12", fetch=fetch)
    assert calls[0] == live_nhl.FIRST_LIVE_DATE and calls[1] == "2026-11-03"   # a week back from the last result


def test_combination_stays_at_the_market_when_the_model_has_no_weight():
    c = live_nhl.combine_with_market(0.70, 2.10, 1.80, [1.0, 0.0, 0.0])
    assert c["combined_home"] == pytest.approx(c["market_home"])
    assert c["ev_home"] < 0 and c["ev_away"] < 0                 # only the bookmaker's margin remains
    d = live_nhl.combine_with_market(0.70, 2.10, 1.80, [1.0, 0.07, 0.0])
    assert c["market_home"] < d["combined_home"] < 0.70          # a small weight moves it only slightly
    assert d["combined_home"] - c["market_home"] < 0.02
