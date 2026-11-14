import numpy as np
import pandas as pd

from challenger import live_nba


def _event(eid, when, home, away, hs=None, as_=None, stype=2, done=True):
    comp = [{"homeAway": "home", "team": {"name": home}}, {"homeAway": "away", "team": {"name": away}}]
    if hs is not None:
        comp[0]["score"], comp[1]["score"] = str(hs), str(as_)
    return {"id": str(eid), "date": when, "season": {"year": 2027, "type": stype},
            "status": {"type": {"completed": done}}, "competitions": [{"competitors": comp}]}


def test_feed_parsing_keeps_real_games_and_uses_the_us_game_date():
    rows = {r["game_id"]: r for r in live_nba.parse_month({"events": [
        _event(1, "2026-11-14T00:00Z", "Pacers", "Knicks", 106, 104),
        _event(2, "2026-11-14T03:30Z", "Trail Blazers", "76ers", 99, 120),
        _event(3, "2026-10-05T00:00Z", "Lakers", "Suns", 90, 88, stype=1),          # pre-season
        _event(4, "2026-10-08T00:00Z", "Raptors", "Lions", 110, 80),                # exhibition opponent
        _event(5, "2026-11-15T01:00Z", "Heat", "Pistons", done=False)]})}
    assert set(rows) == {1, 2, 5}
    assert rows[1]["date"] == "2026-11-13" and rows[1]["season"] == 2026            # evening game, US date
    assert (rows[2]["home"], rows[2]["away"]) == ("Trailblazers", "Seventysixers")  # archive names
    assert (rows[1]["hg"], rows[1]["ag"]) == (106, 104)
    assert rows[5]["final"] is False and rows[5]["hg"] is None


def test_update_stores_only_finished_games_and_skips_the_off_season(tmp_path):
    asked = []

    def fetch(month):
        asked.append(month)
        return {"events": [_event(1, "2026-11-14T00:00Z", "Pacers", "Knicks", 106, 104)]} if month == "2026-11" else {"events": []}

    f = tmp_path / "nba.csv"
    out = live_nba.update_results(f, today="2026-11-14", fetch=fetch)
    assert len(out) == 1 and list(pd.read_csv(f).game_id) == [1]
    assert asked[0] == "2022-10" and not any(m[5:] in ("07", "08", "09") for m in asked)
    asked.clear()
    live_nba.update_results(f, today="2026-12-02", fetch=fetch)
    assert asked == ["2026-11", "2026-12"]                                          # only what is new


def test_back_to_back_flags():
    df = pd.DataFrame({"match_id": [0, 1, 2], "date": pd.to_datetime(["2026-11-10", "2026-11-11", "2026-11-13"]),
                       "home": ["A", "B", "A"], "away": ["B", "A", "C"]})
    out = live_nba.add_b2b(df)
    assert out.b2b_h.tolist() == [0.0, 1.0, 0.0] and out.b2b_a.tolist() == [0.0, 1.0, 0.0]


def test_line_probabilities_move_the_right_way():
    r = pd.Series({"margin": 5.0, "margin_se": 2.0, "total": 230.0, "total_se": 3.0})
    z = np.random.default_rng(0).standard_normal(4000)
    fav = live_nba.with_lines(r, -3.5, 228.5, 12.0, 18.0, z)
    assert fav["cover"] > 0.5 and fav["over"] > 0.5
    big = live_nba.with_lines(r, -9.5, 236.5, 12.0, 18.0, z)
    assert big["cover"] < 0.5 < fav["cover"] and big["over"] < 0.5
    even = live_nba.with_lines(r, -5.0, 230.0, 12.0, 18.0, z)                       # whole-number lines: push excluded
    assert abs(even["cover"] - 0.5) < 0.02 and abs(even["over"] - 0.5) < 0.02
