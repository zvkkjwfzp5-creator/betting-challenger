"""The daily task's network blocks the score feeds, so the live commands must
work from the data stored in the repository."""
import tomllib
import urllib.request

import pytest

from challenger import live_nba, live_nhl

CFG = tomllib.load(open("config.toml", "rb"))


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


@pytest.mark.parametrize("mod", [live_nhl, live_nba])
def test_falls_back_to_stored_results_when_the_feed_is_blocked(mod, no_network):
    stored = mod.stored()
    day = stored[~stored.final.astype(bool)].date.min()          # first stored fixture date
    out = mod.predict(day, CFG)                                  # not asked to be offline: must fall back by itself
    assert len(out) > 0 and out.attrs["source"] == "stored copy"
    assert out.attrs["results_to"] < day                         # nothing from the day itself is used
    text = mod.table(out)
    assert "stored copy" in text and "Information only" in text


@pytest.mark.parametrize("mod", [live_nhl, live_nba])
def test_a_date_without_stored_fixtures_says_so_instead_of_failing(mod, no_network):
    out = mod.predict("2031-07-15", CFG, offline=True)
    assert out.empty and "No" in mod.table(out)


def test_history_files_are_in_the_repository():
    assert live_nhl.HISTORY.exists() and live_nba.HISTORY.exists()
