import numpy as np
import pandas as pd
import pytest


def make_league(n_teams=12, seasons=4, seed=0, drift=0.0, tau=0.3):
    """Synthetic double round-robin league with known strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, tau, n_teams)
    dfn = rng.normal(0, tau, n_teams)
    rows, truth = [], []
    for s in range(seasons):
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        per_day = n_teams // 2
        for i, (h, a) in enumerate(fixtures):
            date = pd.Timestamp(2010 + s, 8, 1) + pd.Timedelta(days=7 * (i // per_day))
            lh = np.exp(0.1 + 0.3 + att[h] - dfn[a])
            la = np.exp(0.1 + att[a] - dfn[h])
            rows.append(("X", date, teams[h], teams[a], rng.poisson(lh), rng.poisson(la)))
            truth.append((lh, la))
        att = att + rng.normal(0, drift, n_teams)
        dfn = dfn + rng.normal(0, drift, n_teams)
    df = pd.DataFrame(rows, columns=["league", "date", "home", "away", "hg", "ag"])
    df[["lam_h_true", "lam_a_true"]] = truth
    df = df.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    df["season"] = df.date.dt.year - (df.date.dt.month < 7).astype(int)
    df["match_id"] = np.arange(len(df))
    return df, dict(zip(teams, att)), dict(zip(teams, dfn))


@pytest.fixture(scope="session")
def league():
    return make_league()
