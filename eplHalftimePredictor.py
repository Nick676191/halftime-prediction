"""
EPL halftime -> full-time prediction: data preparation helpers.

Three pieces:
  1. load_football_data()      - multi-season EPL results with HALFTIME score + match stats
                                 (football-data.co.uk, free CSVs, 1993/94 onward)
  2. build_prematch_table()    - league table as it stood BEFORE each match
                                 (replaces the ESPN standings.csv, which is a single
                                 end-of-season snapshot -> the "only 38 GP" problem)
  3. espn_* helpers            - attach that table to your ESPN tester_df and derive
                                 first-half event counts from ESPN keyEvents / plays

Nothing here uses information from after halftime of the match being predicted,
except the target column itself.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. football-data.co.uk  (recommended primary training set)
# ---------------------------------------------------------------------------
# Columns you get per match (notes: https://www.football-data.co.uk/notes.txt):
#   HTHG/HTAG/HTR  halftime goals & result      FTHG/FTAG/FTR  full-time (target)
#   HS/AS shots, HST/AST shots on target, HC/AC corners, HF/AF fouls,
#   HY/AY yellows, HR/AR reds, Referee, plus pre-match betting odds (B365H/D/A etc.)
# NOTE: shots/corners/cards are FULL-TIME totals -> do not use them as halftime
# features (leakage). HT score, pre-match table, form and odds are safe.

def season_code(start_year: int) -> str:
    """2024 -> '2425' (football-data.co.uk URL format)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def load_football_data(start_years=range(2005, 2025), league="E0") -> pd.DataFrame:
    """Download and stack EPL seasons. E0 = Premier League."""
    frames = []
    for y in start_years:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code(y)}/{league}.csv"
        df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG"])
        df["Season"] = f"{y}-{y + 1}"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, format="mixed")
    return out.sort_values(["Date"]).reset_index(drop=True)


def football_data_features(fd: pd.DataFrame) -> pd.DataFrame:
    """Halftime-safe feature table from football-data.co.uk + pre-match table."""
    table = build_prematch_table(
        fd, date="Date", home="HomeTeam", away="AwayTeam",
        home_goals="FTHG", away_goals="FTAG", season="Season",
    )
    df = fd.merge(table, left_index=True, right_index=True)
    df["ht_goal_diff"] = df["HTHG"] - df["HTAG"]
    df["target"] = df["FTR"]  # H / D / A
    keep = ["Season", "Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "ht_goal_diff", "HTR",
            *table.columns, "target"]
    # pre-match odds are legitimate features (known before kickoff) if present
    keep += [c for c in ["B365H", "B365D", "B365A"] if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# 2. Pre-match league table (works for any results source)
# ---------------------------------------------------------------------------

def build_prematch_table(matches: pd.DataFrame, date: str, home: str, away: str,
                         home_goals: str, away_goals: str, season: str | None = None,
                         form_window: int = 5) -> pd.DataFrame:
    """
    For every match row, return each side's table state using only matches that
    kicked off on an EARLIER date in the same season:
        gp, pts, gd, gf, ga, position, ppg, form (points in last `form_window` games)
    Returned frame is indexed like `matches`, columns prefixed home_/away_.
    Matches on the same date don't see each other (conservative, no leakage).
    """
    m = matches[[date, home, away, home_goals, away_goals] + ([season] if season else [])].copy()
    m[date] = pd.to_datetime(m[date])
    if season is None:
        season = "_season"
        m[season] = 0

    # long format: one row per team per match
    h = pd.DataFrame({"idx": m.index, "season": m[season], "date": m[date], "team": m[home],
                      "gf": m[home_goals], "ga": m[away_goals], "side": "home"})
    a = pd.DataFrame({"idx": m.index, "season": m[season], "date": m[date], "team": m[away],
                      "gf": m[away_goals], "ga": m[home_goals], "side": "away"})
    long = pd.concat([h, a], ignore_index=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    long = long.sort_values(["season", "team", "date"])

    g = long.groupby(["season", "team"], sort=False)
    # shift(1) => stats strictly before this match
    for col in ["pts", "gf", "ga"]:
        long[f"pre_{col}"] = g[col].cumsum() - long[col]
    long["pre_gp"] = g.cumcount()
    long["pre_gd"] = long.pre_gf - long.pre_ga
    long["pre_ppg"] = (long.pre_pts / long.pre_gp).where(long.pre_gp > 0, 0.0)
    long["pre_form"] = g["pts"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).sum()).fillna(0)

    # league position before each match date: rank every team on its latest
    # pre-date totals (points, then GD, then GF, like the EPL table)
    positions = []
    for (sea, d), grp in long.groupby(["season", "date"]):
        before = long[(long.season == sea) & (long.date < d)]
        teams_in_season = long.loc[long.season == sea, "team"].unique()
        snap = (before.groupby("team")[["pts", "gf", "ga"]].sum()
                .reindex(teams_in_season, fill_value=0))
        snap["gd"] = snap.gf - snap.ga
        snap = snap.sort_values(["pts", "gd", "gf"], ascending=False)
        snap["position"] = np.arange(1, len(snap) + 1)
        positions.append(grp[["idx", "side", "team"]].assign(
            pre_position=grp.team.map(snap.position).values))
    pos = pd.concat(positions)
    long = long.merge(pos[["idx", "side", "pre_position"]], on=["idx", "side"])

    cols = ["pre_gp", "pre_pts", "pre_gd", "pre_gf", "pre_ga", "pre_ppg", "pre_form", "pre_position"]
    wide = long.pivot(index="idx", columns="side", values=cols)
    wide.columns = [f"{side}_{c.replace('pre_', '')}" for c, side in wide.columns]
    wide = wide.reindex(matches.index)
    wide["position_diff"] = wide.away_position - wide.home_position  # + means home higher
    wide["ppg_diff"] = wide.home_ppg - wide.away_ppg
    return wide


# ---------------------------------------------------------------------------
# 3. ESPN (Kaggle excel4soccer) helpers
# ---------------------------------------------------------------------------
# Adjust these if your column names differ.
ESPN = dict(
    event="eventId", date="date", league="leagueId",
    home="homeTeamId", away="awayTeamId",
    home_score="homeTeamScore", away_score="awayTeamScore",
    period="period", team="teamId", event_type_text="keyEventText",  # or 'text'/'typeText' in plays
)
EPL_LEAGUE_ID = 700  # ESPN's league id for eng.1; check leagues.csv to confirm


def espn_prematch_table(fixtures: pd.DataFrame, league_id=EPL_LEAGUE_ID, cols=ESPN) -> pd.DataFrame:
    """Pre-match table for every EPL fixture, keyed by eventId. Replaces standings.csv."""
    fx = fixtures[fixtures[cols["league"]] == league_id].copy()
    fx = fx.dropna(subset=[cols["home_score"], cols["away_score"]])  # finished matches only
    table = build_prematch_table(fx, date=cols["date"], home=cols["home"], away=cols["away"],
                                 home_goals=cols["home_score"], away_goals=cols["away_score"])
    return pd.concat([fx[[cols["event"]]], table], axis=1)


def attach_standings(tester_df: pd.DataFrame, fixtures: pd.DataFrame,
                     league_id=EPL_LEAGUE_ID, cols=ESPN) -> pd.DataFrame:
    """
    Join pre-match table onto tester_df.
      - if tester_df has one row per match (eventId unique): adds home_*/away_* columns
      - if one row per team per match (eventId + teamId): adds own_*/opp_* columns
    """
    table = espn_prematch_table(fixtures, league_id, cols)
    ev, team = cols["event"], cols["team"]
    if tester_df[ev].is_unique or team not in tester_df.columns:
        return tester_df.merge(table, on=ev, how="left", validate="many_to_one")

    # per-team rows: figure out which side each row is
    sides = fixtures[[ev, cols["home"], cols["away"]]]
    out = tester_df.merge(sides, on=ev, how="left").merge(table, on=ev, how="left")
    is_home = out[team] == out[cols["home"]]
    stat_cols = [c[5:] for c in table.columns if c.startswith("home_")]
    for s in stat_cols:
        out[f"own_{s}"] = np.where(is_home, out[f"home_{s}"], out[f"away_{s}"])
        out[f"opp_{s}"] = np.where(is_home, out[f"away_{s}"], out[f"home_{s}"])
    out["is_home"] = is_home.astype(int)
    drop = [c for c in table.columns if c.startswith(("home_", "away_"))]
    return out.drop(columns=drop + [cols["home"], cols["away"]])


# keyword -> feature name. Matched case-insensitively against the event text column.
FIRST_HALF_EVENTS = {
    "goal": "goals", "penalty": "penalties", "own goal": "own_goals",
    "yellow card": "yellows", "red card": "reds", "substitution": "subs",
    "offside": "offsides", "corner": "corners", "foul": "fouls",
    "shot on target|saved": "shots_on_target", "attempt|shot": "shots",
}


def espn_first_half_counts(events: pd.DataFrame, fixtures: pd.DataFrame,
                           keywords=FIRST_HALF_EVENTS, cols=ESPN) -> pd.DataFrame:
    """
    Count first-half (period == 1) events per match and side from ESPN keyEvents
    or plays data. Use plays/commentary for shots, corners, offsides and fouls;
    keyEvents only carries goals, cards and subs.
    Returns one row per eventId with ht_home_<x> / ht_away_<x> columns.
    """
    ev, team = cols["event"], cols["team"]
    e = events[events[cols["period"]] == 1].copy()
    text = e[cols["event_type_text"]].astype(str).str.lower()
    for pattern, name in keywords.items():
        e[name] = text.str.contains(pattern, regex=True).astype(int)
    counts = e.groupby([ev, team])[list(keywords.values())].sum().reset_index()

    fx = fixtures[[ev, cols["home"], cols["away"]]]
    counts = counts.merge(fx, on=ev, how="inner")
    counts["side"] = np.where(counts[team] == counts[cols["home"]], "home", "away")
    wide = counts.pivot_table(index=ev, columns="side", values=list(keywords.values()),
                              aggfunc="sum", fill_value=0)
    wide.columns = [f"ht_{side}_{stat}" for stat, side in wide.columns]
    # matches with no first-half events of a kind should read 0, not NaN
    return wide.reindex(fx[ev].unique(), fill_value=0).rename_axis(ev).reset_index()


if __name__ == "__main__":
    # quick self-test on a toy season
    toy = pd.DataFrame({
        "Date": ["2024-08-10", "2024-08-10", "2024-08-17", "2024-08-17", "2024-08-24", "2024-08-24"],
        "HomeTeam": ["A", "C", "B", "D", "A", "B"],
        "AwayTeam": ["B", "D", "A", "C", "C", "D"],
        "FTHG": [2, 0, 1, 1, 3, 0], "FTAG": [0, 0, 1, 2, 0, 1],
    })
    t = build_prematch_table(toy, "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
    print(pd.concat([toy, t[["home_gp", "home_pts", "home_position",
                             "away_gp", "away_pts", "away_position"]]], axis=1))