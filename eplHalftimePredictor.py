"""
EPL halftime -> full-time prediction: data preparation helpers.

Pieces:
  1. load_football_data()      - multi-season EPL results with HALFTIME score + match stats
                                 + pre-match odds (football-data.co.uk, free CSVs)
  2. build_prematch_table()    - league table as it stood BEFORE each match
                                 (replaces the ESPN standings.csv, which is a single
                                 end-of-season snapshot -> the "only 38 GP" problem)
  3. espn_match_table()        - turns ESPN keyEvents + fixtures + teams into ONE ROW PER
                                 MATCH: first-half goals/cards/subs/penalties per side,
                                 full-time result, and the pre-match table
  4. join_espn_to_football_data() - adds the ESPN first-half columns to the
                                 football-data.co.uk rows (odds, HT score) for the same match

Nothing here uses information from after halftime of the match being predicted,
except the target column itself.
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _pick(df: pd.DataFrame, candidates, what: str) -> str:
    """Return the first column of `candidates` present in df, or raise a clear error."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Couldn't find a column for {what}. Tried {list(candidates)}. "
                   f"Available columns: {list(df.columns)}")


def season_from_date(d: pd.Series) -> pd.Series:
    """EPL season label from a date: Aug 2024 .. May 2025 -> '2024-2025'."""
    d = pd.to_datetime(d)
    start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
    return pd.Series([f"{y}-{y + 1}" for y in start], index=d.index)


# ---------------------------------------------------------------------------
# 1. football-data.co.uk  (recommended primary training set)
# ---------------------------------------------------------------------------
# Columns per match (notes: https://www.football-data.co.uk/notes.txt):
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
    # compare on calendar day so same-day kickoffs never see each other
    d = pd.to_datetime(m[date])
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    m[date] = d.dt.normalize()
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
    # subtracting this match => stats strictly before this match
    for col in ["pts", "gf", "ga"]:
        long[f"pre_{col}"] = g[col].cumsum() - long[col]
    long["pre_gp"] = g.cumcount()
    long["pre_gd"] = long.pre_gf - long.pre_ga
    long["pre_ppg"] = (long.pre_pts / long.pre_gp).where(long.pre_gp > 0, 0.0)
    long["pre_form"] = g["pts"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).sum()).fillna(0)

    # league position before each match date: rank every team on its pre-date
    # totals (points, then GD, then GF, like the EPL table)
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
    # before a team's first game the table order is meaningless -> mid-table
    n_teams = long.groupby("season")["team"].nunique()
    mid = (matches[season].map(n_teams) + 1) / 2 if season in matches.columns else (long.team.nunique() + 1) / 2
    for side in ("home", "away"):
        wide.loc[wide[f"{side}_gp"] == 0, f"{side}_position"] = mid if np.isscalar(mid) else mid[wide[f"{side}_gp"] == 0]
    wide["position_diff"] = wide.away_position - wide.home_position  # + means home higher
    wide["ppg_diff"] = wide.home_ppg - wide.away_ppg
    return wide


# ---------------------------------------------------------------------------
# 3. ESPN (Kaggle excel4soccer) -> one row per match
# ---------------------------------------------------------------------------

# Feature name -> (include regex, exclude regex), matched against keyEventName
# (lower-cased). Run  print(tester_df["keyEventName"].unique())  and adjust if
# your file uses different wording.
ESPN_EVENT_PATTERNS = {
    "goals":            (r"goal|penalty - scored", r"disallow|kick|no goal|own goal"),
    "own_goals":        (r"own goal", None),
    "pens_scored":      (r"penalty - scored", None),
    "pens_missed":      (r"penalty - (?:missed|saved)", None),
    "yellows":          (r"yellow", None),
    "reds":             (r"red card", None),
    "subs":             (r"substitution", None),
    "var":              (r"\bvar\b", None),
}

FIXTURE_COLS = dict(
    date=("date", "matchDate", "startDate", "dateTime"),
    home_score=("homeTeamScore", "homeScore"),
    away_score=("awayTeamScore", "awayScore"),
)


def _first_half_mask(ke: pd.DataFrame, name_col: str) -> pd.Series:
    """
    True for events before halftime. Uses a `period` column if the file has one;
    otherwise uses the position of the 'Halftime' event in keyEventOrder.
    """
    if "period" in ke.columns:
        return ke["period"] == 1
    order = _pick(ke, ["keyEventOrder", "sequence", "order"], "event order")
    is_ht = ke[name_col].astype(str).str.lower().str.contains("halftime|half time|half-time")
    ht_order = ke.loc[is_ht].groupby("eventId")[order].min()
    missing = set(ke["eventId"]) - set(ht_order.index)
    if missing:
        print(f"WARNING: {len(missing)} matches have no Halftime event; their "
              f"first-half counts will be 0. e.g. {list(missing)[:5]}")
    return ke[order] < ke["eventId"].map(ht_order).fillna(-np.inf)


def espn_first_half_counts(key_events: pd.DataFrame, fixtures: pd.DataFrame,
                           name_col: str = "keyEventName",
                           patterns=ESPN_EVENT_PATTERNS) -> pd.DataFrame:
    """
    One row per eventId with ht_home_<x> / ht_away_<x> counts for first-half events.
    `key_events` = keyEvents csv already merged with keyEventDescription (has keyEventName).
    Events with no teamId (kickoff, halftime, ...) are ignored.
    """
    ke = key_events[_first_half_mask(key_events, name_col)].dropna(subset=["teamId"]).copy()
    text = ke[name_col].astype(str).str.lower()
    for feat, (inc, exc) in patterns.items():
        hit = text.str.contains(inc, regex=True)
        if exc:
            hit &= ~text.str.contains(exc, regex=True)
        ke[feat] = hit.astype(int)

    fx = fixtures[["eventId", "homeTeamId", "awayTeamId"]].drop_duplicates("eventId")
    ke = ke.drop(columns=[c for c in ["homeTeamId", "awayTeamId"] if c in ke.columns])
    ke = ke.merge(fx, on="eventId", how="inner")
    ke["side"] = np.where(ke["teamId"] == ke["homeTeamId"], "home", "away")

    feats = list(patterns)
    wide = ke.pivot_table(index="eventId", columns="side", values=feats,
                          aggfunc="sum", fill_value=0)
    wide.columns = [f"ht_{side}_{feat}" for feat, side in wide.columns]
    wanted = [f"ht_{s}_{f}" for f in feats for s in ("home", "away")]
    wide = wide.reindex(columns=wanted, fill_value=0)
    # every match in fixtures gets a row, 0 where nothing happened
    wide = wide.reindex(fx["eventId"].unique(), fill_value=0).rename_axis("eventId")
    return wide.reset_index()


def espn_match_table(key_events: pd.DataFrame, fixtures: pd.DataFrame,
                     teams: pd.DataFrame | None = None,
                     name_col: str = "keyEventName") -> pd.DataFrame:
    """
    ONE ROW PER MATCH for every match that appears in `key_events`:
        eventId, date, season, home/away ids (+ names), FT score, target (H/D/A),
        ht_home_* / ht_away_* first-half counts, home_* / away_* pre-match table.

    `key_events` is your tester_df (or keyEvents merged with keyEventDescription).
    Using the event ids from key_events means we only keep EPL matches that were
    actually played, without needing a league id.
    """
    date_c = _pick(fixtures, FIXTURE_COLS["date"], "match date in fixtures")
    hs_c = _pick(fixtures, FIXTURE_COLS["home_score"], "home score in fixtures")
    as_c = _pick(fixtures, FIXTURE_COLS["away_score"], "away score in fixtures")

    fx = fixtures[fixtures["eventId"].isin(key_events["eventId"].unique())]
    fx = fx.drop_duplicates("eventId").dropna(subset=[hs_c, as_c]).copy()
    fx = fx[["eventId", date_c, "homeTeamId", "awayTeamId", hs_c, as_c]].rename(
        columns={date_c: "date", hs_c: "home_score", as_c: "away_score"})
    fx["date"] = pd.to_datetime(fx["date"], utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None)
    fx["season"] = season_from_date(fx["date"])
    fx = fx.sort_values("date").reset_index(drop=True)

    fx["target"] = np.select([fx.home_score > fx.away_score, fx.home_score == fx.away_score],
                             ["H", "D"], "A")

    table = build_prematch_table(fx, "date", "homeTeamId", "awayTeamId",
                                 "home_score", "away_score", season="season")
    out = pd.concat([fx, table], axis=1)
    out = out.merge(espn_first_half_counts(key_events, fx, name_col), on="eventId", how="left")
    out["ht_home_goals_total"] = out.ht_home_goals + out.ht_away_own_goals
    out["ht_away_goals_total"] = out.ht_away_goals + out.ht_home_own_goals

    if teams is not None:
        names = teams.drop_duplicates("teamId").set_index("teamId")
        name_c = _pick(names, ["name", "displayName", "teamName"], "team name in teams")
        out.insert(4, "home_team", out.homeTeamId.map(names[name_c]))
        out.insert(5, "away_team", out.awayTeamId.map(names[name_c]))
    return out


def attach_standings(tester_df: pd.DataFrame, match_table: pd.DataFrame) -> pd.DataFrame:
    """
    Put the pre-match table on each row of tester_df (any granularity: one row
    per key event, per team-match, or per match). Adds own_*/opp_* + is_home
    when tester_df has a teamId, otherwise home_*/away_*.
    """
    stat_cols = [c[5:] for c in match_table.columns
                 if c.startswith("home_") and c not in ("home_score", "home_team")]
    tbl = match_table[["eventId", "homeTeamId", "awayTeamId"]
                      + [f"home_{s}" for s in stat_cols] + [f"away_{s}" for s in stat_cols]]
    base = tester_df.drop(columns=[c for c in ["homeTeamId", "awayTeamId"] if c in tester_df.columns])
    out = base.merge(tbl, on="eventId", how="left", validate="many_to_one")
    if "teamId" not in out.columns:
        return out
    is_home = out["teamId"] == out["homeTeamId"]
    for s in stat_cols:
        out[f"own_{s}"] = np.where(is_home, out[f"home_{s}"], out[f"away_{s}"])
        out[f"opp_{s}"] = np.where(is_home, out[f"away_{s}"], out[f"home_{s}"])
    out["is_home"] = is_home.astype(int)
    return out.drop(columns=[f"{p}_{s}" for p in ("home", "away") for s in stat_cols])


# ---------------------------------------------------------------------------
# 4. ESPN <-> football-data.co.uk
# ---------------------------------------------------------------------------

# ESPN team name -> football-data.co.uk team name (only the ones that differ)
ESPN_TO_FD_NAMES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def join_espn_to_football_data(espn: pd.DataFrame, fd: pd.DataFrame,
                               name_map=ESPN_TO_FD_NAMES) -> pd.DataFrame:
    """
    Add ESPN first-half columns (ht_home_*/ht_away_*) to football-data rows.
    Joins on season + home team + away team (each pairing happens once a season),
    so kickoff-time/timezone differences in the dates don't matter.
    Prints a check of ESPN first-half goals against football-data HTHG/HTAG.
    """
    if "home_team" not in espn.columns:
        raise KeyError("espn table needs home_team/away_team names: call espn_match_table(..., teams=teams_df)")
    e = espn.copy()
    e["HomeTeam"] = e.home_team.replace(name_map)
    e["AwayTeam"] = e.away_team.replace(name_map)
    e = e.rename(columns={"season": "Season"})
    ht_cols = [c for c in e.columns if c.startswith("ht_")]
    merged = fd.merge(e[["Season", "HomeTeam", "AwayTeam", "eventId"] + ht_cols],
                      on=["Season", "HomeTeam", "AwayTeam"], how="inner", validate="one_to_one")

    unmatched = set(e.HomeTeam) - set(fd.loc[fd.Season.isin(e.Season), "HomeTeam"])
    if unmatched:
        print(f"WARNING: ESPN team names not found in football-data: {unmatched} "
              f"-> add them to ESPN_TO_FD_NAMES")
    ok = ((merged.ht_home_goals_total == merged.HTHG) & (merged.ht_away_goals_total == merged.HTAG))
    print(f"Joined {len(merged)} of {len(e)} ESPN matches. ESPN first-half goals match "
          f"football-data HT score in {ok.mean():.1%} of them.")
    return merged


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