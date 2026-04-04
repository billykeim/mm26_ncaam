"""Ingest full NCAA tournament game results from Sports-Reference postseason HTML."""

from __future__ import annotations

import html as html_module
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    SPORTS_REF_RAW,
    TOURNAMENT_RESULTS_CACHE,
    TOURNAMENT_RESULTS_PATH,
    TRAINING_YEARS,
)
from src.utils.name_normalize import load_team_name_map, resolve_sr_bracket_school

SR_BASE = "https://www.sports-reference.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_PAUSE_SEC = 3.0

# Play-in / First Four: team line with seed, school link, then boxscore score link
_PLAYIN_TEAM_RE = re.compile(
    r"<strong>(\d{1,2})</strong>\s*(?:<strong>)?"
    r"<a\s+[^>]*href=['\"](/cbb/schools/[^'\"]+/men/\d{4}\.html)['\"][^>]*>([^<]+)</a>\s*"
    r"<a\s+[^>]*href=['\"](/cbb/boxscores/[^'\"]+\.html)['\"][^>]*>(\d+)</a>",
    re.IGNORECASE,
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_year_html(sess: requests.Session, year: int, cache_dir: Path) -> str:
    """Return page HTML, using ``{year}.html`` cache when present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{year}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    url = f"{SR_BASE}/cbb/postseason/men/{year}-ncaa.html"
    r = sess.get(url, timeout=90)
    r.raise_for_status()
    text = r.text
    cache_file.write_text(text, encoding="utf-8")
    return text


def _parse_playin_paragraph(p: Tag, year: int, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """First Four / Play-In games in a ``<p>`` block (round 0)."""
    raw = str(p)
    by_box: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in _PLAYIN_TEAM_RE.finditer(raw):
        seed = int(m.group(1))
        shref = m.group(2)
        name = html_module.unescape(m.group(3).strip())
        box = m.group(4)
        score = int(m.group(5))
        team_norm = resolve_sr_bracket_school(shref, name, mapping)
        by_box[box].append(
            {"seed": seed, "team_norm": team_norm, "score": score, "href": shref}
        )
    rows: list[dict[str, Any]] = []
    for box_href, teams in by_box.items():
        if len(teams) != 2:
            continue
        a, b = teams[0], teams[1]
        game_id = box_href.split("/")[-1].replace(".html", "")
        widx = 0 if a["score"] > b["score"] else 1
        rows.append(
            _orient_game_row(
                year=year,
                round_code=0,
                game_id=f"{year}_{game_id}",
                team_a=a,
                team_b=b,
                winner_idx=widx,
            )
        )
    return rows


def _parse_game_div(
    game_div: Tag, year: int, mapping: dict[str, Any], round_code: int
) -> dict[str, Any] | None:
    """One ``div`` game node inside a ``div.round`` (regional or national)."""
    team_divs: list[Tag] = []
    for child in game_div.children:
        if not isinstance(child, Tag) or child.name != "div":
            continue
        if child.find("a", href=re.compile(r"/cbb/schools/")):
            team_divs.append(child)
    if len(team_divs) != 2:
        return None
    teams: list[dict[str, Any]] = []
    for td in team_divs:
        span = td.find("span", recursive=False)
        seed: int | None = None
        if span is not None:
            st = span.get_text(strip=True)
            if st.isdigit():
                seed = int(st)
        sa = td.find("a", href=re.compile(r"/cbb/schools/[^\"']+/men/\d{4}\.html"))
        sc = td.find("a", href=re.compile(r"/cbb/boxscores/[^\"']+\.html"))
        if sa is None:
            continue
        shref = str(sa.get("href", ""))
        name = html_module.unescape(sa.get_text(strip=True))
        if sc is not None:
            try:
                score = int(sc.get_text(strip=True))
            except ValueError:
                score = 0
        else:
            score = 0
        team_norm = resolve_sr_bracket_school(shref, name, mapping)
        cls = td.get("class") or []
        if isinstance(cls, str):
            cls = [cls]
        winner = "winner" in cls
        teams.append(
            {
                "seed": seed,
                "team_norm": team_norm,
                "score": score,
                "href": shref,
                "winner": winner,
            }
        )
    if len(teams) != 2:
        return None
    # winner from class; fallback scores
    if teams[0]["winner"]:
        widx = 0
    elif teams[1]["winner"]:
        widx = 1
    elif teams[0]["score"] > teams[1]["score"]:
        widx = 0
    elif teams[1]["score"] > teams[0]["score"]:
        widx = 1
    else:
        widx = 0
    sc_a = team_divs[0].find("a", href=re.compile(r"/cbb/boxscores/"))
    sc_b = team_divs[1].find("a", href=re.compile(r"/cbb/boxscores/"))
    href0 = str(sc_a.get("href", "")) if sc_a else ""
    href1 = str(sc_b.get("href", "")) if sc_b else ""
    if href0 and href1 and href0 == href1:
        box_href = href0
    else:
        box_href = href0 or href1
    if box_href:
        game_id = box_href.split("/")[-1].replace(".html", "")
    else:
        game_id = f"forfeit_{teams[0]['team_norm']}_{teams[1]['team_norm']}".replace(
            " ", "_"
        )
    a, b = teams[0], teams[1]
    return _orient_game_row(
        year=year,
        round_code=round_code,
        game_id=f"{year}_{game_id}",
        team_a={
            "seed": a["seed"],
            "team_norm": a["team_norm"],
            "score": a["score"],
            "href": a["href"],
        },
        team_b={
            "seed": b["seed"],
            "team_norm": b["team_norm"],
            "score": b["score"],
            "href": b["href"],
        },
        winner_idx=widx,
    )


def _orient_game_row(
    year: int,
    round_code: int,
    game_id: str,
    team_a: dict[str, Any],
    team_b: dict[str, Any],
    winner_idx: int,
) -> dict[str, Any]:
    """Place lower seed (better team) in t1; tie-break by ``team_norm``."""
    ta, tb = team_a, team_b
    sa, sb = ta.get("seed"), tb.get("seed")
    swap = False
    if sa is not None and sb is not None:
        if sa > sb:
            swap = True
        elif sa == sb and ta["team_norm"] > tb["team_norm"]:
            swap = True
    elif sa is None and sb is not None:
        swap = True
    elif sa is not None and sb is None:
        swap = False
    else:
        if ta["team_norm"] > tb["team_norm"]:
            swap = True
    if swap:
        ta, tb = tb, ta
        winner_idx = 1 - winner_idx
    t1_won = winner_idx == 0
    return {
        "year": int(year),
        "round": int(round_code),
        "game_id": game_id,
        "t1_team_norm": str(ta["team_norm"]),
        "t2_team_norm": str(tb["team_norm"]),
        "t1_seed": int(ta["seed"]) if ta.get("seed") is not None else -1,
        "t2_seed": int(tb["seed"]) if tb.get("seed") is not None else -1,
        "t1_score": int(ta["score"]),
        "t2_score": int(tb["score"]),
        "result": int(1 if t1_won else 0),
    }


def _iter_round_games(round_div: Tag) -> list[Tag]:
    """
    Collect game ``div`` nodes inside a ``div.round``.

    Some seasons nest games; we climb from boxscore anchors. Forfeits (e.g. 2021 Oregon–VCU)
    have no boxscore links — include direct child game divs with two school rows.
    """
    seen: set[int] = set()
    out: list[Tag] = []

    def add(g: Tag) -> None:
        i = id(g)
        if i not in seen:
            seen.add(i)
            out.append(g)

    for sa in round_div.find_all("a", href=re.compile(r"/cbb/boxscores/")):
        anc = sa.parent
        found: Tag | None = None
        for _ in range(16):
            if anc is None or anc is round_div:
                break
            kids = [c for c in anc.children if isinstance(c, Tag) and c.name == "div"]
            team_rows = [c for c in kids if c.find("a", href=re.compile(r"/cbb/schools/"))]
            if len(team_rows) >= 2:
                found = anc
                break
            anc = anc.parent
        if found is not None:
            add(found)
    for child in round_div.children:
        if not isinstance(child, Tag) or child.name != "div":
            continue
        if child.find("a", href=re.compile(r"/cbb/boxscores/")):
            continue
        kids = [c for c in child.children if isinstance(c, Tag) and c.name == "div"]
        if len(kids) < 2:
            continue
        if kids[0].find("a", href=re.compile(r"/cbb/schools/")) and kids[1].find(
            "a", href=re.compile(r"/cbb/schools/")
        ):
            add(child)
    return out


def parse_tournament_html(html: str, year: int, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse all games from ``div#brackets``."""
    soup = BeautifulSoup(html, "lxml")
    brackets = soup.find("div", id="brackets")
    if brackets is None:
        return []
    out: list[dict[str, Any]] = []
    seen_game_id: set[str] = set()

    def add_row(row: dict[str, Any] | None) -> None:
        if row is None:
            return
        gid = row["game_id"]
        if gid in seen_game_id:
            return
        seen_game_id.add(gid)
        out.append(row)

    for region in brackets.find_all("div", recursive=False):
        rid = region.get("id")
        if rid in (None, "national"):
            continue
        for p in region.find_all("p"):
            if not p.find("strong"):
                continue
            txt = p.get_text()
            if "First Four" in txt or "Play-In" in txt:
                for row in _parse_playin_paragraph(p, year, mapping):
                    add_row(row)
        for br in region.find_all("div", id="bracket"):
            classes = br.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            if "team16" not in classes:
                continue
            round_divs = br.find_all("div", class_="round", recursive=False)
            for ri, rnd in enumerate(round_divs, start=1):
                for game_div in _iter_round_games(rnd):
                    add_row(_parse_game_div(game_div, year, mapping, round_code=ri))

    nat = brackets.find("div", id="national")
    if nat is not None:
        for br in nat.find_all("div", id="bracket"):
            classes = br.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            if not any(x in classes for x in ("team4", "team2", "team16")):
                continue
            round_divs = br.find_all("div", class_="round", recursive=False)
            for ri, rnd in enumerate(round_divs, start=1):
                round_code = 4 + ri
                for game_div in _iter_round_games(rnd):
                    add_row(_parse_game_div(game_div, year, mapping, round_code=round_code))
    return out


def _fix_numpy_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype(int)
    out["t1_seed"] = pd.to_numeric(out["t1_seed"], errors="coerce").astype(int)
    out["t2_seed"] = pd.to_numeric(out["t2_seed"], errors="coerce").astype(int)
    out["t1_score"] = pd.to_numeric(out["t1_score"], errors="coerce").astype(int)
    out["t2_score"] = pd.to_numeric(out["t2_score"], errors="coerce").astype(int)
    out["result"] = pd.to_numeric(out["result"], errors="coerce").astype(np.int8)
    return out


def ingest_all(
    years: list[int] | None = None,
    out_path: Path | None = None,
    cache_dir: Path | None = None,
    pause_sec: float = REQUEST_PAUSE_SEC,
) -> pd.DataFrame:
    """Fetch (or load cache), parse, write ``tournament_results.parquet``."""
    ys = list(years) if years is not None else list(TRAINING_YEARS)
    out_p = Path(out_path) if out_path is not None else TOURNAMENT_RESULTS_PATH
    cache = Path(cache_dir) if cache_dir is not None else TOURNAMENT_RESULTS_CACHE
    mapping = load_team_name_map()
    sess = _session()
    parts: list[pd.DataFrame] = []
    for i, year in enumerate(ys):
        print(f"[ingest_tournament_results] year={year} …")
        h = fetch_year_html(sess, year, cache)
        rows = parse_tournament_html(h, year, mapping)
        df_y = pd.DataFrame(rows)
        print(f"[ingest_tournament_results] year={year}: parsed_games={len(df_y)}")
        parts.append(df_y)
        if i < len(ys) - 1:
            time.sleep(pause_sec)
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if full.empty:
        print("[ingest_tournament_results] WARN empty")
    else:
        full = _fix_numpy_types(full)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_p, index=False)
    print(f"[ingest_tournament_results] wrote {out_p} rows={len(full)}")
    _print_verification(full)
    return full


def _print_verification(df: pd.DataFrame) -> None:
    """Log games per year and per (year, round)."""
    if df.empty:
        return
    print("\n[ingest_tournament_results] games per year:")
    print(df.groupby("year").size().to_string())
    print("\n[ingest_tournament_results] games per year × round (sample):")
    ct = df.groupby(["year", "round"]).size().reset_index(name="n")
    for y in sorted(df["year"].unique()):
        sub = ct[ct["year"] == y]
        print(f"  {int(y)}: {dict(zip(sub['round'], sub['n']))}")
    # Expectations: 68-team era = 67 games; 2008–10 include one opening-round game → 64 total.
    for y, n in df.groupby("year").size().items():
        yi = int(y)
        if yi in (2008, 2009, 2010):
            exp_lo, exp_hi = 63, 64
        else:
            exp_lo, exp_hi = 67, 67
        if not (exp_lo <= n <= exp_hi):
            print(
                f"[ingest_tournament_results] WARN year {yi}: got {n} games, "
                f"expected in [{exp_lo}, {exp_hi}]"
            )


def main() -> None:
    ingest_all()


if __name__ == "__main__":
    main()
