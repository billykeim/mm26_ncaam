"""Ingest Sports-Reference CBB coach history (rate-limited, HTML cache)."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    COACHES_CACHE,
    COACH_NAME_MAP_PATH,
    DATA_PROCESSED,
    SPORTS_REF_RAW,
)

SR_BASE = "https://www.sports-reference.com"


def _df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed/object columns so pyarrow can write parquet."""
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str)
    return out
INDEX_URL = f"{SR_BASE}/cbb/coaches/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns; ensure unique names (SR uses duplicate blanks)."""
    out = df.copy()
    new_cols: list[str] = []
    seen: set[str] = set()
    for j, c in enumerate(out.columns):
        if isinstance(c, tuple):
            parts = [str(x) for x in c if "Unnamed" not in str(x)]
            name = "_".join(parts).strip("_") or f"col_{j}"
        else:
            name = str(c).strip() or f"col_{j}"
        base = name
        k = 0
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen.add(name)
        new_cols.append(name)
    out.columns = new_cols
    return out


def fetch_letter_index(sess: requests.Session, letter: str) -> pd.DataFrame:
    """Load coaches table from ``{letter}-index.html``."""
    path = f"/cbb/coaches/{letter}-index.html"
    url = f"{SR_BASE}{path}"
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    dfs = pd.read_html(StringIO(r.text), attrs={"id": "NCAAM_coaches"}, flavor="lxml")
    if not dfs:
        return pd.DataFrame()
    return _flatten_columns(dfs[0])


def coach_slugs_from_letter_page(html: str) -> dict[str, str]:
    """
    Map coach display name -> detail slug path (e.g. /cbb/coaches/foo-1.html).

    Table rows link to coach pages from the Coach column.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="NCAAM_coaches")
    if not table:
        return {}
    mapping: dict[str, str] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        # Second cell is often Coach (first is Rk)
        link = cells[1].find("a", href=True) if len(cells) > 1 else None
        if not link:
            link = cells[0].find("a", href=True)
        if not link:
            continue
        name = link.get_text(strip=True)
        href = link["href"]
        if "/cbb/coaches/" in href and href.endswith(".html"):
            mapping[name] = href
    return mapping


def collect_coaches_index(sess: requests.Session) -> tuple[pd.DataFrame, dict[str, str]]:
    """Scrape a–z index pages; return combined table + name->href map."""
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    frames: list[pd.DataFrame] = []
    name_href: dict[str, str] = {}

    for letter in letters:
        path = f"/cbb/coaches/{letter}-index.html"
        url = f"{SR_BASE}{path}"
        r = sess.get(url, timeout=60)
        if r.status_code == 404:
            print(f"[coaches] skip {letter}: HTTP 404 (no index page)")
            continue
        if r.status_code != 200:
            print(f"[coaches] skip {letter}: HTTP {r.status_code}")
            continue
        time.sleep(3)
        dfs = pd.read_html(StringIO(r.text), attrs={"id": "NCAAM_coaches"}, flavor="lxml")
        if dfs:
            frames.append(_flatten_columns(dfs[0]))
        for k, v in coach_slugs_from_letter_page(r.text).items():
            name_href.setdefault(k, v)

    if not frames:
        return pd.DataFrame(), name_href
    df = pd.concat(frames, ignore_index=True)
    return df, name_href


def _ncaa_yrs_series(df: pd.DataFrame) -> pd.Series:
    col = [c for c in df.columns if "NCAA" in c and "Yrs" in c]
    if not col:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[col[0]], errors="coerce").fillna(0)


def slug_from_href(href: str) -> str:
    base = href.rsplit("/", 1)[-1]
    return base.replace(".html", "")


def fetch_coach_html(sess: requests.Session, href: str, slug: str) -> Path:
    COACHES_CACHE.mkdir(parents=True, exist_ok=True)
    out = COACHES_CACHE / f"{slug}.html"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = urljoin(SR_BASE, href)
    r = sess.get(url, timeout=90)
    r.raise_for_status()
    out.write_text(r.text, encoding="utf-8")
    return out


def parse_coach_seasons(html: str, slug: str) -> pd.DataFrame:
    """Parse career season table from cached coach page."""
    dfs = pd.read_html(StringIO(html), flavor="lxml")
    if not dfs:
        return pd.DataFrame()
    df = dfs[0]
    df["coach_slug"] = slug
    if "Notes" in df.columns:
        notes = df["Notes"].fillna("").astype(str)
        df["ncaa_tournament"] = notes.str.contains("NCAA Tournament", case=False, na=False)
        df["ncaa_ff"] = notes.str.contains(r"NCAA FF|Final Four", case=False, na=False, regex=True)
        df["ncaa_champ"] = notes.str.contains("NCAA Champ", case=False, na=False)
    else:
        df["ncaa_tournament"] = False
        df["ncaa_ff"] = False
        df["ncaa_champ"] = False
    return df


def season_str_to_end_year(season: object) -> int | None:
    """Map '2015-16' / '2015' to ending season year (e.g. 2016)."""
    if season is None or (isinstance(season, float) and pd.isna(season)):
        return None
    s = str(season).strip()
    if s in ("Career", "Overall", "nan", ""):
        return None
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return 2000 + y2 if y2 < 50 else 1900 + y2
    m2 = re.match(r"^(\d{4})$", s)
    if m2:
        return int(m2.group(1))
    return None


def build_cumulative_coach_store(rows: list[pd.DataFrame]) -> pd.DataFrame:
    """
    One row per (coach_slug, season_year) with cumulative stats through prior seasons.

    season_year is the ending year of the college season (CBBpy convention).
    """
    parts: list[pd.DataFrame] = []
    for cdf in rows:
        if cdf.empty or "Season" not in cdf.columns:
            continue
        slug = cdf["coach_slug"].iloc[0]
        cdf = cdf.copy()
        cdf["season_year"] = cdf["Season"].map(season_str_to_end_year)
        cdf = cdf[cdf["season_year"].notna()]
        cdf = cdf.sort_values("season_year")
        for col in ("ncaa_tournament", "ncaa_ff", "ncaa_champ"):
            if col not in cdf.columns:
                cdf[col] = False
        for src, dst in (
            ("ncaa_tournament", "coach_tourn_appearances"),
            ("ncaa_ff", "coach_final_four_count"),
            ("ncaa_champ", "coach_champ_count"),
        ):
            cdf[dst] = cdf[src].astype(int).cumsum().shift(1).fillna(0).astype(int)
        school_col = "School" if "School" in cdf.columns else None
        if school_col:
            cdf["school"] = cdf[school_col]
        else:
            cdf["school"] = ""
        parts.append(cdf[["coach_slug", "school", "season_year", "coach_tourn_appearances",
                          "coach_final_four_count", "coach_champ_count"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_top_coach_name_map(index_df: pd.DataFrame, name_href: dict[str, str], top_n: int = 100) -> dict:
    """Skeleton JSON for top-N coaches by NCAA tournament years + FF + Champ."""
    df = index_df.copy()
    yrs = _ncaa_yrs_series(df)
    ff_col = [c for c in df.columns if "FF" in c and "NCAA" in c]
    ch_col = [c for c in df.columns if "Champ" in c and "NCAA" in c]
    ff = pd.to_numeric(df[ff_col[0]], errors="coerce").fillna(0) if ff_col else 0
    ch = pd.to_numeric(df[ch_col[0]], errors="coerce").fillna(0) if ch_col else 0
    df["_score"] = yrs * 10 + ff * 5 + ch * 20
    df = df.sort_values("_score", ascending=False).head(top_n)
    out: dict[str, dict] = {}
    coach_col = "Coach" if "Coach" in df.columns else df.columns[1]
    for _, row in df.iterrows():
        name = str(row[coach_col])
        href = name_href.get(name)
        if not href:
            continue
        slug = slug_from_href(href)
        out[name] = {
            "canonical": name,
            "sports_ref_slug": slug,
            "aliases": [],
        }
    return out


def run_ingestion() -> tuple[int, int]:
    """
    Returns
    -------
    tuple[int, int]
        (coach_store rows, index rows)
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    COACHES_CACHE.mkdir(parents=True, exist_ok=True)
    SPORTS_REF_RAW.mkdir(parents=True, exist_ok=True)

    sess = _session()
    # Optional: single-page smoke — main index has no table; we use letter pages only.
    index_df, name_href = collect_coaches_index(sess)
    index_path = SPORTS_REF_RAW / "coaches_index.parquet"
    _df_for_parquet(index_df).to_parquet(index_path, index=False)
    print(f"[coaches] coaches_index: {len(index_df)} rows")

    ncaa_yrs = _ncaa_yrs_series(index_df)
    coach_col = "Coach" if "Coach" in index_df.columns else index_df.columns[1]
    to_fetch = index_df.loc[ncaa_yrs > 0, coach_col].astype(str).unique().tolist()

    season_frames: list[pd.DataFrame] = []
    for name in to_fetch:
        href = name_href.get(name)
        if not href:
            continue
        slug = slug_from_href(href)
        try:
            path = fetch_coach_html(sess, href, slug)
            time.sleep(3)
            html = path.read_text(encoding="utf-8", errors="replace")
            sdf = parse_coach_seasons(html, slug)
            if not sdf.empty:
                season_frames.append(sdf)
        except Exception as e:
            print(f"[coaches] failed {name}: {e}")
            time.sleep(3)

    store = build_cumulative_coach_store(season_frames)
    out_parquet = DATA_PROCESSED / "coach_store.parquet"
    if store.empty:
        store = pd.DataFrame(
            columns=[
                "coach_slug",
                "school",
                "season_year",
                "coach_tourn_appearances",
                "coach_final_four_count",
                "coach_champ_count",
            ]
        )
    _df_for_parquet(store).to_parquet(out_parquet, index=False)
    print(f"[coaches] coach_store: {len(store)} rows")

    cmap = build_top_coach_name_map(index_df, name_href, top_n=100)
    COACH_NAME_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COACH_NAME_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(cmap, f, indent=2, ensure_ascii=False)
    print(f"[coaches] coach_name_map.json entries: {len(cmap)}")

    return len(store), len(index_df)


def main() -> None:
    n_store, n_idx = run_ingestion()
    print(f"[coaches] done: coach_store={n_store}, index={n_idx}")


if __name__ == "__main__":
    main()
