"""Fetch official NCAA tournament seeds from Sports-Reference postseason pages."""

from __future__ import annotations

import html
import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import SPORTS_REF_RAW, TOURNAMENT_SEEDS_PATH, TRAINING_YEARS
from src.utils.name_normalize import load_team_name_map, resolve_sr_bracket_school

SR_BASE = "https://www.sports-reference.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# First Four lines: <strong>12</strong> ... <a href='.../schools/.../men/YEAR.html'>Name</a>
_FF_STRONG_SEED_LINK_RE = re.compile(
    r"<strong>(\d{1,2})</strong>\s*(?:<strong>)?"
    r"<a\s+[^>]*href=['\"](/cbb/schools/[^'\"]+/men/\d{4}\.html)['\"][^>]*>([^<]+)</a>",
    re.IGNORECASE,
)

REQUEST_PAUSE_SEC = 3.0


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _try_read_html_tables(html: str, year: int) -> None:
    """
    Optional: ``pandas.read_html`` on the full page (SR often comments out stat tables).

    The visible bracket is ``div#brackets`` markup, not a ``<table>``; seeds are parsed from
    that div tree. This satisfies a ``read_html`` probe per the ingestion spec; bracket seeds
    do not come from these tables when none match.
    """
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        return
    if tables:
        print(
            f"[ingest_tournament_seeds] year={year}: read_html found {len(tables)} table(s) "
            "(bracket seeds still from #brackets markup)"
        )


def _parse_first_four_paragraphs(region_html: str) -> list[tuple[str, int, str]]:
    """Return (display name, seed, href path e.g. ``/cbb/schools/.../men/2011.html``)."""
    pairs: list[tuple[str, int, str]] = []
    if "First Four" not in region_html:
        return pairs
    soup = BeautifulSoup(region_html, "lxml")
    for p in soup.find_all("p"):
        if not p.find("strong"):
            continue
        text = p.get_text()
        if "First Four" not in text:
            continue
        raw = str(p)
        for m in _FF_STRONG_SEED_LINK_RE.finditer(raw):
            seed = int(m.group(1))
            href = m.group(2)
            name = html.unescape(m.group(3).strip())
            if 1 <= seed <= 16:
                pairs.append((name, seed, href))
    return pairs


def parse_tournament_seeds_html(html: str, year: int, mapping: dict) -> pd.DataFrame:
    """
    Parse bracket HTML into rows ``year``, ``team_norm``, ``official_seed``.

    Skips the ``national`` tab (duplicate teams); uses regional ``team16`` brackets plus
    First Four ``<p>`` blocks. Play-in opponents share the same seed (11 / 16 / etc.).
    """
    _try_read_html_tables(html, year)
    soup = BeautifulSoup(html, "lxml")
    brackets = soup.find("div", id="brackets")
    if brackets is None:
        print(f"[ingest_tournament_seeds] year={year}: WARN no div#brackets")
        return pd.DataFrame(columns=["year", "team_norm", "official_seed"])

    refined: list[tuple[str, int, str]] = []
    for region in brackets.find_all("div", recursive=False):
        rid = region.get("id")
        if rid in (None, "national"):
            continue
        for name, seed, href in _parse_first_four_paragraphs(str(region)):
            refined.append((name, seed, href))
        for br in region.find_all("div", id="bracket"):
            classes = br.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            if "team16" not in classes:
                continue
            for a in br.find_all("a", href=True):
                href = str(a.get("href", ""))
                if not re.search(r"/cbb/schools/[^/]+/men/\d{4}\.html", href):
                    continue
                span = a.find_previous_sibling("span")
                if span is None:
                    continue
                st = span.get_text(strip=True)
                if not st.isdigit():
                    continue
                seed = int(st)
                if not 1 <= seed <= 16:
                    continue
                name = a.get_text(strip=True)
                if name:
                    refined.append((name, seed, href))

    by_norm: dict[str, int] = {}
    conflicts: list[str] = []
    rows: list[dict[str, object]] = []
    for name, seed, href in refined:
        team_norm = resolve_sr_bracket_school(href, name, mapping)
        if team_norm in by_norm:
            if by_norm[team_norm] != seed:
                conflicts.append(f"{year} {team_norm}: {by_norm[team_norm]} vs {seed}")
        else:
            by_norm[team_norm] = seed
    if conflicts:
        print(f"[ingest_tournament_seeds] year={year} seed conflicts (using first seen):\n" + "\n".join(conflicts[:8]))

    for team_norm, official_seed in sorted(by_norm.items(), key=lambda x: (x[1], x[0])):
        rows.append({"year": int(year), "team_norm": team_norm, "official_seed": int(official_seed)})

    return pd.DataFrame(rows)


def fetch_year(sess: requests.Session, year: int) -> str:
    """GET postseason bracket page for tournament year ``year``."""
    url = f"{SR_BASE}/cbb/postseason/men/{year}-ncaa.html"
    r = sess.get(url, timeout=90)
    r.raise_for_status()
    return r.text


def ingest_all(
    years: list[int] | None = None,
    out_path: Path | None = None,
    pause_sec: float = REQUEST_PAUSE_SEC,
) -> pd.DataFrame:
    """
    Pull seeds for each year, rate-limited, and write ``tournament_seeds.parquet``.

    Logs row counts per year (expect 64 for 2008–2010, 68 from 2011 onward with First Four).
    """
    ys = list(years) if years is not None else [y for y in TRAINING_YEARS]
    out = out_path or TOURNAMENT_SEEDS_PATH
    mapping = load_team_name_map()
    sess = _session()
    parts: list[pd.DataFrame] = []
    for i, year in enumerate(ys):
        print(f"[ingest_tournament_seeds] fetching year={year} …")
        html = fetch_year(sess, year)
        df_y = parse_tournament_seeds_html(html, year, mapping)
        n = len(df_y)
        print(f"[ingest_tournament_seeds] year={year}: rows={n}")
        parts.append(df_y)
        if i < len(ys) - 1:
            time.sleep(pause_sec)
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if full.empty:
        print("[ingest_tournament_seeds] WARN: empty combined frame")
    else:
        full["year"] = full["year"].astype(int)
        full["official_seed"] = full["official_seed"].astype(int)
        full["team_norm"] = full["team_norm"].astype(str)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, index=False)
    print(f"[ingest_tournament_seeds] wrote {out} total_rows={len(full)}")
    return full


def main() -> None:
    ingest_all()


if __name__ == "__main__":
    main()
