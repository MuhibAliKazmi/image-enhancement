"""
Dataset acquisition for the low-light image enhancement study.

Searches Wikimedia Commons for openly licensed night-time / low-light Earth
observation imagery, downloads a 1920 px wide copy of each candidate, measures
its brightness, and keeps the darkest, most useful images (low mean luminance
but non-trivial dynamic range).

Only public-domain or Creative Commons licensed files are accepted. Every
selected image is documented in ``dataset_sources.csv`` with its Commons file
page, author, licence and licence URL so the provenance can be verified.

Usage
-----
    python src/fetch_dataset.py

The script is idempotent: images already present in ``data/original`` are kept
and the CSV is rewritten from the on-disk set.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode, urlparse, urlunparse

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "original"
CSV_PATH = PROJECT_ROOT / "dataset_sources.csv"

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "CENG376-ImageProcessing-Project/1.0 "
    "(university coursework; contact: student@example.edu)"
)

# Search themes -> Commons search strings. The theme is stored in the CSV so
# that results can later be grouped by scene type.
SEARCHES: dict[str, list[str]] = {
    "earth_at_night_satellite": [
        "Earth at night satellite city lights",
        "Black Marble NASA night lights",
        "night lights of Europe satellite",
        "night lights of Asia satellite",
        "night lights of Africa satellite",
        "night lights of North America satellite",
    ],
    "iss_city_lights": [
        "ISS night Earth city lights",
        "city at night from International Space Station",
        "Nile at night from space",
    ],
    "low_light_satellite": [
        "NASA Earth Observatory night image",
        "night time satellite image of Earth",
        "aurora from space ISS night",
    ],
    "dark_landscape": [
        "night landscape long exposure",
        "mountain landscape at night stars",
        "aurora borealis landscape night",
        "dark landscape twilight",
    ],
}

# Selection targets: how many images to keep per theme.
THEME_QUOTA = {
    "earth_at_night_satellite": 6,
    "iss_city_lights": 4,
    "low_light_satellite": 3,
    "dark_landscape": 3,
}

MAX_WIDTH = 1920
ALLOWED_LICENCE_PATTERNS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "pd-",
)
REJECT_TITLE_WORDS = (
    "logo", "map of", "diagram", "poster", "cover", "timeline",
    "chart", "graph", "icon", "flag", "webm", "svg", "gif",
)


@dataclass
class Candidate:
    title: str
    theme: str
    file_page: str
    download_url: str
    licence: str
    licence_url: str
    author: str
    credit: str
    description: str
    date: str
    width: int
    height: int
    mean_luma: float = 0.0
    std_luma: float = 0.0
    p99_luma: float = 0.0
    sha1: str = ""


def _strip_html(text: str) -> str:
    """Turn the small HTML snippets in Commons metadata into plain text."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _licence_ok(short_name: str) -> bool:
    name = (short_name or "").lower()
    if not name:
        return False
    return any(p in name for p in ALLOWED_LICENCE_PATTERNS)


def _slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"^File:", "", text)
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "image"


def _clean_url(url: str) -> str:
    """Drop Wikimedia tracking query parameters from a thumb URL."""
    parts = urlparse(url)
    return urlunparse((parts.scheme, parts.netloc, parts.path, "", "", ""))


def commons_search(query: str, limit: int = 25) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata|sha1",
        "iiurlwidth": str(MAX_WIDTH),
    }
    resp = requests.get(
        API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json().get("query", {}).get("pages", [])


def build_candidates(theme: str, search: str) -> list[Candidate]:
    out: list[Candidate] = []
    try:
        pages = commons_search(search)
    except Exception as exc:  # network hiccup -> skip this query
        print(f"    search failed ({exc})")
        return out

    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        title = page.get("title", "")
        if not info or not info.get("thumburl"):
            continue
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if any(w in title.lower() for w in REJECT_TITLE_WORDS):
            continue

        meta = info.get("extmetadata", {})
        licence = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
        if not _licence_ok(licence):
            continue

        width, height = info.get("width", 0), info.get("height", 0)
        if width < 1000 or height < 600:
            continue
        aspect = width / height if height else 0
        if not (1.2 <= aspect <= 2.1):
            continue

        out.append(
            Candidate(
                title=title,
                theme=theme,
                file_page=info.get("descriptionurl", ""),
                download_url=_clean_url(info["thumburl"]),
                licence=licence,
                licence_url=_strip_html(meta.get("LicenseUrl", {}).get("value", "")),
                author=_strip_html(meta.get("Artist", {}).get("value", ""))[:180],
                credit=_strip_html(meta.get("Credit", {}).get("value", ""))[:180],
                description=_strip_html(
                    meta.get("ImageDescription", {}).get("value", "")
                )[:400],
                date=_strip_html(meta.get("DateTimeOriginal", {}).get("value", ""))[:60],
                width=width,
                height=height,
            )
        )
    return out


def download_and_measure(cand: Candidate) -> Candidate | None:
    """Download one image into memory and compute brightness statistics."""
    try:
        resp = requests.get(
            cand.download_url,
            headers={"User-Agent": USER_AGENT},
            timeout=90,
        )
        resp.raise_for_status()
        buf = np.frombuffer(resp.content, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"    download failed: {exc}")
        return None

    if image is None or image.shape[0] < 400:
        return None

    # Normalise size: keep aspect ratio, cap the long edge at 1600 px.
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1600:
        scale = 1600 / long_edge
        image = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cand.mean_luma = float(gray.mean())
    cand.std_luma = float(gray.std())
    cand.p99_luma = float(np.percentile(gray, 99))
    cand.sha1 = hashlib.sha1(image.tobytes()).hexdigest()
    cand._encoded = image  # type: ignore[attr-defined]
    return cand


def is_useful_low_light(c: Candidate) -> bool:
    """Heuristic for 'underexposed but containing real structure'."""
    return (
        c.mean_luma <= 95.0
        and c.std_luma >= 18.0
        and c.p99_luma >= 90.0
    )


def select_candidates(cands: Iterable[Candidate]) -> list[Candidate]:
    """Pick the darkest useful candidates while respecting theme quotas."""
    by_theme: dict[str, list[Candidate]] = {t: [] for t in THEME_QUOTA}
    seen_sha: set[str] = set()
    seen_title: set[str] = set()

    for c in sorted(cands, key=lambda c: c.mean_luma):
        if c.theme not in by_theme:
            continue
        if c.sha1 in seen_sha or c.title in seen_title:
            continue
        if not is_useful_low_light(c):
            continue
        # Skip duplicate uploads of the same scene (e.g. "large", "cropped").
        base = re.sub(r"\((large|cropped|small|huge)\)", "", c.title.lower())
        base = re.sub(r"[^a-z0-9]", "", base)
        if base in seen_title:
            continue
        seen_sha.add(c.sha1)
        seen_title.add(base)
        by_theme[c.theme].append(c)

    selected: list[Candidate] = []
    for theme, quota in THEME_QUOTA.items():
        selected.extend(by_theme[theme][:quota])
    return selected


def save(cands: list[Candidate]) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, c in enumerate(cands, start=1):
        name = f"{idx:02d}_{c.theme}_{_slug(c.title, 45)}.jpg"
        path = DATA_DIR / name
        image = getattr(c, "_encoded", None)
        if image is None:
            continue
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            continue
        path.write_bytes(buf.tobytes())

        row = asdict(c)
        row.pop("_encoded", None)
        row["filename"] = name
        row["source"] = "Wikimedia Commons"
        row["source_url"] = c.file_page
        row["download_url_used"] = c.download_url
        row["stored_width"] = image.shape[1]
        row["stored_height"] = image.shape[0]
        rows.append(row)
        print(f"  saved {name}  (mean={c.mean_luma:.1f}, std={c.std_luma:.1f})")
    return rows


COLUMNS = [
    "filename", "theme", "original_title", "source", "source_url",
    "download_url_used", "licence", "licence_url", "author", "credit",
    "description", "date", "width", "height", "stored_width",
    "stored_height", "mean_luma", "std_luma", "p99_luma",
]


def write_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    print("Searching Wikimedia Commons for openly licensed low-light imagery ...")
    candidates: list[Candidate] = []
    for theme, queries in SEARCHES.items():
        for query in queries:
            print(f"  [{theme}] {query}")
            candidates.extend(build_candidates(theme, query))
            time.sleep(0.4)

    # De-duplicate before the (slower) downloads.
    unique: dict[str, Candidate] = {}
    for c in candidates:
        unique.setdefault(c.file_page, c)
    print(f"\n{len(unique)} unique candidate files found; downloading ...")

    measured: list[Candidate] = []
    for c in unique.values():
        result = download_and_measure(c)
        if result is not None:
            measured.append(result)
        time.sleep(0.25)

    print(f"{len(measured)} candidates downloaded; selecting ...")
    selected = select_candidates(measured)
    print(f"Selected {len(selected)} images:")
    rows = save(selected)
    write_csv(rows)
    print(f"\nWrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
