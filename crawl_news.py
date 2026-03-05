from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time
import unittest
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dotenv import load_dotenv
from readability import Document
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from urllib3.util.retry import Retry

from database.opensearch import OpenSearchKB

load_dotenv()

# -----------------------------
# Config
# -----------------------------


SITEMAP_SOURCES: Dict[str, List[str]] = {
    "reuters": [
        "https://www.reuters.com/sitemap_index.xml",
        "https://www.reuters.com/sitemap.xml",
    ],
    "cnbc": [
        "https://www.cnbc.com/sitemapAll.xml",
        "https://www.cnbc.com/sitemap.xml",
    ],
    "sec": [
        "https://www.sec.gov/sitemap.xml",
    ],
    "ecb": [
        "https://www.ecb.europa.eu/sitemap.xml",
    ],
    "coindesk": [
        "https://www.coindesk.com/sitemap.xml",
    ],
    "cointelegraph": [
        "https://cointelegraph.com/sitemap.xml",
        "https://cointelegraph.com/sitemap/post.xml",
    ],
}

SOURCE_ALLOWED_DOMAINS: Dict[str, set] = {
    "reuters": {"reuters.com", "www.reuters.com"},
    "cnbc": {"cnbc.com", "www.cnbc.com"},
    "sec": {"sec.gov", "www.sec.gov"},
    "ecb": {"ecb.europa.eu", "www.ecb.europa.eu"},
    "coindesk": {"coindesk.com", "www.coindesk.com"},
    "cointelegraph": {"cointelegraph.com", "www.cointelegraph.com"},
}

DEFAULT_HEADERS = {
    "User-Agent": "FakeCryptoClaimDetector/1.0 (contact: your_email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

REQUEST_TIMEOUT = 20
MIN_SLEEP = 0.15
MAX_SLEEP = 0.45
MAX_SITEMAP_DEPTH = 3
MAX_SITEMAPS_PER_SOURCE = 80
MAX_SITEMAP_URLS_PER_SOURCE = 3000
SAVE_BATCH_SIZE = 5

# Extensions bị bỏ qua (không phải HTML)
NON_HTML_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".gz",
    ".rar",
    ".mp4",
    ".mp3",
    ".png",
    ".jpg",
    ".jpeg",
}

TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "guccounter",
    "guce_referrer",
    "guce_referrer_sig",
}

WALL_PATTERNS = re.compile(
    r"(enable javascript|cookie consent|agree to our|subscribe to|sign in to continue)",
    re.I,
)


# -----------------------------
# Utilities
# -----------------------------


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def is_non_html_extension(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return False
    return any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS)


def is_html_content_type(content_type: str) -> bool:
    ct = (content_type or "").lower()
    if not ct:
        return True
    return "text/html" in ct or "application/xhtml+xml" in ct


def is_possible_wall(html: str) -> bool:
    return bool(WALL_PATTERNS.search((html or "")[:50000]))


def decode_response_text(resp: requests.Response) -> str:
    """Decode response body robustly to avoid compressed/binary garbage text."""
    raw = resp.content or b""
    encoding_hdr = (resp.headers.get("Content-Encoding") or "").lower()
    data = raw

    # Fallback decompression for servers that send compressed bytes unexpectedly.
    try:
        if "gzip" in encoding_hdr and raw[:2] == b"\x1f\x8b":
            data = gzip.decompress(raw)
        elif "deflate" in encoding_hdr:
            try:
                data = zlib.decompress(raw)
            except zlib.error:
                data = zlib.decompress(raw, -zlib.MAX_WBITS)
        elif "br" in encoding_hdr:
            try:
                import brotli  # type: ignore

                data = brotli.decompress(raw)
            except Exception:
                try:
                    import brotlicffi as brotli  # type: ignore

                    data = brotli.decompress(raw)
                except Exception:
                    data = raw
    except Exception:
        data = raw

    try:
        if resp.apparent_encoding:
            resp.encoding = resp.apparent_encoding
    except Exception:
        pass

    encoding = resp.encoding or "utf-8"
    try:
        return data.decode(encoding, errors="replace")
    except Exception:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return resp.text


def drop_tracking(u: str) -> str:
    try:
        p = urlparse(u)
        netloc = p.netloc.lower()

        kept_params = []
        for key, value in parse_qsl(p.query, keep_blank_values=True):
            key_lower = key.lower()
            if key_lower in TRACKING_KEYS or key_lower.startswith("utm_"):
                continue
            kept_params.append((key, value))

        new_query = urlencode(kept_params, doseq=True)
        return urlunparse((p.scheme, netloc, p.path, p.params, new_query, ""))
    except Exception:
        return u


def pick_canonical_url(url: str, html: str) -> str:
    """Try to get canonical URL from <link rel=canonical> else return input url."""
    try:
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("link", attrs={"rel": re.compile("canonical", re.I)})
        if tag and tag.get("href"):
            return tag["href"].strip()
    except Exception:
        pass
    return url


def strip_nav_footer(text: str) -> str:
    """Heuristic cleanup: remove repeated boilerplate lines."""
    if not text:
        return ""

    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]

    cleaned: List[str] = []
    for l in lines:
        if len(l) <= 2:
            continue
        if re.search(
            r"(sign up|subscribe|cookie|privacy policy|terms of use|all rights reserved)",
            l,
            re.I,
        ):
            continue
        cleaned.append(l)

    out: List[str] = []
    prev: Optional[str] = None
    for l in cleaned:
        if prev is not None and l == prev:
            continue
        out.append(l)
        prev = l

    return "\n".join(out).strip()


def is_absolute_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


# -----------------------------
# HTTP client with retries
# -----------------------------


def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(DEFAULT_HEADERS)
    s.headers.setdefault("From", "your_email@example.com")
    return s


# -----------------------------
# Sitemap parsing
# -----------------------------


def is_allowed_source_domain(source: str, url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    if not host:
        return False
    allowed = SOURCE_ALLOWED_DOMAINS.get(source, set())
    return any(host == d or host.endswith("." + d) for d in allowed)


def parse_sitemap_document(xml_text: str) -> Dict[str, List[Any]]:
    """Parse sitemap XML into nested sitemap URLs and page URLs."""
    soup = BeautifulSoup(xml_text, "xml")
    nested_sitemaps: List[str] = []
    page_items: List[Dict[str, Any]] = []

    for sm in soup.find_all("sitemap"):
        loc_tag = sm.find("loc")
        if not loc_tag:
            continue
        loc = loc_tag.get_text(strip=True)
        if loc:
            nested_sitemaps.append(loc)

    for u in soup.find_all("url"):
        loc_tag = u.find("loc")
        if not loc_tag:
            continue
        loc = loc_tag.get_text(strip=True)
        if not loc:
            continue
        lastmod_tag = u.find("lastmod")
        lastmod = lastmod_tag.get_text(strip=True) if lastmod_tag else ""
        page_items.append(
            {
                "title": "",
                "link": loc,
                "published_raw": lastmod,
                "summary": "",
            }
        )

    return {"sitemaps": nested_sitemaps, "urls": page_items}


def discover_from_sitemaps(
    session: requests.Session,
    source: str,
    seeds: List[str],
    since_dt: Optional[datetime] = None,
    max_urls: int = MAX_SITEMAP_URLS_PER_SOURCE,
) -> List[Dict[str, Any]]:
    discovered: List[Dict[str, Any]] = []
    queue: List[tuple] = [(u, 0) for u in seeds]
    visited_sitemaps: set = set()

    while queue and len(visited_sitemaps) < MAX_SITEMAPS_PER_SOURCE:
        sitemap_url, depth = queue.pop(0)
        sitemap_key = drop_tracking(sitemap_url.strip())
        if not sitemap_key or sitemap_key in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_key)

        try:
            r = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                continue

            parsed = parse_sitemap_document(decode_response_text(r))
            nested = parsed["sitemaps"]
            urls = parsed["urls"]

            if depth < MAX_SITEMAP_DEPTH:
                for child in nested:
                    queue.append((child, depth + 1))

            seen_recent_in_sitemap = False
            for it in urls:
                link = (it.get("link") or "").strip()
                if not link:
                    continue
                if not is_allowed_source_domain(source, link):
                    continue
                if is_non_html_extension(link):
                    continue
                if since_dt is not None:
                    lm_dt = parse_datetime_utc_maybe(it.get("published_raw", ""))
                    if lm_dt is not None:
                        if lm_dt >= since_dt:
                            seen_recent_in_sitemap = True
                        elif seen_recent_in_sitemap:
                            # Sitemap thường được sort mới -> cũ, gặp bài cũ thì dừng sớm.
                            break
                        else:
                            continue
                discovered.append(it)
                if len(discovered) >= max_urls:
                    break
            if len(discovered) >= max_urls:
                break
        except Exception:
            continue

    return discovered


def parse_datetime_utc_maybe(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_datetime_maybe(s: str) -> Optional[str]:
    dt = parse_datetime_utc_maybe(s)
    return dt.isoformat() if dt is not None else None


# -----------------------------
# Full content extraction
# -----------------------------


def extract_from_jsonld(html: str, base_url: str) -> Dict[str, Any]:
    """Try JSON-LD extraction: headline, datePublished, articleBody, author, etc."""
    soup = BeautifulSoup(html, "lxml")
    out: Dict[str, Any] = {}

    scripts = soup.find_all(
        "script", attrs={"type": re.compile(r"application/ld\+json", re.I)}
    )
    candidates: List[Dict[str, Any]] = []

    for sc in scripts:
        raw = (sc.string or sc.get_text(strip=False) or "").strip()
        if not raw:
            continue
        data = safe_json_loads(raw)
        if data is None:
            continue

        if isinstance(data, dict):
            candidates.append(data)
        elif isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    candidates.append(x)

    # Flatten @graph
    flat: List[Dict[str, Any]] = []
    for c in candidates:
        if "@graph" in c and isinstance(c["@graph"], list):
            for g in c["@graph"]:
                if isinstance(g, dict):
                    flat.append(g)
        flat.append(c)

    # Choose first Article/NewsArticle
    article_obj: Optional[Dict[str, Any]] = None
    for obj in flat:
        t = obj.get("@type")
        if isinstance(t, list):
            tset = {str(x).lower() for x in t}
        else:
            tset = {str(t).lower()} if t else set()

        if any(x in tset for x in ["article", "newsarticle", "reportage"]):
            article_obj = obj
            break

    if not article_obj:
        return out

    out["title"] = article_obj.get("headline") or article_obj.get("name")
    out["published_at"] = parse_datetime_maybe(article_obj.get("datePublished", ""))
    out["modified_at"] = parse_datetime_maybe(article_obj.get("dateModified", ""))

    # Author
    author = article_obj.get("author")
    if isinstance(author, dict):
        out["author"] = author.get("name")
    elif isinstance(author, list):
        names: List[str] = []
        for a in author:
            if isinstance(a, dict) and a.get("name"):
                names.append(str(a["name"]))
            elif isinstance(a, str):
                names.append(a)
        out["author"] = ", ".join(names) if names else None
    elif isinstance(author, str):
        out["author"] = author

    # Main entity / canonical
    main_entity = article_obj.get("mainEntityOfPage")
    if isinstance(main_entity, dict) and main_entity.get("@id"):
        raw_id = str(main_entity["@id"]).strip()
        out["canonical_url"] = (
            raw_id if is_absolute_url(raw_id) else urljoin(base_url, raw_id)
        )
    elif isinstance(main_entity, str):
        raw_id = main_entity.strip()
        out["canonical_url"] = (
            raw_id if is_absolute_url(raw_id) else urljoin(base_url, raw_id)
        )

    # articleBody
    body = article_obj.get("articleBody")
    if isinstance(body, str) and body.strip():
        out["text"] = strip_nav_footer(body.strip())

    return out


def sanitize_html(html: str) -> str:
    """Strip null bytes và control characters khiến lxml bị lỗi."""
    # Xóa null bytes và các control chars ngoại trừ \t \n \r
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", html)


def extract_readability(html: str) -> Dict[str, Any]:
    html = sanitize_html(html)
    doc = Document(html)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "lxml")
    text = soup.get_text("\n", strip=True)

    return {
        "title": doc.short_title() or doc.title(),
        "text": strip_nav_footer(text),
    }


def extract_meta(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    def og(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
        return None

    def meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
        return None

    return {
        "title": og("og:title") or meta("title"),
        "published_at": parse_datetime_maybe(
            og("article:published_time") or meta("article:published_time") or ""
        ),
        "description": og("og:description") or meta("description"),
    }


def extract_full_article(url: str, html: str) -> Dict[str, Any]:
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

    canonical = pick_canonical_url(url, html)
    meta = extract_meta(html)
    jsonld = extract_from_jsonld(html, base)

    # Fallback readability if JSON-LD doesn't contain text
    readable: Dict[str, Any] = {}
    if not jsonld.get("text"):
        readable = extract_readability(html)

    title = jsonld.get("title") or meta.get("title") or readable.get("title")
    text = jsonld.get("text") or readable.get("text") or ""

    # If still empty, attempt naive paragraph join
    if not text:
        soup = BeautifulSoup(html, "lxml")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = strip_nav_footer("\n".join([p for p in paras if len(p) > 40]))

    published_at = jsonld.get("published_at") or meta.get("published_at")
    normalized_title = normalize_ws(title or "")
    if not normalized_title:
        normalized_title = "[no-title]"

    return {
        "url": url,
        "canonical_url": jsonld.get("canonical_url") or canonical,
        "title": normalized_title,
        "published_at": published_at,
        "author": jsonld.get("author"),
        "description": meta.get("description"),
        "text": text,
    }


# -----------------------------
# Storage (JSONL + dedup)
# -----------------------------


def load_seen(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(line)
    return seen


def append_seen(path: str, keys: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")


def save_articles_batch(source: str, out_dir: str, rows: List[Dict[str, Any]]) -> None:
    kb = OpenSearchKB(index_name=os.getenv("OP_KB_NAME"), embedding_dim=768)
    return kb.insert_many(rows)


# -----------------------------
# Main crawl
# -----------------------------


def crawl_source(
    source: str,
    out_dir: str,
    since_dt: Optional[datetime] = None,
    encoder: Optional[SentenceTransformer] = None,
) -> int:
    """Crawl one source. Return number of articles saved.

    Args:
        source:   key inside SITEMAP_SOURCES
        out_dir:  output directory
        since_dt: if given, only keep articles whose published_at >= since_dt
                  (UTC-aware). Articles with no timestamp are always kept.
    """
    sitemap_seeds = SITEMAP_SOURCES.get(source, [])
    if not sitemap_seeds:
        print(
            f"[!] No sitemap configured for source='{source}'. Edit SITEMAP_SOURCES first."
        )
        return 0

    session = build_session()

    seen_path = os.path.join(out_dir, f"seen_{source}.txt")
    seen = load_seen(seen_path)

    discovered: List[Dict[str, Any]] = []
    skip_stats: Dict[str, int] = {
        "seen": 0,
        "non_html_ext": 0,
        "http_error": 0,
        "non_html_ct": 0,
        "wall_or_consent": 0,
        "short_content": 0,
        "too_old": 0,
        "date_parse_warn": 0,
        "fetch_errors": 0,
    }

    # 1) Sitemap discovery
    discovered.extend(
        discover_from_sitemaps(
            session,
            source,
            sitemap_seeds,
            since_dt=since_dt,
            max_urls=MAX_SITEMAP_URLS_PER_SOURCE,
        )
    )

    # Dedup by link at discovery stage
    uniq: Dict[str, Dict[str, Any]] = {}
    for it in discovered:
        link = it.get("link")
        if link:
            uniq[drop_tracking(link.strip())] = it

    items = list(uniq.values())

    # 2) Fetch + extract
    pending_rows: List[Dict[str, Any]] = []
    pending_seen: List[str] = []
    saved_count = 0

    for it in items:
        url = it["link"].strip()
        normalized_url = drop_tracking(url)

        url_key = sha256_text(normalized_url)
        if url_key in seen:
            skip_stats["seen"] += 1
            continue
        if is_non_html_extension(url):
            skip_stats["non_html_ext"] += 1
            continue

        # polite sleep
        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                skip_stats["http_error"] += 1
                continue

            raw_content_type = resp.headers.get("Content-Type") or ""
            if not is_html_content_type(raw_content_type):
                skip_stats["non_html_ct"] += 1
                continue

            html = decode_response_text(resp)
            if is_possible_wall(html):
                skip_stats["wall_or_consent"] += 1
                continue

            art = extract_full_article(url, html)

            # Basic quality gate
            text = art.get("text") or ""
            if len(text) < 20:
                skip_stats["short_content"] += 1
                continue

            # ---- Time filter ----
            pub_iso = art.get("published_at") or parse_datetime_maybe(
                it.get("published_raw", "")
            )
            if since_dt is not None and pub_iso:
                try:
                    pub_dt = dtparser.parse(pub_iso)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    pub_dt = pub_dt.astimezone(timezone.utc)
                    if pub_dt < since_dt:
                        skip_stats["too_old"] += 1
                        continue
                except Exception:
                    skip_stats["date_parse_warn"] += 1

            canonical = drop_tracking(art.get("canonical_url") or url)
            content_hash = sha256_text(normalize_ws(text))
            record_id = sha256_text(canonical + "|" + content_hash)

            row = {
                "id": record_id,
                "source": source,
                "url": art.get("url"),
                "canonical_url": canonical,
                "title": art.get("title"),
                "published_at": pub_iso,
                "summary_from_sitemap": it.get("summary"),
                "author": art.get("author"),
                "description": art.get("description"),
                "text": text,
                "content_hash": content_hash,
                "fetched_at": now_utc_iso(),
                "embedding": None,
            }

            pending_rows.append(row)
            pending_seen.append(url_key)

            if len(pending_rows) >= SAVE_BATCH_SIZE:
                text_list = [it["text"] for it in pending_rows]
                embeddings = encoder.encode(text_list)
                for i, row in enumerate(pending_rows):
                    row["embedding"] = embeddings[i].tolist()
                save_articles_batch(source, out_dir, pending_rows)
                append_seen(seen_path, pending_seen)
                saved_count += len(pending_rows)
                pending_rows = []
                pending_seen = []

        except Exception:
            skip_stats["fetch_errors"] += 1

    if pending_rows:
        text_list = [it["text"] for it in pending_rows]
        embeddings = encoder.encode(text_list)
        for i, row in enumerate(pending_rows):
            row["embedding"] = embeddings[i].tolist()
        save_articles_batch(source, out_dir, pending_rows)
        append_seen(seen_path, pending_seen)
        saved_count += len(pending_rows)

    skipped_total = sum(skip_stats.values())
    print(
        f"[SOURCE] {source}: checked={len(items)} saved={saved_count} skipped={skipped_total}"
    )
    return saved_count


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Crawl tất cả nguồn tin tức được cấu hình trong SITEMAP_SOURCES."
    )
    ap.add_argument(
        "--since",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Chỉ lấy bài có published_at trong khoảng [now - SECONDS, now]. "
            "Ví dụ: --since 86400 để lấy bài trong 24h gần nhất. "
            "Nếu không truyền thì không lọc theo thời gian (cào toàn bộ)."
        ),
    )
    ap.add_argument("--out", default="data", help="Thư mục đầu ra (mặc định: data/)")
    ap.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args(argv)

    since_dt: Optional[datetime] = None
    if args.since is not None:
        since_dt = datetime.now(timezone.utc).replace(microsecond=0)
        from datetime import timedelta

        since_dt = since_dt - timedelta(seconds=args.since)
        print(f"[*] Lọc bài từ {since_dt.isoformat()} trở về sau ({args.since}s)")
    else:
        print("[*] Không lọc thời gian - cào toàn bộ bài hiện có trong sitemap")

    sources = [s for s, sitemaps in SITEMAP_SOURCES.items() if sitemaps]
    if not sources:
        print(
            "[!] Không có source nào được cấu hình trong SITEMAP_SOURCES. Hãy thêm sitemap URL trước."
        )
        return 1

    print(f"[*] Sẽ cào {len(sources)} source(s): {', '.join(sorted(sources))}")

    encoder = SentenceTransformer(args.embedding_model)
    total = 0
    for source in sorted(sources):
        total += crawl_source(source, args.out, since_dt, encoder)

    print(f"\n[*] Hoàn tất. Tổng cộng {total} bài mới đã lưu.")
    return 0


class TestExtraction(unittest.TestCase):
    def test_jsonld_articlebody_and_relative_canonical(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "NewsArticle",
          "headline": "Hello World",
          "datePublished": "2026-03-02T10:00:00Z",
          "mainEntityOfPage": {"@type": "WebPage", "@id": "/markets/hello"},
          "articleBody": "Line1\\nLine2"
        }
        </script>
        </head><body></body></html>
        """
        out = extract_from_jsonld(html, "https://example.com")
        self.assertEqual(out["title"], "Hello World")
        self.assertEqual(out["canonical_url"], "https://example.com/markets/hello")
        self.assertIn("Line1", out["text"])

    def test_jsonld_absolute_canonical(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Article",
          "headline": "A",
          "mainEntityOfPage": {"@id": "https://example.com/a"},
          "articleBody": "Body"
        }
        </script>
        </head></html>
        """
        out = extract_from_jsonld(html, "https://example.com")
        self.assertEqual(out["canonical_url"], "https://example.com/a")

    def test_parse_sitemap_document_urlset(self):
        xml = """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>https://example.com/a</loc>
            <lastmod>2026-03-02T10:00:00Z</lastmod>
          </url>
        </urlset>
        """
        parsed = parse_sitemap_document(xml)
        self.assertEqual(len(parsed["sitemaps"]), 0)
        self.assertEqual(len(parsed["urls"]), 1)
        self.assertEqual(parsed["urls"][0]["link"], "https://example.com/a")
        self.assertEqual(parsed["urls"][0]["published_raw"], "2026-03-02T10:00:00Z")

    def test_drop_tracking_removes_utm_and_fragment(self):
        u = "https://example.com/a?utm_source=x&id=123&utm_medium=y#frag"
        out = drop_tracking(u)
        self.assertEqual(out, "https://example.com/a?id=123")

    def test_drop_tracking_keeps_non_tracking_query(self):
        u = "https://example.com/a?symbol=BTCUSD&page=2"
        out = drop_tracking(u)
        self.assertEqual(out, u)

    def test_drop_tracking_lowercases_host_and_drops_fragment(self):
        u = "https://EXAMPLE.COM/A?Symbol=BTCUSD#frag"
        out = drop_tracking(u)
        self.assertEqual(out, "https://example.com/A?Symbol=BTCUSD")

    def test_parse_datetime_maybe_with_z_suffix(self):
        iso = parse_datetime_maybe("2026-03-02T10:00:00Z")
        self.assertEqual(iso, "2026-03-02T10:00:00+00:00")

    def test_non_html_extension_detection(self):
        self.assertTrue(is_non_html_extension("https://example.com/file.pdf"))
        self.assertFalse(is_non_html_extension("https://example.com/article.html"))

    def test_wall_pattern_detection(self):
        self.assertTrue(is_possible_wall("Please enable JavaScript to continue"))
        self.assertFalse(
            is_possible_wall("Normal article body about markets and rates.")
        )


if __name__ == "__main__":
    # If invoked as a script, run crawler.
    # For tests, use: python -m unittest finance_crawler.py
    raise SystemExit(main())
