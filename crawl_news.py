from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from readability import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Config
# -----------------------------


RSS_FEEDS: Dict[str, List[str]] = {
    "reuters": [
        # Reuters feeds công khai bị lỗi 404, tạm thời bỏ qua
    ],
    "yahoo": [
        # Yahoo Finance có RSS tin tài chính tổng hợp:
        "https://www.yahoo.com/news/rss/finance",  # Yahoo Finance news RSS feed (tài chính) :contentReference[oaicite:1]{index=1}
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ],
    "cnbc": [
        # CNBC RSS (có thể dùng các URL RSS feed đã tồn tại):
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC top news RSS (thường dùng) :contentReference[oaicite:2]{index=2}
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",  # CNBC Business :contentReference[oaicite:3]{index=3}
    ],
    "sec": [
        "https://www.sec.gov/rss/news/press.xml",
    ],
    "fed": [
        "https://www.federalreserve.gov/feeds/press_all.xml",
    ],
    "ecb": [
        "https://www.ecb.europa.eu/rss/press.html",
    ],
    "ap": [
        # AP (Associated Press) có rất nhiều RSS feed theo chủ đề:
        "https://apnews.com/hub/business?outputType=rss",  # AP business news RSS
    ],
}

DEFAULT_HEADERS = {
    "User-Agent": "FakeCryptoClaimDetector/1.0 (contact: your_email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

REQUEST_TIMEOUT = 20
MIN_SLEEP = 0.8
MAX_SLEEP = 2.2

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


def drop_tracking(u: str) -> str:
    try:
        p = urlparse(u)
        if not p.query and not p.fragment:
            return u

        kept_params = []
        for key, value in parse_qsl(p.query, keep_blank_values=True):
            key_lower = key.lower()
            if key_lower in TRACKING_KEYS or key_lower.startswith("utm_"):
                continue
            kept_params.append((key, value))

        new_query = urlencode(kept_params, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, ""))
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
# RSS parsing
# -----------------------------


def parse_rss_items(xml_text: str) -> List[Dict[str, Any]]:
    """Parse RSS/Atom minimally without extra libs."""
    soup = BeautifulSoup(xml_text, "xml")

    items: List[Dict[str, Any]] = []

    # RSS <item>
    for it in soup.find_all("item"):
        link = it.find("link").get_text(strip=True) if it.find("link") else ""
        title = it.find("title").get_text(strip=True) if it.find("title") else ""
        pub = ""
        if it.find("pubDate"):
            pub = it.find("pubDate").get_text(strip=True)
        elif it.find("date"):
            pub = it.find("date").get_text(strip=True)

        desc = (
            it.find("description").get_text(" ", strip=True)
            if it.find("description")
            else ""
        )

        if link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published_raw": pub,
                    "summary": normalize_ws(
                        BeautifulSoup(desc, "lxml").get_text(" ", strip=True)
                    )
                    if desc
                    else "",
                }
            )

    # Atom <entry>
    if not items:
        for ent in soup.find_all("entry"):
            title = ent.find("title").get_text(strip=True) if ent.find("title") else ""

            link = ""
            link_tags = ent.find_all("link")
            # ưu tiên rel="alternate" type html
            for lt in link_tags:
                rel = (lt.get("rel") or "").lower()
                typ = (lt.get("type") or "").lower()
                href = (lt.get("href") or "").strip()
                if href and (rel in ("", "alternate")) and ("html" in typ or typ == ""):
                    link = href
                    break
            # fallback: lấy href bất kỳ
            if not link:
                for lt in link_tags:
                    href = (lt.get("href") or "").strip()
                    if href:
                        link = href
                        break
            # fallback: <link>TEXT</link>
            if not link:
                lt = ent.find("link")
                if lt:
                    link = (lt.get_text(strip=True) or "").strip()

            pub = ""
            if ent.find("updated"):
                pub = ent.find("updated").get_text(strip=True)
            elif ent.find("published"):
                pub = ent.find("published").get_text(strip=True)

            summ = (
                ent.find("summary").get_text(" ", strip=True)
                if ent.find("summary")
                else ""
            )

            if link:
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "published_raw": pub,
                        "summary": normalize_ws(
                            BeautifulSoup(summ, "lxml").get_text(" ", strip=True)
                        )
                        if summ
                        else "",
                    }
                )

    return items


def parse_datetime_maybe(s: str) -> Optional[str]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


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

    return {
        "url": url,
        "canonical_url": jsonld.get("canonical_url") or canonical,
        "title": normalize_ws(title or ""),
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


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Main crawl
# -----------------------------


def crawl_source(
    source: str,
    out_dir: str,
    since_dt: Optional[datetime] = None,
) -> int:
    """Crawl one source. Return number of articles saved.

    Args:
        source:   key inside RSS_FEEDS
        out_dir:  output directory
        since_dt: if given, only keep articles whose published_at >= since_dt
                  (UTC-aware). Articles with no timestamp are always kept.
    """
    feeds = RSS_FEEDS.get(source, [])
    if not feeds:
        print(
            f"[!] No RSS feeds configured for source='{source}'. Edit RSS_FEEDS first."
        )
        return 0

    session = build_session()
    if source == "yahoo":
        session.headers.update(
            {
                "Referer": "https://finance.yahoo.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
            }
        )

    seen_path = os.path.join(out_dir, f"seen_{source}.txt")
    out_path = os.path.join(out_dir, f"articles_{source}.jsonl")
    seen = load_seen(seen_path)

    discovered: List[Dict[str, Any]] = []

    # 1) RSS discovery
    for feed_url in feeds:
        try:
            r = session.get(feed_url, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                print(f"[RSS] {feed_url} -> HTTP {r.status_code}")
                continue

            parsed = parse_rss_items(r.text)
            if not parsed:
                print(
                    f"[DBG] empty parse for {feed_url}, first 120 chars = {r.text[:120]!r}"
                )
            else:
                discovered.extend(parsed)
        except Exception as e:
            print(f"[RSS] error {feed_url}: {e}")

    # Dedup by link at discovery stage
    uniq: Dict[str, Dict[str, Any]] = {}
    for it in discovered:
        link = it.get("link")
        if link:
            uniq[link] = it

    items = list(uniq.values())
    print(
        f"[+] [{source}] Discovered {len(items)} candidate URLs from {len(feeds)} RSS feeds"
    )

    # 2) Fetch + extract
    saved_rows: List[Dict[str, Any]] = []
    new_seen: List[str] = []

    for idx, it in enumerate(items, start=1):
        url = it["link"].strip()

        url_key = sha256_text(url)
        if url_key in seen:
            continue
        if is_non_html_extension(url):
            print(f"[SKIP] [{source}] non-html extension: {url}")
            continue

        # polite sleep
        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                print(
                    f"[GET] [{source}] {idx}/{len(items)} {url} -> HTTP {resp.status_code}"
                )
                continue

            raw_content_type = resp.headers.get("Content-Type") or ""
            if not is_html_content_type(raw_content_type):
                print(
                    f"[SKIP] [{source}] non-html content-type ({raw_content_type.lower()}): {url}"
                )
                continue

            html = resp.text
            if is_possible_wall(html):
                print(f"[SKIP] [{source}] possible wall/consent: {url}")
                continue

            art = extract_full_article(url, html)

            # Basic quality gate
            text = art.get("text") or ""
            if len(text) < 400:
                print(f"[DBG] [{source}] short content ({len(text)} chars): {url}")

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
                        print(
                            f"[SKIP] [{source}] too old ({pub_dt.isoformat()}): {url[:80]}"
                        )
                        continue
                except Exception:
                    pass  # can't parse date -> keep the article

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
                "summary_from_rss": it.get("summary"),
                "author": art.get("author"),
                "description": art.get("description"),
                "text": text,
                "content_hash": content_hash,
                "fetched_at": now_utc_iso(),
            }

            saved_rows.append(row)
            new_seen.append(url_key)

            print(f"[OK] [{source}] saved #{len(saved_rows)}: {row['title'][:80]}")

        except Exception as e:
            print(f"[ERR] [{source}] {idx}/{len(items)} {url}: {e}")

    if saved_rows:
        append_jsonl(out_path, saved_rows)
        append_seen(seen_path, new_seen)

    print(f"[+] [{source}] Saved {len(saved_rows)} articles -> {out_path}")
    return len(saved_rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Crawl tất cả nguồn tin tức được cấu hình trong RSS_FEEDS."
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
    args = ap.parse_args(argv)

    since_dt: Optional[datetime] = None
    if args.since is not None:
        since_dt = datetime.now(timezone.utc).replace(microsecond=0)
        from datetime import timedelta

        since_dt = since_dt - timedelta(seconds=args.since)
        print(f"[*] Lọc bài từ {since_dt.isoformat()} trở về sau ({args.since}s)")
    else:
        print("[*] Không lọc thời gian - cào toàn bộ bài hiện có trong RSS")

    sources = [s for s, feeds in RSS_FEEDS.items() if feeds]
    if not sources:
        print(
            "[!] Không có source nào được cấu hình trong RSS_FEEDS. Hãy thêm RSS URL trước."
        )
        return 1

    print(f"[*] Sẽ cào {len(sources)} source(s): {', '.join(sorted(sources))}")
    total = 0
    for source in sorted(sources):
        total += crawl_source(source, args.out, since_dt)

    print(f"\n[*] Hoàn tất. Tổng cộng {total} bài mới đã lưu.")
    return 0


# -----------------------------
# Tests
# -----------------------------


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

    def test_parse_rss_items_basic_rss(self):
        xml = """
        <rss><channel>
          <item><title>T</title><link>https://x.com/1</link><pubDate>Mon, 02 Mar 2026 10:00:00 GMT</pubDate>
          <description><![CDATA[<p>Hi</p>]]></description></item>
        </channel></rss>
        """
        items = parse_rss_items(xml)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["link"], "https://x.com/1")
        self.assertEqual(items[0]["title"], "T")
        self.assertEqual(items[0]["summary"], "Hi")

    def test_drop_tracking_removes_utm_and_fragment(self):
        u = "https://example.com/a?utm_source=x&id=123&utm_medium=y#frag"
        out = drop_tracking(u)
        self.assertEqual(out, "https://example.com/a?id=123")

    def test_drop_tracking_keeps_non_tracking_query(self):
        u = "https://example.com/a?symbol=BTCUSD&page=2"
        out = drop_tracking(u)
        self.assertEqual(out, u)

    def test_parse_datetime_maybe_with_z_suffix(self):
        iso = parse_datetime_maybe("2026-03-02T10:00:00Z")
        self.assertEqual(iso, "2026-03-02T10:00:00+00:00")

    def test_non_html_extension_detection(self):
        self.assertTrue(is_non_html_extension("https://example.com/file.pdf"))
        self.assertFalse(is_non_html_extension("https://example.com/article.html"))

    def test_wall_pattern_detection(self):
        self.assertTrue(is_possible_wall("Please enable JavaScript to continue"))
        self.assertFalse(is_possible_wall("Normal article body about markets and rates."))


if __name__ == "__main__":
    # If invoked as a script, run crawler.
    # For tests, use: python -m unittest finance_crawler.py
    raise SystemExit(main())
