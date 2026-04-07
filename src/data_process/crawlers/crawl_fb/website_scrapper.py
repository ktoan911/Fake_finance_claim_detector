import logging
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.data_process.crawlers.crawl_fb.utils.output_handler import Post

logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

LISTING_SELECTORS = [
    "article a",
    ".post a",
    ".entry a",
    ".news-item a",
    ".item a",
    "h2 a",
    "h3 a",
    ".title a",
    ".tieu-de a",
    "a.post-title",
    ".tin-tuc a",
    ".bai-viet a",
]

CONTENT_SELECTORS = [
    "article",
    ".post-content",
    ".entry-content",
    ".content-detail",
    ".article-body",
    ".news-content",
    "#content",
    "main",
    ".noi-dung",
    ".bai-viet-chi-tiet",
]

DATE_SELECTORS_ATTRS = [
    "time[datetime]",
    "[itemprop='datePublished']",
    "[itemprop='dateModified']",
    "meta[property='article:published_time']",
    "[datetime]",
    "abbr[title]",
]

DATE_SELECTORS_TEXT = [
    "time",
    ".date",
    ".post-date",
    ".published",
    ".entry-date",
    ".news-date",
    ".ngay-dang",
    ".post-meta .date",
    ".entry-meta .date",
]

DATE_RE = re.compile(
    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})"
    r"|(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})"
    r"|(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})"
)

DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%d %m %Y",
]

DATE_FORMATS_ELEM = [
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%Y-%m-%d",
]

_VI_MONTHS_RE = re.compile(r"tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.I)


def _try_parse(text: str) -> datetime | None:
    text = text.strip()
    matches = DATE_RE.findall(text)
    candidates = [s for group in matches for s in group if s]
    if not candidates:
        candidates = [text]
    for candidate in candidates:
        # Xử lý "ngày X tháng Y năm Z"
        m = _VI_MONTHS_RE.search(candidate)
        if m:
            try:
                return datetime(int(m.group(2)), int(m.group(1)), 1)
            except Exception:
                pass
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(candidate.strip(), fmt)
            except ValueError:
                continue
    return None


def _parse_date_from_element(el) -> datetime | None:
    for attr in ["datetime", "content", "title"]:
        val = el.get(attr)
        if val:
            d = _try_parse(val)
            if d:
                return d

    raw = el.get_text(separator=" ", strip=True)
    raw = re.sub(r"[^\d\/\-\s\.]", " ", raw).strip()
    raw = re.sub(r"\s{2,}", " ", raw).strip()

    if not raw:
        return None

    m = re.search(r"(\d{1,2}[\/-]\d{1,2}[\/-]\d{4})", raw)
    if m:
        candidate = m.group(1)
        a, b, year = re.split(r"[\/-]", candidate)
        a, b, year = int(a), int(b), int(year)
        if a > 12 and b <= 12:
            try:
                return datetime(year, b, a)
            except Exception:
                pass
        elif b > 12 and a <= 12:
            try:
                return datetime(year, a, b)
            except Exception:
                pass
        else:
            for fmt in DATE_FORMATS_ELEM:
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue

    return _try_parse(raw)


class WebCrawler:
    def __init__(self, date_threshold: datetime, timeout: int = 15):
        self.date_threshold = date_threshold
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def crawl(self, url: str) -> list[Post]:
        """Crawl bài viết từ trang chủ / trang danh sách."""
        posts: list[Post] = []
        logger.info("Crawl website: %s", url)
        try:
            article_links = self._get_article_links(url)
            logger.info("Tìm thấy %d link bài viết", len(article_links))
            for link in article_links:
                post = self._crawl_article(link, source=url)
                if post is None:
                    continue
                if post.date and post.date < self.date_threshold:
                    logger.debug("Bỏ qua bài cũ: %s (%s)", post.title, post.date)
                    continue
                posts.append(post)
                logger.debug("  + %s [%s]", post.title[:60], post.date)
        except Exception as e:
            logger.error("Lỗi crawl %s: %s", url, e)
        return posts

    def _get_article_links(self, url: str) -> list[str]:
        resp = self.session.get(url, timeout=self.timeout)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        links: list[str] = []
        seen: set[str] = set()

        for selector in LISTING_SELECTORS:
            for a in soup.select(selector):
                href = a.get("href", "")
                if not href or href.startswith("#"):
                    continue
                full = urljoin(base, href)
                if urlparse(full).netloc != urlparse(url).netloc:
                    continue
                if full.rstrip("/") == url.rstrip("/"):
                    continue
                if full not in seen:
                    seen.add(full)
                    links.append(full)

        return links

    def _crawl_article(self, url: str, source: str) -> Post | None:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning("Không lấy được %s: %s", url, e)
            return None

        # ---- Tiêu đề ----
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "").strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
        if not title:
            title = soup.title.string.strip() if soup.title else url

        post_date = None

        for selector in DATE_SELECTORS_ATTRS:
            el = soup.select_one(selector)
            if el:
                post_date = _parse_date_from_element(el)
                if post_date:
                    break

        if not post_date:
            for selector in CONTENT_SELECTORS:
                content_el = soup.select_one(selector)
                if content_el and len(content_el.get_text(strip=True)) > 100:
                    content_text = content_el.get_text(separator="\n")
                    standalone = re.search(
                        r"(?:^|\n)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\s*(?:\n|$)",
                        content_text,
                    )
                    if standalone:
                        post_date = _try_parse(standalone.group(1))
                    if post_date:
                        break

        if not post_date:
            for selector in DATE_SELECTORS_TEXT:
                el = soup.select_one(selector)
                if el:
                    post_date = _parse_date_from_element(el)
                    if post_date:
                        break

        if not post_date:
            m = DATE_RE.search(soup.get_text())
            if m:
                raw = next((s for s in m.groups() if s), "")
                post_date = _try_parse(raw)

        # ---- Nội dung ----
        content = ""
        for selector in CONTENT_SELECTORS:
            el = soup.select_one(selector)
            if el:
                # Loại script/style
                for tag in el(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                content = el.get_text(separator="\n", strip=True)
                if len(content) > 100:
                    break

        if not content:
            # Fallback: body
            body = soup.body
            if body:
                for tag in body(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                content = body.get_text(separator="\n", strip=True)

        if not content or len(content) < 50:
            return None

        # ---- Ảnh đính kèm ----
        images: list[str] = []
        _content_el = None
        for selector in CONTENT_SELECTORS:
            _content_el = soup.select_one(selector)
            if _content_el and len(_content_el.get_text(strip=True)) > 100:
                break
        if _content_el is None:
            _content_el = soup.body
        if _content_el:
            seen_imgs: set[str] = set()
            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            for img in _content_el.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if not src or src.startswith("data:"):
                    continue
                full_src = urljoin(base, src)
                if full_src not in seen_imgs:
                    seen_imgs.add(full_src)
                    images.append(full_src)

        return Post(
            source=source,
            post_type="web",
            title=title,
            date=post_date,
            url=url,
            content=content,
            images=images,
        )
