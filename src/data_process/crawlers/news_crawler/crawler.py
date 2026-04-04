import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import aiohttp
import torch
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from transformers import AutoModel, AutoTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)

from src.database.opensearch import OpenSearchKB

# Load biến môi trường
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Danh sách URL đã được cập nhật/sửa lỗi 404
URLS_TO_CRAWL = [
    # --- Cơ quan quản lý ---
    "https://www.sbv.gov.vn/webcenter/portal/vi/menu/trangchu/ttsk",
    "https://div.gov.vn/tin-tuc-su-kien",
    "https://baochinhphu.vn/tai-chinh-ngan-hang.html",  # Đã sửa .htm -> .html
    "https://mof.gov.vn/webcenter/portal/btc/r/tc/ttsk",
    "https://cic.gov.vn",
    # --- Ngân hàng (Cập nhật link mới) ---
    "https://www.vietinbank.vn/vn/tin-tuc/",
    "https://bidv.com.vn/vn/tin-tuc-su-kien",
    "https://www.agribank.com.vn/vn/ve-agribank/tin-tuc-su-kien",
    "https://techcombank.com/khach-hang-ca-nhan/thong-tin-moi",  # Đã cập nhật
    "https://www.vpbank.com.vn/tin-tuc",
    "https://www.mbbank.com.vn/chi-tiet/tin-tuc",
    "https://acb.com.vn/tin-tuc",
    "https://www.sacombank.com.vn/trang-chu/tin-tuc/tin-sacombank.html",  # Đã cập nhật
    "https://hdbank.com.vn/vi/about/news",
    "https://tpb.vn/tin-tuc",
    "https://www.ocb.com.vn/vi/tin-tuc",
    "https://www.shb.com.vn/category/tin-tuc/",  # Đã cập nhật
    "https://www.seabank.com.vn/tin-tuc",  # Đã cập nhật
    # --- Báo chí ---
    "https://cafef.vn/tai-chinh-ngan-hang.chn",
    "https://vneconomy.vn/tai-chinh.htm",
    "https://vietnambiz.vn/tai-chinh/ngan-hang.htm",
    "https://vietstock.vn/tai-chinh.htm",
    "https://tinnhanhchungkhoan.vn/ngan-hang/",
    "https://tapchikinhtetaichinh.vn/kinh-te",  # Đã cập nhật
    "https://tapchikinhtetaichinh.vn/tai-chinh",  # Đã cập nhật
    "https://cafebiz.vn/tai-chinh.chn",
    "https://vnexpress.net/kinh-doanh/ngan-hang",
    "https://tuoitre.vn/kinh-doanh/tai-chinh.htm",
    "https://thanhnien.vn/kinh-te/tai-chinh-ngan-hang.htm",
    "https://dantri.com.vn/kinh-doanh/tai-chinh.htm",
    "https://laodong.vn/tien-te-dau-tu/",
    "https://plo.vn/kinh-te/tai-chinh-ngan-hang/",
    "https://znews.vn/tai-chinh.html",
    "https://vtv.vn/tai-chinh-ngan-hang.html",  # Đã cập nhật
    "https://vov.vn/kinh-te/tai-chinh/",
    # --- Cảnh báo ---
    "https://tinnhiemmang.vn/tin-tuc",
]


STATIC_FILE_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".zip",
    ".rar",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".mp4",
    ".mp3",
}
BLOCKED_URL_HINTS = [
    "/lien-he",
    "/gioi-thieu",
    "/about",
    "/careers",
    "/tuyen-dung",
    "/privacy",
    "/dieu-khoan",
    "/tag/",
    "/video/",
    "/podcast/",
    "/rss",
    "/search",
]
ARTICLE_URL_HINTS = [
    "tin",
    "news",
    "article",
    "chi-tiet",
    "su-kien",
    "ngan-hang",
    "tai-chinh",
    "kinh-doanh",
]
BLOCKED_CONTENT_HINTS = [
    "access denied",
    "forbidden",
    "just a moment",
    "captcha",
    "cloudflare",
    "verify you are human",
]


def normalize_domain(netloc: str) -> str:
    """Chuẩn hóa domain để so sánh ổn định giữa www/non-www."""
    host = netloc.split("@")[-1].split(":")[0].lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_same_domain(base_netloc: str, candidate_netloc: str) -> bool:
    base_host = normalize_domain(base_netloc)
    candidate_host = normalize_domain(candidate_netloc)
    if not base_host or not candidate_host:
        return False
    return candidate_host == base_host or candidate_host.endswith(f".{base_host}")


def normalize_article_url(raw_url: str) -> str:
    """Chuẩn hóa URL bài viết để giảm trùng lặp do fragment."""
    parsed = urlparse(raw_url)
    cleaned = parsed._replace(fragment="")
    return cleaned.geturl()


def should_skip_href(href: str) -> bool:
    href_lower = href.lower().strip()
    if not href_lower:
        return True
    return href_lower.startswith(("mailto:", "tel:", "#"))


def extract_link_candidates(link, base_url: str) -> list[str]:
    """Lấy các URL tiềm năng từ href/data-attrs/onclick."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw_url: str):
        if not raw_url:
            return
        raw = raw_url.strip()
        if not raw:
            return
        raw_lower = raw.lower()
        if raw_lower.startswith(("mailto:", "tel:", "#")):
            return
        if raw.startswith("//"):
            raw = f"https:{raw}"

        full_url = normalize_article_url(urljoin(base_url, raw))
        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            return
        if full_url in seen:
            return
        seen.add(full_url)
        candidates.append(full_url)

    href = link.get("href", "")
    if href and not href.lower().strip().startswith("javascript:"):
        _add(href)

    for attr in ("data-href", "data-url", "data-link", "data-redirect-url"):
        _add(link.get(attr, ""))

    inline_scripts = [link.get("onclick", ""), link.get("href", "")]
    for script_text in inline_scripts:
        if not script_text:
            continue
        for match in re.findall(
            r"""['"]((?:https?:)?//[^'"]+|/[^'"]+)['"]""", script_text
        ):
            _add(match)

    return candidates


def looks_like_article_url(full_url: str, title: str) -> bool:
    parsed = urlparse(full_url)
    path = parsed.path.lower()
    full_url_lower = full_url.lower()

    if not path or path == "/":
        return False
    if any(path.endswith(ext) for ext in STATIC_FILE_EXTENSIONS):
        return False
    if any(hint in full_url_lower for hint in BLOCKED_URL_HINTS):
        return False

    score = 0
    if re.search(r"(?:^|[/-])20\d{2}(?:[/-]|$)", full_url_lower):
        score += 2
    if path.endswith((".html", ".htm", ".chn", ".aspx", ".cms")):
        score += 2
    if any(hint in full_url_lower for hint in ARTICLE_URL_HINTS):
        score += 1
    if path.count("/") >= 2:
        score += 1
    if len(title) >= 20:
        score += 1

    return score >= 2


def parse_published_at(soup: BeautifulSoup) -> str | None:
    published_at = None
    meta_pub = (
        soup.find("meta", attrs={"property": "article:published_time"})
        or soup.find("meta", attrs={"name": "article:published_time"})
        or soup.find("meta", attrs={"name": "pubdate"})
        or soup.find("meta", attrs={"name": "publishdate"})
        or soup.find("meta", attrs={"property": "og:published_time"})
    )
    if meta_pub and meta_pub.get("content"):
        published_at = meta_pub["content"].strip()
    else:
        scripts = soup.find_all(
            "script",
            attrs={"type": re.compile(r"application/ld\+json", re.I)},
        )
        for sc in scripts:
            try:
                raw = sc.string if sc.string else sc.get_text(strip=False)
                data = json.loads(raw)
            except Exception:
                continue

            nodes = data if isinstance(data, list) else [data]
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if node.get("datePublished"):
                    published_at = node["datePublished"]
                    break
                graph = node.get("@graph")
                if isinstance(graph, list):
                    for g in graph:
                        if isinstance(g, dict) and g.get("datePublished"):
                            published_at = g["datePublished"]
                            break
                if published_at:
                    break
            if published_at:
                break

    if not published_at:
        return None

    try:
        dt = dtparser.parse(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return published_at


def extract_article_text(soup: BeautifulSoup) -> str:
    """Ưu tiên vùng article/content; fallback toàn trang nếu cần."""
    for tag in soup(
        ["script", "style", "noscript", "iframe", "header", "footer", "nav", "form"]
    ):
        tag.decompose()

    selectors = [
        "article",
        "[itemprop='articleBody']",
        ".article-content",
        ".article__body",
        ".news-content",
        ".news-detail",
        ".detail-content",
        ".content-detail",
        ".entry-content",
        ".fck_detail",
        ".ck-content",
        ".td-post-content",
        ".detail__content",
        ".main-detail-body",
        ".box-content-detail",
        ".article-body",
        "main",
    ]

    candidate = None
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            candidate = node
            break

    if candidate is None:
        best_len = 0
        for node in soup.find_all(["article", "section", "main", "div"], limit=500):
            paras = node.find_all("p")
            if len(paras) < 2:
                continue
            total_len = sum(len(p.get_text(" ", strip=True)) for p in paras)
            if total_len > best_len:
                best_len = total_len
                candidate = node

    scope = candidate if candidate is not None else soup
    raw_lines = [
        p.get_text(" ", strip=True)
        for p in scope.find_all(["p", "li", "h2", "h3"])
        if p.get_text(" ", strip=True)
    ]
    cleaned_lines = [line for line in raw_lines if len(line) >= 20]
    content = "\n".join(cleaned_lines)

    if len(content) >= 120:
        return content

    fallback_lines = [
        line.strip()
        for line in scope.get_text("\n", strip=True).splitlines()
        if len(line.strip()) >= 40
    ]
    return "\n".join(fallback_lines[:200])


def parse_article_html(html: str) -> tuple[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    content = extract_article_text(soup)
    content_lower = content.lower()
    if (
        any(hint in content_lower for hint in BLOCKED_CONTENT_HINTS)
        and len(content) < 2500
    ):
        content = ""
    published_at = parse_published_at(soup)
    return content, published_at


def build_url_variants(url: str) -> list[str]:
    """Sinh biến thể www/non-www để tăng tỷ lệ truy cập thành công."""
    parsed = urlparse(url)
    if not parsed.netloc:
        return [url]
    variants = [url]
    host = parsed.netloc
    if host.startswith("www."):
        alt_host = host[4:]
    else:
        alt_host = f"www.{host}"
    if alt_host != host:
        variants.append(parsed._replace(netloc=alt_host).geturl())
    return variants


async def extract_links_and_data(page, url):
    """Trích xuất các liên kết bài viết và dữ liệu cơ bản từ trang chuyên mục."""
    try:
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        articles_data = []
        links = []

        # Tiên quyết tìm thẻ bài báo hoặc các thẻ bao bọc (tránh menu, footer, v.v)
        containers = soup.find_all(["article", "section"])
        if not containers:
            containers = soup.select(
                "h1, h2, h3, h4, div.news-item, li.news-item, div.post-item, div.item-news, .list-news"
            )

        if containers:
            for c in containers:
                links.extend(c.find_all("a", href=True))

        # Nếu thu được quá ít link (có thể cấu trúc trang khác), fallback lấy toàn bộ link
        if len(links) < 15:
            links = soup.find_all("a", href=True)

        seen_urls = set()
        base_domain = urlparse(url).netloc

        for link in links:
            text = re.sub(r"\s+", " ", link.get_text(strip=True))
            if not text:
                text = re.sub(r"\s+", " ", link.get("title", "").strip())

            # Lọc các link rỗng hoặc quá ngắn
            if len(text) < 8:
                continue

            candidate_urls = extract_link_candidates(link, url)
            for full_url in candidate_urls:
                parsed = urlparse(full_url)

                # 1. Lọc bằng domain: loại bỏ link trỏ ra ngoài
                if not is_same_domain(base_domain, parsed.netloc):
                    continue

                # 2. Lọc link rác bằng heuristic (ưu tiên link giống bài viết)
                if not looks_like_article_url(full_url, text):
                    continue

                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                articles_data.append(
                    {"source_url": url, "article_url": full_url, "title": text}
                )

        # Fallback nới lỏng: nếu heuristic chưa bắt được link bài, lấy link nội bộ có title đủ dài
        if not articles_data:
            for link in links:
                text = re.sub(r"\s+", " ", link.get_text(strip=True))
                if not text:
                    text = re.sub(r"\s+", " ", link.get("title", "").strip())
                if len(text) < 12:
                    continue

                candidate_urls = extract_link_candidates(link, url)
                for full_url in candidate_urls:
                    parsed = urlparse(full_url)
                    if not is_same_domain(base_domain, parsed.netloc):
                        continue
                    if any(
                        parsed.path.lower().endswith(ext)
                        for ext in STATIC_FILE_EXTENSIONS
                    ):
                        continue
                    if any(hint in full_url.lower() for hint in BLOCKED_URL_HINTS):
                        continue
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)
                    articles_data.append(
                        {"source_url": url, "article_url": full_url, "title": text}
                    )
                    if len(articles_data) >= 60:
                        break
                if len(articles_data) >= 60:
                    break

        return articles_data
    except Exception as e:
        logging.error(f"Lỗi khi trích xuất dữ liệu từ {url}: {e}")
        return []


async def process_url(context, url, semaphore, results):
    """Truy cập một URL, chờ tải trang và trích xuất dữ liệu."""
    async with semaphore:
        logging.info(f"Đang xử lý: {url}")
        page = None
        try:
            page = await context.new_page()

            # Chặn tải hình ảnh, font, media để tăng tốc
            await page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in ["image", "font", "media"]
                    else route.continue_()
                ),
            )

            # Cố gắng đi tới URL; nếu lỗi DNS thì thử biến thể www/non-www
            response = None
            last_error = None
            reached_url = url
            timeout_fallback = False
            for candidate_url in build_url_variants(url):
                response = None
                try:
                    response = await page.goto(
                        candidate_url, wait_until="domcontentloaded", timeout=45000
                    )
                    reached_url = candidate_url
                    if response and response.status >= 400:
                        logging.warning(
                            f"Lỗi HTTP {response.status} khi truy cập {candidate_url}"
                        )
                        continue
                    break
                except Exception as nav_err:
                    last_error = nav_err
                    err_str = str(nav_err)
                    if "Timeout" in err_str:
                        logging.warning(
                            f"Timeout 45s tại {candidate_url}. Bỏ qua chờ thêm, bắt đầu bóc tách HTML hiện có..."
                        )
                        reached_url = candidate_url
                        timeout_fallback = True
                        break
                    logging.warning(f"Lỗi khi truy cập {candidate_url}: {err_str}")
                    continue

            if response is None and last_error is not None and not timeout_fallback:
                err_str = str(last_error)
                if "ERR_NAME_NOT_RESOLVED" in err_str:
                    logging.warning(
                        f"Không thể phân giải tên miền (Web sập hoặc sai URL): {url}"
                    )
                else:
                    logging.error(f"Lỗi mạng khi điều hướng đến {url}: {err_str}")
                return
            if response is not None and response.status >= 400:
                # Tất cả biến thể URL đều trả mã lỗi HTTP.
                return

            # Chờ để load các trang render bằng JS (VD: vietcombank load bằng API)
            # Giảm từ 3s → 1.5s; đủ cho hầu hết SPA render lần đầu
            await asyncio.sleep(1.5)

            # Thử cuộn trang xuống để kích hoạt lazy-loading (Bọc trong try-except phòng khi trang bị treo JS)
            try:
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(0.5)
            except Exception:
                pass  # Bỏ qua nếu cuộn trang lỗi

            # Trích xuất dữ liệu
            effective_url = page.url if page.url else reached_url
            extracted_data = await extract_links_and_data(page, effective_url)
            if extracted_data:
                for item in extracted_data:
                    # Giữ nguồn gốc là URL gốc user khai báo để báo cáo cuối kỳ.
                    item["source_url"] = url
                results.extend(extracted_data)
                logging.info(
                    f"Đã tìm thấy {len(extracted_data)} liên kết tiềm năng từ {url}"
                )
            else:
                logging.warning(f"Không tìm thấy dữ liệu trên {url}")

        except Exception as e:
            logging.error(f"Lỗi không xác định khi xử lý {url}: {e}")
        finally:
            if page:
                await page.close()


async def fetch_article_content(session, item, semaphore, cutoff_time=None):
    """Lấy nội dung chi tiết bài viết, có retry 3 lần khi bị lỗi mạng.

    Nếu cutoff_time được truyền vào, bài viết có published_at trước cutoff_time
    sẽ bị bỏ qua ngay lập tức (early-exit) mà không cần xử lý tiếp.
    """
    async with semaphore:
        candidate_urls = build_url_variants(item["article_url"])
        for candidate_url in candidate_urls:
            for attempt in range(3):  # retry 3 lần mỗi candidate
                try:
                    async with session.get(candidate_url, timeout=15) as response:
                        if response.status == 200:
                            html = await response.text(errors="ignore")
                            content, published_at = parse_article_html(html)

                            # --- Early timestamp filter ---
                            if cutoff_time and published_at:
                                try:
                                    pub_dt = dtparser.parse(published_at)
                                    if pub_dt.tzinfo is None:
                                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                                    if pub_dt < cutoff_time:
                                        item["content"] = ""
                                        item["published_at"] = published_at
                                        item["_skipped_old"] = True
                                        return
                                except Exception:
                                    pass

                            if content:
                                item["content"] = content
                                item["published_at"] = published_at
                                item["article_url"] = normalize_article_url(
                                    str(response.url)
                                )
                                return
                        elif response.status in (403, 406, 429):
                            break
                        else:
                            await asyncio.sleep(0.2)
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        logging.debug(
                            f"Lỗi lấy nội dung bằng aiohttp ({candidate_url}): {e}"
                        )

        item["content"] = ""
        item["published_at"] = None


async def fetch_article_content_playwright(context, item, semaphore, cutoff_time=None):
    """Fallback lấy nội dung bằng Playwright cho các URL bị chặn/JS-render.

    Nếu cutoff_time được truyền vào, bài viết có published_at trước cutoff_time
    sẽ bị bỏ qua ngay lập tức (early-exit).
    """
    async with semaphore:
        page = None
        try:
            page = await context.new_page()
            await page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in ["image", "font", "media"]
                    else route.continue_()
                ),
            )
            for candidate_url in build_url_variants(item["article_url"]):
                response = await page.goto(
                    candidate_url, wait_until="domcontentloaded", timeout=45000
                )
                if response and response.status >= 400:
                    continue

                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass

                await asyncio.sleep(1.0)
                try:
                    await page.evaluate("window.scrollBy(0, 1400)")
                    await asyncio.sleep(0.5)
                except Exception:
                    pass

                html = await page.content()
                content, published_at = parse_article_html(html)

                # --- Early timestamp filter ---
                if cutoff_time and published_at:
                    try:
                        pub_dt = dtparser.parse(published_at)
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        if pub_dt < cutoff_time:
                            item["_skipped_old"] = True
                            return
                    except Exception:
                        pass

                if content:
                    item["content"] = content
                    if not item.get("published_at"):
                        item["published_at"] = published_at
                    item["article_url"] = normalize_article_url(
                        page.url or candidate_url
                    )
                    return
        except Exception as e:
            logging.debug(
                f"Fallback Playwright thất bại cho {item.get('article_url')}: {e}"
            )
        finally:
            if page:
                await page.close()


async def main(args):
    results = []
    concurrency_limit = 10  # tăng từ 5 → 10 để crawl danh sách nhanh hơn
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Tính cutoff_time sớm để truyền vào các hàm fetch (early-exit)
    cutoff_time = None
    if args.timestamp:
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=args.timestamp)

    logging.info("Bắt đầu quá trình cào dữ liệu...")

    async with async_playwright() as p:
        # Cấu hình Chromium
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-http2",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )

        # Ngụy trang Context mạnh hơn
        context = await browser.new_context(
            ignore_https_errors=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="vi-VN",
            timezone_id="Asia/Ho_Chi_Minh",
            java_script_enabled=True,
        )

        # Thêm header ngụy trang
        await context.set_extra_http_headers(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        # Thay đổi navigator.webdriver = false để đánh lừa WAF
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        tasks = [process_url(context, url, semaphore, results) for url in URLS_TO_CRAWL]
        await asyncio.gather(*tasks)

        if results:
            logging.info(
                f"Đã thu thập {len(results)} liên kết. Bắt đầu tải nội dung bài viết..."
            )
            content_semaphore = asyncio.Semaphore(30)  # tăng từ 15 → 30
            connector = aiohttp.TCPConnector(ssl=False, limit=60)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.google.com/",
            }
            async with aiohttp.ClientSession(
                connector=connector, headers=headers
            ) as session:
                fetch_tasks = [
                    fetch_article_content(session, item, content_semaphore, cutoff_time)
                    for item in results
                ]
                await asyncio.gather(*fetch_tasks)

            # Bỏ qua bài đã bị đánh dấu _skipped_old (quá cũ) khỏi fallback Playwright
            missing_items = [
                item
                for item in results
                if not item.get("content", "").strip() and not item.get("_skipped_old")
            ]
            if missing_items:
                logging.info(
                    f"Còn {len(missing_items)} bài chưa có nội dung sau aiohttp. Thử fallback bằng Playwright..."
                )
                fallback_semaphore = asyncio.Semaphore(6)  # tăng từ 4 → 6
                fallback_tasks = [
                    fetch_article_content_playwright(
                        context, item, fallback_semaphore, cutoff_time
                    )
                    for item in missing_items
                ]
                await asyncio.gather(*fallback_tasks)

        await context.close()
        await browser.close()

    # Lọc kết quả cuối: bỏ bài không có nội dung hoặc đã bị early-exit vì quá cũ
    filtered_by_time = 0
    valid_results = []
    for item in results:
        if item.get("_skipped_old"):
            filtered_by_time += 1
            continue
        if not item.get("content", "").strip():
            continue

        # Safety net: lọc lại timestamp cho bài không có published_at lúc fetch
        if cutoff_time and item.get("published_at"):
            try:
                pub_dt = dtparser.parse(item["published_at"])
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < cutoff_time:
                    filtered_by_time += 1
                    continue
            except Exception:
                pass

        valid_results.append(item)

    # Fix (4): Deduplicate theo article_url (nhiều nguồn có thể trỏ cùng bài)
    seen_article_urls: set[str] = set()
    deduped_results = []
    for item in valid_results:
        url = item.get("article_url", "")
        if url not in seen_article_urls:
            seen_article_urls.add(url)
            deduped_results.append(item)
    if len(deduped_results) < len(valid_results):
        logging.info(
            f"Đã loại bỏ {len(valid_results) - len(deduped_results)} bài viết trùng article_url."
        )
    valid_results = deduped_results

    logging.info(
        f"Đã lọc bỏ {len(results) - len(valid_results) - filtered_by_time} bài viết không có nội dung."
    )
    if args.timestamp:
        logging.info(
            f"Đã lọc bỏ {filtered_by_time} bài viết đăng trước thời điểm {cutoff_time.isoformat()} (early-exit + safety net)."
        )

    # --- Thống kê số bài có nội dung hợp lệ theo từng nguồn ---
    source_content_counts = {source_url: 0 for source_url in URLS_TO_CRAWL}
    for item in valid_results:
        src = item.get("source_url")
        if not src:
            continue
        source_content_counts[src] = source_content_counts.get(src, 0) + 1

    logging.info("[THỐNG KÊ NGUỒN] Số bài có nội dung hợp lệ theo từng nguồn:")
    for source_url in URLS_TO_CRAWL:
        logging.info(
            f"  - {source_url}: {source_content_counts.get(source_url, 0)} bài"
        )

    # --- Báo cáo nguồn không cào được bài viết nào ---
    sources_with_links = {item["source_url"] for item in results}
    sources_with_content = {item["source_url"] for item in valid_results}

    no_links = [u for u in URLS_TO_CRAWL if u not in sources_with_links]
    links_but_no_content = [
        u
        for u in URLS_TO_CRAWL
        if u in sources_with_links and u not in sources_with_content
    ]

    if no_links:
        logging.warning(
            f"[NGUỒN TRẮNG - không tìm được link nào] ({len(no_links)} nguồn):\n"
            + "\n".join(f"  - {u}" for u in no_links)
        )
    if links_but_no_content:
        logging.warning(
            f"[NGUỒN RỖNG - có link nhưng không có nội dung hợp lệ] ({len(links_but_no_content)} nguồn):\n"
            + "\n".join(f"  - {u}" for u in links_but_no_content)
        )
    if not no_links and not links_but_no_content:
        logging.info("Tất cả nguồn đều cào được ít nhất 1 bài viết hợp lệ.")

    # Giới hạn số bài xử lý nếu có --max-articles
    if args.max_articles and len(valid_results) > args.max_articles:
        logging.info(
            f"Giới hạn xử lý: lấy {args.max_articles}/{len(valid_results)} bài (--max-articles)."
        )
        valid_results = valid_results[: args.max_articles]

    # --- Xử lý dữ liệu định kỳ theo batch ---
    batch_size = args.batch_size
    processed_count = 0

    expected_dim = int(os.getenv("RETRIEVER_EMBEDDING_DIM", 768))
    kb = None
    all_json_results = []

    if args.save_mode == "opensearch":
        kb = OpenSearchKB(
            index_name=os.getenv("OP_KB_NAME"),
            embedding_dim=expected_dim,
        )

    # ---------------------------------------------------------------------------
    # JSON mode: lưu thẳng text, KHÔNG load model embedding (tiết kiệm hàng tiếng)
    # ---------------------------------------------------------------------------
    if args.save_mode == "json":
        logging.info(
            f"save_mode=json: Bỏ qua bước embedding. Lưu {len(valid_results)} bài thô..."
        )
        output_file = "crawler_output.json"
        json_out = []
        for item in valid_results:
            title = re.sub(
                r"\s+", " ", re.sub(r"[\n\t\r]+", " ", item.get("title", ""))
            ).strip()
            content = re.sub(
                r"\s+", " ", re.sub(r"[\n\t\r]+", " ", item.get("content", ""))
            ).strip()
            if not content:
                continue
            chunk_id_raw = item.get("article_url", "") + "_0"
            # Nếu không parse được ngày đăng, dùng thời điểm cào làm fallback
            pub_at = item.get("published_at") or datetime.now(timezone.utc).isoformat()
            json_out.append(
                {
                    "id": hashlib.md5(chunk_id_raw.encode("utf-8")).hexdigest(),
                    "title": title,
                    "content": content,
                    "article_url": item.get("article_url", ""),
                    "source_url": item.get("source_url", ""),
                    "published_at": pub_at,
                }
            )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        logging.info(f"Đã lưu {len(json_out)} bài vào {output_file}.")
        logging.info(f"Hoàn thành! Tổng cộng {len(json_out)} bài hợp lệ.")
        return

    # ---------------------------------------------------------------------------
    # Late Chunking helpers
    # ---------------------------------------------------------------------------
    def _split_sentences(text: str, chunk_size: int = 1500) -> list[str]:
        """Tách văn bản thành các câu, gộp câu ngắn lại cho đến khi đủ chunk_size."""
        # Fix (6): Không tách sau viết tắt dạng "TP.", "TS.", "PGS.", "GS.", v.v.
        sentences = [
            s.strip()
            for s in re.split(
                r"(?<!\b[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][.])(?<=[.!?])\s+",
                text,
            )
            if s.strip()
        ]
        if not sentences:
            return [text]
        chunks, current = [], ""
        for sent in sentences:
            if not current:
                current = sent
                if len(current) >= chunk_size:
                    chunks.append(current)
                    current = ""
                continue
            candidate = f"{current} {sent}"
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def late_chunk_embed(
        tokenizer,
        model,
        full_text: str,
        chunk_texts: list[str],
        max_length: int = 2048,
    ) -> list[list[float]]:
        probe = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=False,
        )
        n_tokens = probe["input_ids"].shape[1]

        # Fix (1): fallback nếu vượt max_length
        if n_tokens >= max_length:
            logging.warning(
                f"Text quá dài ({n_tokens} tokens >= {max_length}), fallback sang encode từng chunk."
            )
            fallback = []
            for c in chunk_texts:
                enc = tokenizer(
                    c, return_tensors="pt", truncation=True, max_length=max_length
                )
                with torch.no_grad():
                    out = model(**enc)
                emb = out.last_hidden_state.mean(dim=1)[0]
                fallback.append(emb.tolist())
            return fallback

        # Tokenize full document với offset_mapping
        full_enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offset_mapping = full_enc.pop("offset_mapping")[0]  # (T, 2)

        with torch.no_grad():
            output = model(**full_enc)
        # token_embeddings: (1, T, H) → (T, H)
        token_embeddings = output.last_hidden_state[0]
        hidden_size = token_embeddings.shape[-1]

        # Xác định vị trí char của từng chunk trong full_text
        chunk_embeddings = []
        search_start = 0
        for chunk in chunk_texts:
            char_start = full_text.find(chunk, search_start)
            if char_start == -1:
                chunk_embeddings.append([0.0] * hidden_size)
                continue
            char_end = char_start + len(chunk)
            search_start = char_end

            # Fix (2): dùng overlap condition thay vì strict containment
            token_mask = [
                (s < char_end and e > char_start) for s, e in offset_mapping.tolist()
            ]
            if not any(token_mask):
                chunk_embeddings.append([0.0] * hidden_size)
                continue

            # Mean-pool các token trong span
            indices = torch.tensor(
                [i for i, m in enumerate(token_mask) if m], dtype=torch.long
            )
            span_emb = token_embeddings[indices].mean(dim=0)
            chunk_embeddings.append(span_emb.tolist())

        return chunk_embeddings

    # ---------------------------------------------------------------------------
    # Load model 1 lần để dùng chung cho các batch
    # ---------------------------------------------------------------------------
    logging.info("Đang khởi tạo model embedding trên CPU (late chunking)...")
    tokenizer = None
    model = None
    try:
        model_name = os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        logging.info(f"Đã tải model: {model_name}")
    except Exception as e:
        logging.error(f"Không thể tải model embedding: {e}")

    for i in range(0, len(valid_results), batch_size):
        batch_items = valid_results[i : i + batch_size]
        processed_batch = []

        for item in batch_items:
            # Bước 0: Chuẩn hóa văn bản
            title = re.sub(
                r"\s+", " ", re.sub(r"[\n\t\r]+", " ", item.get("title", ""))
            ).strip()
            content = re.sub(
                r"\s+", " ", re.sub(r"[\n\t\r]+", " ", item.get("content", ""))
            ).strip()

            if not content:
                continue

            # Bước 1: Xác định ranh giới chunk (câu / nhóm câu)
            chunk_texts = (
                _split_sentences(content) if len(content) > 1500 else [content]
            )

            # Bước 2: Late chunking — tạo full_text để tokenize 1 lần
            # Ghép title + content để ngữ cảnh title lan tỏa vào token embeddings
            full_text = f"{title} {content}".strip() if title else content

            # Bước 3: Tính late-chunk embeddings
            if tokenizer is not None and model is not None:
                try:
                    chunk_embeddings = late_chunk_embed(
                        tokenizer, model, full_text, chunk_texts
                    )
                except Exception as e:
                    logging.error(f"Lỗi late_chunk_embed: {e}")
                    chunk_embeddings = [[]] * len(chunk_texts)
            else:
                chunk_embeddings = [[]] * len(chunk_texts)

            # Bước 4: Tạo document cho mỗi chunk
            for chunk_idx, (chunk, emb) in enumerate(
                zip(chunk_texts, chunk_embeddings)
            ):
                new_item = item.copy()
                new_item["title"] = title
                new_item["content"] = f"{title} {chunk}".strip() if title else chunk
                # Fallback published_at → ngày cào nếu null
                if not new_item.get("published_at"):
                    new_item["published_at"] = datetime.now(timezone.utc).isoformat()
                chunk_id_raw = f"{new_item.get('article_url', '')}_{chunk_idx}"
                new_item["id"] = hashlib.md5(chunk_id_raw.encode("utf-8")).hexdigest()
                # Fix (5): chỉ gán embedding nếu dim khớp với expected_dim
                if emb and len(emb) == expected_dim:
                    new_item["embedding"] = emb
                elif emb:
                    logging.warning(
                        f"Embedding dim sai: got {len(emb)}, expected {expected_dim}. Bỏ qua chunk."
                    )
                processed_batch.append(new_item)

        if args.save_mode == "opensearch":
            kb.insert_many(processed_batch)
        elif args.save_mode == "json":
            all_json_results.extend(processed_batch)

        processed_count += len(processed_batch)
        logging.info(
            f"Đã xử lý và tạo mã nhúng (late chunking) xong batch có {len(processed_batch)} đoạn."
        )

    if args.save_mode == "json":
        output_file = "crawler_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_json_results, f, ensure_ascii=False, indent=2)
        logging.info(
            f"Đã lưu {len(all_json_results)} đoạn dữ liệu vào tệp {output_file}."
        )

    logging.info(
        f"Hoàn thành! Tổng cộng đã xử lý xong {processed_count} đoạn dữ liệu hợp lệ."
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Crawler for crypto news")
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Crawl articles from the last X seconds (e.g., 86400 for 1 day)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing and embedding generation",
    )
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["opensearch", "json"],
        default="opensearch",
        help="Chọn nơi lưu dữ liệu: 'opensearch' hoặc 'json'",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Giới hạn tối đa số bài viết xử lý (mặc định không giới hạn)",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main(args))
