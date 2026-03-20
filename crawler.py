import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import aiohttp
import torch
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from transformers import AutoModel, AutoTokenizer

from database.opensearch import OpenSearchKB

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


async def extract_links_and_data(page, url):
    """Trích xuất các liên kết bài viết và dữ liệu cơ bản từ trang chuyên mục."""
    try:
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        articles_data = []
        links = []

        # Tiên quyết tìm thẻ bài báo hoặc các thẻ bao bọc (tránh menu, footer, v.v)
        containers = soup.find_all(["article"])
        if not containers:
            containers = soup.select("h2, h3, h4, div.news-item, li.news-item")

        if containers:
            for c in containers:
                links.extend(c.find_all("a", href=True))

        # Nếu thu được quá ít link (có thể cấu trúc trang khác), fallback lấy toàn bộ link
        if len(links) < 15:
            links = soup.find_all("a", href=True)

        seen_urls = set()
        base_domain = urlparse(url).netloc

        valid_keywords = [
            "tin",
            "news",
            "article",
            "202",
            "ngan-hang",
            ".html",
            "-",
            "chi-tiet",
            "su-kien",
            "tai-chinh",
        ]

        for link in links:
            href = link["href"]
            text = link.get_text(strip=True)

            # Lọc các link rỗng hoặc quá ngắn
            if not href or len(text) < 15:
                continue

            full_url = urljoin(url, href)

            # 1. Lọc bằng domain: loại bỏ link trỏ ra ngoài
            if urlparse(full_url).netloc != base_domain:
                continue

            # 2. Lọc link rác (không có chứa keyword liên quan tin tức)
            full_url_lower = full_url.lower()
            if not any(word in full_url_lower for word in valid_keywords):
                continue

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            articles_data.append(
                {"source_url": url, "article_url": full_url, "title": text}
            )

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

            # Cố gắng đi tới URL và bắt toàn bộ ngoại lệ (Kể cả Timeout)
            try:
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=45000
                )
                if response and response.status >= 400:
                    logging.warning(f"Lỗi HTTP {response.status} khi truy cập {url}")
                    # Đối với HTTP 403/406, trang vẫn có thể trả về thông báo lỗi dạng HTML, ta dừng tại đây
                    return
            except Exception as nav_err:
                err_str = str(nav_err)
                if "Timeout" in err_str:
                    logging.warning(
                        f"Timeout 45s tại {url}. Bỏ qua chờ thêm, bắt đầu bóc tách HTML hiện có..."
                    )
                elif "ERR_NAME_NOT_RESOLVED" in err_str:
                    logging.warning(
                        f"Không thể phân giải tên miền (Web sập hoặc sai URL): {url}"
                    )
                    return
                else:
                    logging.error(f"Lỗi mạng khi điều hướng đến {url}: {err_str}")
                    return

            # Chờ để load các trang render bằng JS (VD: vietcombank load bằng API)
            # Chờ một lúc để các trang SPA (React/Vue/API) kịp gọi XMLHttpRequest và render HTML ra cây DOM
            await asyncio.sleep(3)

            # Thử cuộn trang xuống để kích hoạt lazy-loading (Bọc trong try-except phòng khi trang bị treo JS)
            try:
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(1.5)
            except Exception:
                pass  # Bỏ qua nếu cuộn trang lỗi

            # Trích xuất dữ liệu
            extracted_data = await extract_links_and_data(page, url)
            if extracted_data:
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


async def fetch_article_content(session, item, semaphore):
    """Lấy nội dung chi tiết bài viết, có retry 3 lần khi bị lỗi mạng."""
    async with semaphore:
        for attempt in range(3):  # Fix (3): retry 3 lần
            try:
                async with session.get(item["article_url"], timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Tìm thẻ chứa nội dung chính
                        article = soup.find("article")
                        if not article:
                            article = soup.find(
                                "div",
                                class_=lambda c: (
                                    c
                                    and (
                                        "content" in c.lower() or "detail" in c.lower()
                                    )
                                ),
                            )
                        if not article:
                            article = soup

                        paragraphs = article.find_all("p")
                        raw_text = [p.get_text(strip=True) for p in paragraphs]
                        content = "\n".join(
                            [text for text in raw_text if len(text) > 30]
                        )
                        item["content"] = content

                        # Trích xuất published_at
                        published_at = None
                        meta_pub = (
                            soup.find(
                                "meta", attrs={"property": "article:published_time"}
                            )
                            or soup.find(
                                "meta", attrs={"name": "article:published_time"}
                            )
                            or soup.find("meta", attrs={"name": "pubdate"})
                        )

                        if meta_pub and meta_pub.get("content"):
                            published_at = meta_pub["content"].strip()
                        else:
                            scripts = soup.find_all(
                                "script",
                                attrs={
                                    "type": re.compile(r"application/ld\+json", re.I)
                                },
                            )
                            for sc in scripts:
                                try:
                                    data = json.loads(
                                        sc.string
                                        if sc.string
                                        else sc.get_text(strip=False)
                                    )
                                    if isinstance(data, dict):
                                        if "datePublished" in data:
                                            published_at = data["datePublished"]
                                            break
                                        if "@graph" in data and isinstance(
                                            data["@graph"], list
                                        ):
                                            for g in data["@graph"]:
                                                if (
                                                    isinstance(g, dict)
                                                    and "datePublished" in g
                                                ):
                                                    published_at = g["datePublished"]
                                                    break
                                    if published_at:
                                        break
                                except Exception:
                                    pass

                        if published_at:
                            try:
                                dt = dtparser.parse(published_at)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                item["published_at"] = dt.astimezone(
                                    timezone.utc
                                ).isoformat()
                            except Exception:
                                item["published_at"] = published_at
                        else:
                            item["published_at"] = None
                    else:
                        item["content"] = ""
                        item["published_at"] = None
                    return  # thành công → thoát retry loop
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    item["content"] = ""
                    item["published_at"] = None


async def main(args):
    results = []
    concurrency_limit = 5
    semaphore = asyncio.Semaphore(concurrency_limit)

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

        await context.close()
        await browser.close()

    if results:
        logging.info(
            f"Đã thu thập {len(results)} liên kết. Bắt đầu tải nội dung bài viết..."
        )
        content_semaphore = asyncio.Semaphore(15)
        connector = aiohttp.TCPConnector(ssl=False)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(
            connector=connector, headers=headers
        ) as session:
            fetch_tasks = [
                fetch_article_content(session, item, content_semaphore)
                for item in results
            ]
            await asyncio.gather(*fetch_tasks)

    valid_results = []
    cutoff_time = None
    if args.timestamp:
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=args.timestamp)

    filtered_by_time = 0
    for item in results:
        if not item.get("content") or len(item["content"]) <= 10:
            continue

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
        f"Đã lọc bỏ {len(results) - len(valid_results) - filtered_by_time} bài viết không có nội dung hoặc quá ngắn."
    )
    if args.timestamp:
        logging.info(
            f"Đã lọc bỏ {filtered_by_time} bài viết đăng trước thời điểm {cutoff_time.isoformat()}."
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

    # --- Xử lý dữ liệu định kỳ theo batch ---
    batch_size = args.batch_size
    processed_count = 0
    kb = OpenSearchKB(
        index_name=os.getenv("OP_KB_NAME"),
        embedding_dim=int(os.getenv("RETRIEVER_EMBEDDING_DIM")),
    )

    # ---------------------------------------------------------------------------
    # Late Chunking helpers
    # ---------------------------------------------------------------------------
    def _split_sentences(text: str, chunk_size: int = 200) -> list[str]:
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
        """
        Late Chunking:
        1. Tokenize toàn bộ tài liệu, lấy token embeddings (contextualized).
        2. Với mỗi chunk, map char span → token span rồi mean-pool.

        Fix (1): Nếu full_text vượt max_length token → fallback sang
        encode từng chunk riêng (tránh zero vector hàng loạt do truncate).
        """
        # Tokenize để check độ dài, chưa truncate
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
            chunk_texts = _split_sentences(content) if len(content) > 200 else [content]

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
                chunk_id_raw = f"{new_item.get('article_url', '')}_{chunk_idx}"
                new_item["id"] = hashlib.md5(chunk_id_raw.encode("utf-8")).hexdigest()
                # Fix (5): chỉ gán embedding nếu dim khớp với kb.embedding_dim
                if emb and len(emb) == kb.embedding_dim:
                    new_item["embedding"] = emb
                elif emb:
                    logging.warning(
                        f"Embedding dim sai: got {len(emb)}, expected {kb.embedding_dim}. Bỏ qua chunk."
                    )
                processed_batch.append(new_item)

        kb.insert_many(processed_batch)

        processed_count += len(processed_batch)
        logging.info(
            f"Đã xử lý và tạo mã nhúng (late chunking) xong batch có {len(processed_batch)} đoạn."
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
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main(args))
