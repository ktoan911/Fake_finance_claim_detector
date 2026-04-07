import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_process.crawlers.crawl_fb import group_post_scraper_v2, post_scraper
from src.data_process.crawlers.crawl_fb.group_post_scraper_v2 import (
    fetch_posts as fetch_group_posts,
)
from src.data_process.crawlers.crawl_fb.main import (
    extract_group_id_from_url,
    extract_user_id_from_url,
)
from src.data_process.crawlers.crawl_fb.post_scraper import (
    fetch_posts as fetch_page_posts,
)
from src.models.fusion_inference import verify_claims_true_false


def crawl_page_posts(url: str, count: int, min_comments: int = 0):
    """
    Cào bài viết từ một trang cá nhân (page/profile).
    """
    print(f"\n[Page] {url}")
    page_id = extract_user_id_from_url(url)
    if not page_id:
        print("  ✗ Không lấy được Page ID từ URL")
        return

    print(f"  Page ID: {page_id}")
    post_scraper.USER_ID = page_id
    post_scraper.BASE_HEADERS["referer"] = (
        f"https://www.facebook.com/profile.php?id={page_id}"
    )

    all_posts_data = []

    def process_batch(batch_posts, total_so_far, total_limit):
        print(
            f"  Đang xử lý batch {len(batch_posts)} bài ({total_so_far}/{total_limit})..."
        )
        for j, post in enumerate(batch_posts, 1):
            post_id = post.get("post_id")
            if not post_id:
                continue
            print(f"    [{j}] Post {post_id}...")
            filtered_post = {
                "post_id": post.get("post_id"),
                "message": post.get("message"),
                "comment_count": post.get("comment_count"),
                "group_name": post.get("group_name"),
            }
            all_posts_data.append(filtered_post)
            print("✓ Đã lấy dữ liệu")
            time.sleep(1)

    print(f"  Đang tải {count} bài viết...")
    try:
        posts = fetch_page_posts(
            count, min_comments, batch_size=2, on_batch_complete=process_batch
        )
        print(f"  ✓ Hoàn thành: {len(posts)} bài")

        return all_posts_data
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")


def crawl_group_posts(url: str, count: int, min_comments: int = 0):
    """
    Cào bài viết từ một nhóm (group).
    """
    print(f"\n[Group] {url}")
    group_id = extract_group_id_from_url(url)
    if not group_id:
        print("  ✗ Không lấy được Group ID từ URL")
        return

    print(f"  Group ID: {group_id}")
    group_post_scraper_v2.GROUP_ID = group_id
    group_post_scraper_v2.HEADERS["referer"] = (
        f"https://www.facebook.com/groups/{group_id}/"
    )

    all_posts_data = []

    def process_batch(batch_posts, total_so_far, total_limit):
        print(
            f"  Đang xử lý batch {len(batch_posts)} bài ({total_so_far}/{total_limit})..."
        )
        for j, post in enumerate(batch_posts, 1):
            post_id = post.get("post_id")
            if not post_id:
                continue
            print(f"    [{j}] Post {post_id}...")
            filtered_post = {
                "post_id": post.get("post_id"),
                "message": post.get("message"),
                "comment_count": post.get("comment_count"),
                "group_name": post.get("group_name"),
            }
            all_posts_data.append(filtered_post)
            print("      ✓ Đã lấy dữ liệu")
            time.sleep(1)

    print(f"  Đang tải {count} bài viết...")
    try:
        posts = fetch_group_posts(
            count, min_comments, batch_size=2, on_batch_complete=process_batch
        )
        print(f"  ✓ Hoàn thành: {len(posts)} bài")

        return all_posts_data
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")


def crawl_from_file(file_path: str, count: int, min_comments: int = 0):
    """
    Đọc danh sách link từ file txt và chạy crawl cho từng link.
    """
    print(f"\n[File] Đọc danh sách link từ: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"  ✗ Không thể đọc file {file_path}: {e}")
        return

    print(f"  Tìm thấy {len(links)} link.")
    for i, link in enumerate(links, 1):
        print(f"\n--- Đang xử lý link {i}/{len(links)} ---")
        if "/groups/" in link or "/group/" in link:
            groups = crawl_group_posts(link, count, min_comments)
        else:
            pages = crawl_page_posts(link, count, min_comments)
    return pages + groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facebook Scraper bằng dòng lệnh (CLI)"
    )
    parser.add_argument("--url", help="URL của trang, nhóm hoặc đường dẫn file .txt")
    parser.add_argument(
        "--count", type=int, default=5, help="Số lượng bài viết tối đa cần lấy"
    )
    parser.add_argument(
        "--min_comments",
        type=int,
        default=0,
        help="Số lượng bình luận tối thiểu cần cho mỗi bài",
    )
    args = parser.parse_args()
    if not args.url:
        print(
            "Lỗi: Bạn cần cung cấp URL (trang, nhóm) hoặc file .txt thông qua tham số --url."
        )
        sys.exit(1)

    posts = crawl_from_file(args.url, args.count, args.min_comments)
    if not posts:
        print(
            "Không có bài viết nào được tìm thấy để kiểm chứng hoặc file không hợp lệ."
        )
        sys.exit(1)

    texts = [post["message"] for post in posts]
    results = verify_claims_true_false(
        claims=texts,
        fusion_model_path="models/fusion_model.pt",
        llm_model_path="models/lora_llm",
        retriever_model_path="AITeamVN/Vietnamese_Embedding",
        opensearch_index="news_kb",
        llm_evidence_top_k=5,
        debug=True,
    )
    for post, result in zip(posts, results):
        post["result"] = result
    # save json file
    with open("posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)
