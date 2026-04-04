"""
utils/output_handler.py
-----------------------
Post dataclass và hàm lưu kết quả crawl website theo 4 định dạng:
JSON, Markdown (.md), CSV, Parquet.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse


# ─── Dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Post:
    """Đại diện một bài viết được crawl từ website."""
    source: str                          # URL trang chủ / tên site
    post_type: str                       # "web"
    title: str                           # Tiêu đề bài viết
    date: Optional[datetime]             # Ngày đăng (có thể None)
    url: str                             # URL bài viết
    content: str                         # Nội dung văn bản
    images: List[str] = field(default_factory=list)  # Danh sách URL ảnh


# ─── Internal helpers ──────────────────────────────────────────────────────

def _safe_name(name: str, max_len: int = 80) -> str:
    """Chuyển tên thành chuỗi hợp lệ cho tên thư mục/file."""
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    return (cleaned or "unknown")[:max_len]


def _url_slug(url: str, max_len: int = 60) -> str:
    """Tạo slug từ URL để dùng làm tên file."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    return (path or "post")[-max_len:] or "post"


def _post_to_dict(post: Post) -> dict:
    """Chuyển Post thành dict có thể serialize."""
    return {
        "source": post.source,
        "post_type": post.post_type,
        "title": post.title,
        "date": post.date.strftime("%Y-%m-%d") if post.date else None,
        "url": post.url,
        "content": post.content,
        "images": post.images,
    }


def _post_to_markdown(post: Post) -> str:
    """Render Post thành chuỗi Markdown."""
    date_str = post.date.strftime("%d/%m/%Y") if post.date else "Không rõ"
    lines = [
        f"# {post.title}",
        "",
        f"**Nguồn:** {post.source}",
        f"**Ngày đăng:** {date_str}",
        f"**URL:** {post.url}",
        "",
        "---",
        "",
        post.content,
        "",
    ]
    if post.images:
        lines.append("## Hình ảnh đính kèm")
        lines.append("")
        for img in post.images:
            lines.append(f"- {img}")
    return "\n".join(lines)


def _post_to_flat(post: Post) -> dict:
    """Chuyển Post thành dict phẳng (cho CSV/Parquet)."""
    d = _post_to_dict(post)
    d["images"] = "; ".join(d["images"])
    return d


# ─── Public save function ──────────────────────────────────────────────────

def save_web_post(post: Post, output_dir: str = "web_post",
                  output_format: str = "json") -> str:
    """Lưu một Post vào output_dir theo định dạng chỉ định.

    Cấu trúc thư mục: output_dir/<source_slug>/<url_slug>.<ext>

    Args:
        post:          Đối tượng Post cần lưu.
        output_dir:    Thư mục gốc để lưu (mặc định "web_post").
        output_format: "json" | "md" | "csv" | "parquet"

    Returns:
        Đường dẫn file đã lưu.
    """
    source_slug = _safe_name(post.source or "web")
    url_slug = _url_slug(post.url)

    folder = os.path.join(output_dir, source_slug, url_slug)
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, url_slug)

    if output_format == "json":
        path = f"{base}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_post_to_dict(post), f, ensure_ascii=False, indent=2)

    elif output_format == "md":
        path = f"{base}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(_post_to_markdown(post))

    elif output_format == "csv":
        import pandas as pd
        path = f"{base}.csv"
        pd.DataFrame([_post_to_flat(post)]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )

    elif output_format == "parquet":
        import pandas as pd
        path = f"{base}.parquet"
        pd.DataFrame([_post_to_flat(post)]).astype(str).to_parquet(
            path, index=False
        )

    else:
        raise ValueError(f"Output format không hợp lệ: {output_format!r}")

    return path
