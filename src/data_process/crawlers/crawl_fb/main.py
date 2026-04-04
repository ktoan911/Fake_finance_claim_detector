import base64
import json
import os
import re
import time
from html import unescape

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from crawl_fb.comment_scraper import PROXIES, fetch_comments, fetch_replies
from crawl_fb.group_post_scraper_v2 import fetch_posts as fetch_group_posts
from crawl_fb.post_scraper import fetch_posts as fetch_page_posts

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2b55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    """Remove emoji and special icon characters from text."""
    if not text:
        return text
    text = _EMOJI_PATTERN.sub("", text)
    return " ".join(text.split())


# ─── Output format helpers ─────────────────────────────────────────────────


def _normalize_post(post_data, comments_data):
    """Build a canonical post dict from any scraper's raw output + fetched comments.

    Canonical fields (same for every output format):
        post_id, source, post_date, post_url, post_text,
        image_urls (list), video_urls (list), comment_count, comments (list)
    """
    source = post_data.get("page_name") or post_data.get("group_name") or ""
    text = post_data.get("text") or post_data.get("message") or ""
    post_date = (post_data.get("created_time") or "").split(" ")[
        0
    ]  # date only, no time
    post_url = post_data.get("permalink") or post_data.get("permalink_url") or ""

    image_urls, video_urls = [], []
    for m in post_data.get("media") or []:
        if m.get("type") == "photo" and m.get("url"):
            image_urls.append(m["url"])
        elif m.get("type") == "video" and m.get("url"):
            video_urls.append(m["url"])
    for p in post_data.get("photos") or []:
        url = p.get("url") if isinstance(p, dict) else p
        if url and url not in image_urls:
            image_urls.append(url)
    for v in post_data.get("videos") or []:
        url = v.get("url") if isinstance(v, dict) else v
        if url and url not in video_urls:
            video_urls.append(url)
    for url in post_data.get("image_urls") or []:
        if url and url not in image_urls:
            image_urls.append(url)

    return {
        "post_id": post_data.get("post_id", ""),
        "source": source,
        "post_date": post_date,
        "post_url": post_url,
        "post_text": text,
        "image_urls": image_urls,
        "video_urls": video_urls,
        "comment_count": post_data.get("comment_count", len(comments_data)),
        "comments": comments_data,
    }


def _render_markdown(norm):
    """Render a normalized post dict as a Markdown document."""
    lines = [f"# Post {norm['post_id']}", ""]
    if norm["source"]:
        lines += [f"**Source:** {norm['source']}", ""]
    if norm["post_date"]:
        lines += [f"**Date:** {norm['post_date']}", ""]
    if norm["post_url"]:
        lines += [f"**URL:** <{norm['post_url']}>", ""]
    lines += ["**Content:**", "", norm["post_text"] or "(no text)", ""]

    if norm["image_urls"]:
        lines.append("**Images:**")
        for url in norm["image_urls"]:
            lines.append(f"- {url}")
        lines.append("")
    if norm["video_urls"]:
        lines.append("**Videos:**")
        for url in norm["video_urls"]:
            lines.append(f"- {url}")
        lines.append("")

    comments = norm["comments"]
    lines += ["---", f"## Comments ({len(comments)})", ""]
    for i, c in enumerate(comments, 1):
        ctext = c.get("text", "") or ""
        reactions = c.get("reaction_count", "0")
        lines.append(f"### {i}. {ctext}")
        lines.append(f"*Reactions: {reactions}*")
        replies = c.get("replies", [])
        if replies:
            lines += ["", "**Replies:**"]
            for r in replies:
                rtext = r.get("text", "") or ""
                rreact = r.get("reaction_count", "0")
                lines.append(f"- {rtext} *(Reactions: {rreact})*")
        lines.append("")

    return "\n".join(lines)


def _build_flat_rows(norm):
    """Build flat rows for CSV/Parquet from a normalized post dict."""
    base = {
        "post_id": norm["post_id"],
        "source": norm["source"],
        "post_date": norm["post_date"],
        "post_url": norm["post_url"],
        "post_text": norm["post_text"],
        "image_urls": "; ".join(norm["image_urls"]),
        "video_urls": "; ".join(norm["video_urls"]),
        "comment_count": norm["comment_count"],
    }
    rows = []
    for c in norm["comments"]:
        rows.append(
            {
                **base,
                "comment_text": c.get("text", "") or "",
                "comment_reactions": c.get("reaction_count", ""),
                "is_reply": False,
                "parent_comment": "",
            }
        )
        for r in c.get("replies", []):
            rows.append(
                {
                    **base,
                    "comment_text": r.get("text", "") or "",
                    "comment_reactions": r.get("reaction_count", ""),
                    "is_reply": True,
                    "parent_comment": c.get("text", "") or "",
                }
            )
    if not rows:
        rows.append(
            {
                **base,
                "comment_text": "",
                "comment_reactions": "",
                "is_reply": False,
                "parent_comment": "",
            }
        )
    return rows


def extract_user_id_from_url(url):
    """Extract Facebook User ID from a profile URL"""
    # First, try to extract ID directly from URL
    url_patterns = [r"profile\.php\?id=(\d+)", r"/profile/(\d+)", r"id=(\d+)"]

    for pattern in url_patterns:
        match = re.search(pattern, url)
        if match:
            user_id = match.group(1)
            print(f"  Found User ID in URL: {user_id}")
            return user_id

    # If no ID in URL, fetch the page and search in HTML
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        print(f"  No ID in URL, fetching page: {url}")
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=20)
        html = response.text

        # Try multiple patterns to find user ID in HTML
        patterns = [
            r"fb://profile/(\d+)",  # BEST signal
            r'"profile_owner":"(\d+)"',
            r'"userID":"(\d+)"',
            r"owner_id=(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                user_id = match.group(1)
                print(f"  Found User ID: {user_id}")
                return user_id

        print("   User ID not found (profile may be private or login wall)")
        return None

    except Exception as e:
        print(f"   Error fetching URL: {e}")
        return None


def extract_group_id_from_url(url):
    """Extract Facebook Group ID from a group URL"""
    # First, try to extract ID directly from URL
    url_patterns = [r"/groups/(\d+)", r"group_id=(\d+)", r"gid=(\d+)"]

    for pattern in url_patterns:
        match = re.search(pattern, url)
        if match:
            group_id = match.group(1)
            print(f"  Found Group ID in URL: {group_id}")
            return group_id

    # If no ID in URL, fetch the page and search in HTML
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        print(f"  No ID in URL, fetching group page: {url}")
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=20)
        html = response.text

        patterns = [
            r"fb://group/(\d+)",  # BEST signal
            r"fb://group/\?id=(\d+)",  # iOS URL format
            r'"group_id":"(\d+)"',
            r'"groupID":"(\d+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                group_id = match.group(1)
                print(f"  Found Group ID: {group_id}")
                return group_id

        print("   Group ID not found (group may be private or login wall)")
        return None

    except Exception as e:
        print(f"   Error fetching URL: {e}")
        return None


def extract_post_id_from_url(url):
    """Extract Facebook Post ID from a post URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        print(f"  Fetching post: {url}")
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=20)
        html = response.text

        # Extract og:url meta tag
        og_url_match = re.search(r'<meta property="og:url" content="([^"]+)"', html)

        post_id = None

        if og_url_match:
            og_url = unescape(og_url_match.group(1))

            # Case 1: /posts/POST_ID/ (group posts) or /posts/.../POST_ID/ (user posts)
            m = re.search(r"/posts/(?:[^/]+/)?(\d+)", og_url)

            # Case 2: permalink.php?story_fbid=POST_ID
            if not m:
                m = re.search(r"story_fbid=(\d+)", og_url)

            if m:
                post_id = m.group(1)

        if post_id:
            print(f"  Found Post ID: {post_id}")
            return post_id

        print("   Post ID not found in URL")
        return None

    except Exception as e:
        print(f"   Error fetching URL: {e}")
        return None


def convert_post_id_to_feedback_id(post_id):
    """Convert post_id to feedback_id using base64 encoding"""
    feedback_id = base64.b64encode(f"feedback:{post_id}".encode()).decode()
    return feedback_id


def fetch_comments_for_post(post_id):
    """Fetch all comments and replies for a given post_id."""
    feedback_id = convert_post_id_to_feedback_id(post_id)
    print(f"  Fetching comments for post {post_id}...")

    all_data = []
    comments, post_info = fetch_comments(feedback_id)

    for c in comments:
        c["replies"] = fetch_replies(c)

        # Clean emoji from comment and reply text
        c["text"] = clean_text(c.get("text", ""))
        for r in c["replies"]:
            r["text"] = clean_text(r.get("text", ""))

        # Remove internal fields
        c_clean = {k: v for k, v in c.items() if not k.startswith("_")}
        all_data.append(c_clean)

    print(f"  Found {len(all_data)} comments")
    return all_data, post_info


def save_post_data(post_type, post_id, post_data, comments_data, output_format="json"):
    """Normalize post + comments then save in the chosen format.

    All four formats share identical fields:
    post_id, source, post_date, post_url, post_text,
    image_urls, video_urls, comment_count, comments.
    """
    norm = _normalize_post(post_data, comments_data)

    source_folder = (
        "".join(
            c
            for c in (norm["source"] or "Unknown")
            if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        or "Unknown"
    )

    folder_path = os.path.join(post_type, source_folder, post_id)
    os.makedirs(folder_path, exist_ok=True)
    base_path = os.path.join(folder_path, post_id)

    if output_format == "json":
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(norm, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {base_path}.json")

    elif output_format == "md":
        with open(f"{base_path}.md", "w", encoding="utf-8") as f:
            f.write(_render_markdown(norm))
        print(f"  Saved to {base_path}.md")

    elif output_format == "csv":
        pd.DataFrame(_build_flat_rows(norm)).to_csv(
            f"{base_path}.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  Saved to {base_path}.csv")

    elif output_format == "parquet":
        pd.DataFrame(_build_flat_rows(norm)).astype(str).to_parquet(
            f"{base_path}.parquet", index=False
        )
        print(f"  Saved to {base_path}.parquet")

    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def display_menu():
    """Display the main menu."""
    print("\n" + "=" * 60)
    print("   FACEBOOK SCRAPER")
    print("=" * 60)
    print("\nChoose what to scrape:")
    print("  1. Simple Post (comments from a single post)")
    print("  2. Page Posts (posts + comments from a page)")
    print("  3. Group Posts (posts + comments from a group)")
    print("  4. Exit")
    print("=" * 60)


def scrape_simple_post():
    """Scrape comments from a single post."""
    print("\n--- SIMPLE POST SCRAPER ---")
    print("\nChoose input method:")
    print("  1. Enter Post URL (auto-extract ID)")
    print("  2. Enter Post ID directly")

    input_choice = input("Your choice (1 or 2): ").strip()
    post_id = None

    if input_choice == "1":
        post_url = input("Enter Post URL: ").strip()
        if not post_url:
            print("Invalid URL")
            return
        post_id = extract_post_id_from_url(post_url)
        if not post_id:
            print("Could not extract Post ID from URL")
            return
    elif input_choice == "2":
        post_id = input("Enter Post ID: ").strip()
        if not post_id:
            print("Invalid post ID")
            return
    else:
        print("Invalid choice")
        return

    print("\nChoose output format:")
    print("  1. JSON  2. Markdown (.md)  3. CSV  4. Parquet")
    fmt_choice = input("Format (1-4, default=1): ").strip() or "1"
    output_format = {"1": "json", "2": "md", "3": "csv", "4": "parquet"}.get(
        fmt_choice, "json"
    )

    print(f"\nFetching comments for post {post_id}...")
    comments, post_info = fetch_comments_for_post(post_id)

    # Collect image URLs (no download)
    image_urls = []
    if post_info and post_info.get("media_id"):
        media_id = post_info["media_id"]
        print(f"Collecting image links for media_id: {media_id}...")
        try:
            import single_post_image

            current_node = media_id
            visited = set()
            while current_node and current_node not in visited:
                visited.add(current_node)
                payload = single_post_image.build_payload(current_node, post_id)
                r = requests.post(
                    single_post_image.GRAPHQL_URL,
                    headers=single_post_image.HEADERS,
                    data=payload,
                    proxies=single_post_image.PROXIES,
                )
                cleaned_blocks = single_post_image.process_raw_graphql(r.text)
                if not cleaned_blocks:
                    break
                image_url = None
                for block in cleaned_blocks:
                    if "currMedia" in block:
                        image_url = block["currMedia"].get("image", {}).get("uri")
                        break
                if image_url:
                    image_urls.append(image_url)
                next_node = None
                for block in cleaned_blocks:
                    if (
                        "nextMediaAfterNodeId" in block
                        and block["nextMediaAfterNodeId"]
                    ):
                        nid = block["nextMediaAfterNodeId"].get("id")
                        if nid:
                            next_node = nid
                            break
                current_node = next_node
            print(f"  Found {len(image_urls)} image link(s)")
        except Exception as e:
            print(f"  Warning: error fetching image links: {e}")

    post_data = {
        "post_id": post_id,
        "type": "simple_post",
        "post_info": post_info,
        "image_urls": image_urls,
    }

    save_post_data("simple_post", post_id, post_data, comments, output_format)
    print(f"\nDone! Saved to simple_post/{post_id}/")


def _ask_output_format():
    """Prompt user to choose output format."""
    print("\nChoose output format:")
    print("  1. JSON  2. Markdown (.md)  3. CSV  4. Parquet")
    fc = input("Format (1-4, default=1): ").strip() or "1"
    return {"1": "json", "2": "md", "3": "csv", "4": "parquet"}.get(fc, "json")


def scrape_page_posts():
    """Scrape posts and comments from a page."""
    print("\n--- PAGE POST SCRAPER ---")
    print("\nChoose input method:")
    print("  1. Enter Page URL (auto-extract ID)")
    print("  2. Enter Page/User ID directly")

    input_choice = input("Your choice (1 or 2): ").strip()
    page_id = None

    if input_choice == "1":
        page_url = input("Enter Page URL: ").strip()
        if not page_url:
            print("Invalid URL")
            return
        page_id = extract_user_id_from_url(page_url)
        if not page_id:
            print("Could not extract User ID from URL")
            return
    elif input_choice == "2":
        page_id = input("Enter Page/User ID: ").strip()
        if not page_id:
            print("Invalid page ID")
            return
    else:
        print("Invalid choice")
        return

    output_format = _ask_output_format()
    try:
        count = int(input("How many posts to fetch? ").strip())
    except ValueError:
        print("Invalid number")
        return

    import post_scraper

    post_scraper.USER_ID = page_id
    post_scraper.BASE_HEADERS["referer"] = (
        f"https://www.facebook.com/profile.php?id={page_id}"
    )

    print(f"\nFetching {count} posts from page {page_id}...")
    posts = fetch_page_posts(count)

    print(f"\nFound {len(posts)} posts. Now fetching comments...")
    for i, post in enumerate(posts, 1):
        post_id = post.get("post_id")
        if not post_id:
            continue
        print(f"\n[{i}/{len(posts)}] Processing post {post_id}...")
        try:
            comments, _ = fetch_comments_for_post(post_id)
            save_post_data("page_post", post_id, post, comments, output_format)
            time.sleep(1)
        except Exception as e:
            print(f"  Error fetching comments: {e}")
            save_post_data("page_post", post_id, post, [], output_format)

    print(f"\nDone! Saved {len(posts)} posts to page_post/")


def scrape_group_posts():
    """Scrape posts and comments from a group."""
    print("\n--- GROUP POST SCRAPER ---")
    print("\nChoose input method:")
    print("  1. Enter Group URL (auto-extract ID)")
    print("  2. Enter Group ID directly")

    input_choice = input("Your choice (1 or 2): ").strip()
    group_id = None

    if input_choice == "1":
        group_url = input("Enter Group URL: ").strip()
        if not group_url:
            print("Invalid URL")
            return
        group_id = extract_group_id_from_url(group_url)
        if not group_id:
            print("Could not extract Group ID from URL")
            return
    elif input_choice == "2":
        group_id = input("Enter Group ID: ").strip()
        if not group_id:
            print("Invalid group ID")
            return
    else:
        print("Invalid choice")
        return

    output_format = _ask_output_format()
    try:
        count = int(input("How many posts to fetch? ").strip())
    except ValueError:
        print("Invalid number")
        return

    import group_post_scraper_v2

    group_post_scraper_v2.GROUP_ID = group_id
    group_post_scraper_v2.HEADERS["referer"] = (
        f"https://www.facebook.com/groups/{group_id}/"
    )

    print(f"\nFetching {count} posts from group {group_id}...")
    posts = fetch_group_posts(count)

    print(f"\nFound {len(posts)} posts. Now fetching comments...")
    for i, post in enumerate(posts, 1):
        post_id = post.get("post_id")
        if not post_id:
            continue
        print(f"\n[{i}/{len(posts)}] Processing post {post_id}...")
        try:
            comments, _ = fetch_comments_for_post(post_id)
            save_post_data("group_post", post_id, post, comments, output_format)
            time.sleep(1)
        except Exception as e:
            print(f"  Error fetching comments: {e}")
            save_post_data("group_post", post_id, post, [], output_format)

    print(f"\nDone! Saved {len(posts)} posts to group_post/")


def main():
    """Main entry point – CLI menu."""
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            scrape_simple_post()
        elif choice == "2":
            scrape_page_posts()
        elif choice == "3":
            scrape_group_posts()
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")

        if choice in ["1", "2", "3"]:
            cont = (
                input("\nPress Enter to return to menu (or 'q' to quit): ")
                .strip()
                .lower()
            )
            if cont == "q":
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
