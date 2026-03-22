import json
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

GRAPHQL_URL = "https://www.facebook.com/api/graphql/"

# ========= CONFIG (FILL THESE) =========
USER_ID = "100019577483175"  # profile / page id
PAGE_NAME = None  # Will be extracted automatically
DOC_ID = "25430544756617998"  # ProfileCometTimelineFeedRefetchQuery


# ========= RETRY HELPER =========
def retry_request(url, headers, data, proxies, max_retries=5):
    """Make a POST request with retry logic"""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                url, headers=headers, data=data, proxies=proxies, timeout=30
            )
            if r.status_code == 200:
                return r
            print(f"   Attempt {attempt}/{max_retries}: Status {r.status_code}")
        except Exception as e:
            print(f"   Attempt {attempt}/{max_retries}: {str(e)}")

        if attempt < max_retries:
            wait_time = attempt * 2  # Exponential backoff: 2, 4, 6, 8, 10 seconds
            print(f"   Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception(f"Failed after {max_retries} attempts")


def fetch_remaining_image_urls(last_media_id, post_id, current_image_count):
    """Fetch remaining image URLs for posts with 5+ images (no downloads)."""
    if not last_media_id or not post_id:
        return []

    DOC_ID_PHOTO = "26168653472729001"
    HEADERS_PHOTO = {
        "user-agent": "Mozilla/5.0",
        "content-type": "application/x-www-form-urlencoded",
        "origin": "https://www.facebook.com",
        "x-fb-friendly-name": "CometPhotoRootContentQuery",
    }

    remaining = []
    current_node = last_media_id
    visited = set()
    image_index = current_image_count + 1

    while current_node and current_node not in visited and image_index <= 50:
        visited.add(current_node)

        variables = {
            "isMediaset": True,
            "renderLocation": "comet_media_viewer",
            "nodeID": current_node,
            "mediasetToken": f"pcb.{post_id}",
            "scale": 2,
            "feedLocation": "COMET_MEDIA_VIEWER",
            "feedbackSource": 65,
            "focusCommentID": None,
            "privacySelectorRenderLocation": "COMET_MEDIA_VIEWER",
            "useDefaultActor": False,
            "shouldShowComments": True,
        }

        payload = {
            "av": "0",
            "__user": "0",
            "__a": "1",
            "doc_id": DOC_ID_PHOTO,
            "variables": json.dumps(variables),
        }

        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS_PHOTO,
                data=payload,
                proxies=PROXIES,
                timeout=30,
            )
            if r.status_code != 200:
                break

            cleaned_blocks = parse_fb_response(r.text)
            if not cleaned_blocks:
                break

            image_url = None
            for block in cleaned_blocks:
                if "currMedia" in block:
                    image_url = block["currMedia"].get("image", {}).get("uri")
                    break

            if image_url:
                remaining.append({"type": "photo", "url": image_url})
                image_index += 1

            next_node = None
            for block in cleaned_blocks:
                if "nextMediaAfterNodeId" in block and block["nextMediaAfterNodeId"]:
                    node_id = block["nextMediaAfterNodeId"].get("id")
                    if node_id:
                        next_node = node_id
                        break

            if next_node:
                current_node = next_node
                time.sleep(0.5)
            else:
                break

        except Exception as e:
            print(f"  Warning: error fetching next image: {e}")
            break

    return remaining


# -----------------------------
# Extract all "data" blocks from raw text
# -----------------------------
def extract_data_blocks(raw_text):
    blocks = []
    i = 0
    n = len(raw_text)

    while True:
        idx = raw_text.find('"data"', i)
        if idx == -1:
            break

        brace_start = raw_text.find("{", idx)
        if brace_start == -1:
            break

        depth = 0
        for j in range(brace_start, n):
            if raw_text[j] == "{":
                depth += 1
            elif raw_text[j] == "}":
                depth -= 1
                if depth == 0:
                    block_text = raw_text[brace_start : j + 1]
                    try:
                        block = json.loads(block_text)
                        blocks.append(block)
                    except Exception:
                        pass
                    i = j + 1
                    break
        else:
            break

    return blocks


# -----------------------------
# Clean unwanted keys
# -----------------------------
def clean_data_blocks(blocks):
    cleaned = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block.pop("errors", None)
        block.pop("extensions", None)

        cleaned.append(block)

    return cleaned


# -----------------------------
# Parse Facebook response using cleaning logic
# -----------------------------
def parse_fb_response(text):
    text = text.replace("for (;;);", "").strip()
    extracted = extract_data_blocks(text)
    cleaned = clean_data_blocks(extracted)

    # Return the cleaned array as-is
    return cleaned


BASE_HEADERS = {
    "user-agent": "Mozilla/5.0",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.facebook.com",
    "referer": f"https://www.facebook.com/profile.php?id={USER_ID}",
}

# Get proxy configuration
PROXY = os.getenv("PROXY")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

if PROXY:
    print(f"Using proxy: {PROXY}")


def extract_page_name(node):
    """Extract page/user name from post node"""
    try:
        # Try from actors
        actors = (
            node.get("comet_sections", {})
            .get("content", {})
            .get("story", {})
            .get("actors", [])
        )
        if actors and len(actors) > 0:
            return actors[0].get("name")

        # Try from feedback > owning_profile
        feedback = node.get("feedback", {})
        owning_profile = feedback.get("owning_profile", {})
        if owning_profile:
            return owning_profile.get("name") or owning_profile.get("short_name")

        return None
    except Exception:
        return None


def extract_comment_count(node):
    """Extract comment count from post node"""
    try:
        # Path 1: feedback.comment_rendering_instance.comments.total_count
        comment_count = (
            node.get("feedback", {})
            .get("comment_rendering_instance", {})
            .get("comments", {})
            .get("total_count")
        )
        if comment_count is not None:
            return comment_count

        # Path 2: comet_sections.feedback.story.story_ufi_container.story.feedback_context.feedback_target_with_context.comment_rendering_instance.comments.total_count
        comet_sections = node.get("comet_sections", {})
        feedback_section = comet_sections.get("feedback", {})
        story = feedback_section.get("story", {})
        story_ufi_container = story.get("story_ufi_container", {})
        ufi_story = story_ufi_container.get("story", {})
        feedback_context = ufi_story.get("feedback_context", {})
        feedback_target = feedback_context.get("feedback_target_with_context", {})
        comment_count = (
            feedback_target.get("comment_rendering_instance", {})
            .get("comments", {})
            .get("total_count")
        )
        if comment_count is not None:
            return comment_count

        # Path 3: comet_sections.feedback.story.story_ufi_container.story.feedback_context.feedback_target_with_context.comet_ufi_summary_and_actions_renderer.feedback.comment_rendering_instance.comments.total_count
        comet_ufi = feedback_target.get(
            "comet_ufi_summary_and_actions_renderer", {}
        ).get("feedback", {})
        comment_count = (
            comet_ufi.get("comment_rendering_instance", {})
            .get("comments", {})
            .get("total_count")
        )
        if comment_count is not None:
            return comment_count

        # Path 4: comet_sections.feedback.story.feedback_context.feedback_target_with_context.comment_rendering_instance.comments.total_count (old structure)
        comet_sections = node.get("comet_sections", {})
        feedback_section = comet_sections.get("feedback", {})
        story = feedback_section.get("story", {})
        feedback_context = story.get("feedback_context", {})
        feedback_target = feedback_context.get("feedback_target_with_context", {})
        comment_count = (
            feedback_target.get("comment_rendering_instance", {})
            .get("comments", {})
            .get("total_count")
        )
        if comment_count is not None:
            return comment_count

        # Path 5: feedback.comments_count_summary_renderer.feedback.comment_rendering_instance.comments.total_count
        comments_renderer = (
            node.get("feedback", {})
            .get("comments_count_summary_renderer", {})
            .get("feedback", {})
        )
        comment_count = (
            comments_renderer.get("comment_rendering_instance", {})
            .get("comments", {})
            .get("total_count")
        )
        if comment_count is not None:
            return comment_count

        return 0
    except Exception:
        return 0


def is_reel_or_video_post(node):
    """Check if the post is a reel or video post"""
    # Check for reel in story type
    story_type = node.get("__typename", "")
    if "reel" in story_type.lower():
        return True

    # Check if comet_sections has content that indicates reel
    comet_sections = node.get("comet_sections", {})
    content = comet_sections.get("content", {})

    # Check for reel in content typename
    content_typename = content.get("__typename", "")
    if "reel" in content_typename.lower():
        return True

    # Check attachments for video/reel content
    attachments = node.get("attachments") or []
    for att in attachments:
        styles = att.get("styles") or {}
        attachment = styles.get("attachment") or {}

        # Check if it's a video attachment
        single_media = attachment.get("media")
        if single_media:
            media_typename = single_media.get("__typename", "")
            if media_typename == "Video":
                return True
            # Check for reel in typename or anywhere in media object
            if "reel" in str(single_media).lower():
                return True

        # Check in all_subattachments for videos
        all_media = attachment.get("all_subattachments", {}).get("nodes", [])
        for m in all_media:
            media_node = m.get("media") or {}
            if media_node.get("__typename") == "Video":
                return True
            # Check for reel substring
            if "reel" in str(media_node).lower():
                return True

    return False


def extract_media(node, post_id):
    """Extract photo and video URLs from a post node (no downloads)."""
    media = []
    image_count = 0
    last_media_id = None

    attachments = node.get("attachments") or []
    for att in attachments:
        styles = att.get("styles") or {}
        attachment = styles.get("attachment") or {}

        single_media = attachment.get("media")
        if single_media:
            if "photo_image" in single_media:
                image_url = single_media["photo_image"]["uri"]
                last_media_id = single_media.get("id")
                image_count += 1
                media.append({"type": "photo", "url": image_url})
            elif "image" in single_media:
                image_url = single_media["image"]["uri"]
                last_media_id = single_media.get("id")
                image_count += 1
                media.append({"type": "photo", "url": image_url})

            if single_media.get("__typename") == "Video":
                media.append({"type": "video", "url": single_media.get("playable_url")})

        all_media = attachment.get("all_subattachments", {}).get("nodes", [])
        for m in all_media:
            media_node = m.get("media") or {}
            if "image" in media_node:
                image_url = media_node["image"]["uri"]
                last_media_id = media_node.get("id")
                image_count += 1
                media.append({"type": "photo", "url": image_url})
            if media_node.get("__typename") == "Video":
                media.append({"type": "video", "url": media_node.get("playable_url")})

    # Fetch remaining image URLs if there may be more than 5
    if image_count == 5 and last_media_id:
        remaining = fetch_remaining_image_urls(last_media_id, post_id, image_count)
        media.extend(remaining)

    return media


def extract_post_timestamp(node):
    """Extract Unix creation timestamp from a post node."""
    ts = node.get("creation_time")
    if ts:
        return int(ts)
    for item in node.get("comet_sections", {}).get("metadata", []) or []:
        if isinstance(item, dict):
            ts = item.get("story", {}).get("creation_time")
            if ts:
                return int(ts)
    return None


def post_already_exists(post_id, base_folder, name_folder):
    """Check if a post has already been scraped (checks folder existence)."""
    if not post_id or not name_folder:
        return False
    post_dir = os.path.join(base_folder, name_folder, str(post_id))
    return os.path.isdir(post_dir)


def fetch_posts(limit=10, min_comments=0, batch_size=10, on_batch_complete=None):
    """Fetch posts from Facebook page.

    Args:
        limit: Maximum number of posts to fetch.
        min_comments: Minimum comments required (0 = no filter).
        batch_size: Posts per batch before calling on_batch_complete.
        on_batch_complete: Callback(batch_posts, total_so_far, limit).
    """
    global PAGE_NAME
    all_posts = []
    batch_posts = []
    cursor = None
    page_num = 1

    if min_comments > 0:
        print(f"Filtering posts with at least {min_comments} comments")

    while len(all_posts) < limit:
        variables = {
            "count": 3,
            "cursor": cursor,
            "id": USER_ID,
            "feedLocation": "TIMELINE",
            "renderLocation": "timeline",
            "scale": 2,
            "useDefaultActor": False,
        }

        payload = {
            "av": "0",
            "__user": "0",
            "__a": "1",
            "doc_id": DOC_ID,
            "variables": json.dumps(variables),
        }

        # Retry loop for empty response handling
        max_empty_retries = 3
        empty_retry_count = 0
        cleaned_data = []

        while empty_retry_count < max_empty_retries:
            r = retry_request(GRAPHQL_URL, BASE_HEADERS, payload, PROXIES)
            # with open("response.txt", "w", encoding="utf-8") as f:
            #     f.write(r.text)
            print("Status code:", r.status_code)
            cleaned_data = parse_fb_response(r.text)

            if cleaned_data and len(cleaned_data) > 0:
                # Got valid data, break retry loop
                break
            else:
                empty_retry_count += 1
                if empty_retry_count < max_empty_retries:
                    print(
                        f"   Empty response, retrying ({empty_retry_count}/{max_empty_retries})..."
                    )
                    time.sleep(2)  # Wait before retry
                else:
                    print(
                        f"   Empty response after {max_empty_retries} attempts, skipping page"
                    )

        # # Save cleaned data for verification
        # with open(f"cleaned_page_{page_num}.json", "w", encoding="utf-8") as f:
        #     json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        # print(f"Saved cleaned_page_{page_num}.json")

        # If still empty after retries, stop pagination (can't get next cursor from empty response)
        if not cleaned_data or len(cleaned_data) == 0:
            print("   No data received after retries, stopping pagination")
            break

        # Collect all Story nodes from the response
        # Stories can be in two places:
        # 1. Inside timeline_list_feed_units.edges[]
        # 2. As standalone nodes with __typename: "Story"

        story_nodes = []
        timeline_block = None

        for block in cleaned_data:
            if not isinstance(block, dict):
                continue

            node = block.get("node", {})
            node_typename = node.get("__typename")

            # Check if this block has timeline edges
            if "timeline_list_feed_units" in node:
                timeline_block = block
                edges = node["timeline_list_feed_units"].get("edges", [])
                for edge in edges:
                    edge_node = edge.get("node")
                    if edge_node and edge_node.get("__typename") == "Story":
                        story_nodes.append(edge_node)

            # Check if this block itself is a Story node
            elif node_typename == "Story":
                story_nodes.append(node)

            # Check for Story nodes inside Group edges (edge case)
            elif node_typename == "Group":
                edges = node.get("group_feed", {}).get("edges", [])
                for edge in edges:
                    edge_node = edge.get("node", {})
                    if edge_node.get("__typename") == "Story":
                        story_nodes.append(edge_node)

        print(f"Found {len(story_nodes)} posts in page {page_num}")

        # Process all collected Story nodes
        for node in story_nodes:
            if is_reel_or_video_post(node):
                continue

            ts = extract_post_timestamp(node)
            comment_count = extract_comment_count(node)
            if min_comments > 0 and comment_count < min_comments:
                continue

            if not PAGE_NAME:
                PAGE_NAME = extract_page_name(node)
                if PAGE_NAME:
                    print(f"Page name: {PAGE_NAME}")

            post_id = node.get("post_id")
            if not post_id:
                continue

            temp_page_name = PAGE_NAME or extract_page_name(node)
            if temp_page_name:
                temp_name_folder = (
                    "".join(
                        c for c in temp_page_name if c.isalnum() or c in (" ", "-", "_")
                    ).strip()
                    or "Unknown"
                )
                if post_already_exists(post_id, "page_post", temp_name_folder):
                    print(f"  Skipping already scraped post: {post_id}")
                    continue

            feedback_id = node.get("feedback", {}).get("id")
            message = (
                node.get("comet_sections", {})
                .get("content", {})
                .get("story", {})
                .get("message", {})
                .get("text")
            )
            permalink = None
            try:
                permalink = node["attachments"][0]["styles"]["attachment"]["url"]
            except Exception:
                pass

            created_time = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if ts is not None
                else None
            )

            if PAGE_NAME:
                name_folder = (
                    "".join(
                        c for c in PAGE_NAME if c.isalnum() or c in (" ", "-", "_")
                    ).strip()
                    or "Unknown"
                )
            else:
                name_folder = "Unknown"

            post = {
                "post_id": post_id,
                "feedback_id": feedback_id,
                "text": message,
                "permalink": permalink,
                "created_time": created_time,
                "comment_count": comment_count,
                "page_name": PAGE_NAME,
                "media": extract_media(node, post_id),
            }

            batch_posts.append(post)
            all_posts.append(post)

            if batch_size > 0 and len(batch_posts) >= batch_size and on_batch_complete:
                on_batch_complete(batch_posts, len(all_posts), limit)
                batch_posts = []

            if len(all_posts) >= limit:
                break

        # update cursor - get page_info from timeline_block or find it in cleaned_data
        page_info = timeline_block["node"]["timeline_list_feed_units"].get("page_info")

        # If not in timeline_block, search for it in cleaned_data array
        if not page_info:
            for block in cleaned_data:
                if isinstance(block, dict) and "page_info" in block:
                    page_info = block["page_info"]
                    break

        page_info = page_info or {}
        cursor = page_info.get("end_cursor")

        if not cursor:
            print("No more pages. Stopping pagination.")
            break

        time.sleep(1)
        page_num += 1

    # Process any remaining posts in the final batch
    if batch_posts and on_batch_complete:
        on_batch_complete(batch_posts, len(all_posts), limit)

    return all_posts


if __name__ == "__main__":
    count = int(input("How many posts to fetch? "))
    posts = fetch_posts(count)
    with open("posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(posts)} posts to posts.json")
    print(f"Saved {len(posts)} posts to posts.json")
