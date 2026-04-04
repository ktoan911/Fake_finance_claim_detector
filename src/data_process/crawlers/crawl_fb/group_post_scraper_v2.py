import json
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

GRAPHQL_URL = "https://www.facebook.com/api/graphql/"

# ========= CONFIG (FILL THESE) =========
GROUP_ID = "363757814515154"  # group id
GROUP_NAME = None  # Will be extracted automatically
DOC_ID = "25716860671307636"  # GroupsCometFeedRegularStoriesPaginationQuery

HEADERS = {
    "user-agent": "Mozilla/5.0",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.facebook.com",
    "referer": f"https://www.facebook.com/groups/{GROUP_ID}/",
}

# Get proxy configuration
PROXY = os.getenv("PROXY")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

if PROXY:
    print(f"Using proxy: {PROXY}")


def extract_group_name(node):
    """Extract group name from post node"""
    try:
        # Try from context_layout > story > comet_sections > title > story > to
        context_layout = node.get("comet_sections", {}).get("context_layout", {})
        story = context_layout.get("story", {})
        title_section = story.get("comet_sections", {}).get("title", {})
        title_story = title_section.get("story", {})
        to_obj = title_story.get("to", {})
        if to_obj.get("__typename") == "Group":
            return to_obj.get("name")

        # Try from content > story > target_group (if available)
        content = node.get("comet_sections", {}).get("content", {})
        content_story = content.get("story", {})
        target_group = content_story.get("target_group", {})
        if target_group and "name" in target_group:
            return target_group.get("name")

        # Try from feedback > associated_group
        feedback = node.get("feedback", {})
        associated_group = feedback.get("associated_group", {})
        if associated_group and "name" in associated_group:
            return associated_group.get("name")

        return None
    except Exception:
        return None


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
            print(f"  ⏳ Retrying in {wait_time} seconds...")
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


def extract_data_blocks(raw_text):
    """Extract all 'data' blocks from raw text"""
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


def clean_data_blocks(blocks):
    """Clean unwanted keys from data blocks"""
    cleaned = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block.pop("errors", None)
        block.pop("extensions", None)

        cleaned.append(block)

    return cleaned


def parse_fb_response(text):
    """Parse Facebook response using the same logic as post_scraper"""
    text = text.replace("for (;;);", "").strip()
    extracted = extract_data_blocks(text)
    cleaned = clean_data_blocks(extracted)
    return cleaned


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

        # Path 6: comet_sections.feedback.story.story_ufi_container.story.feedback_context.feedback_target_with_context.comet_ufi_summary_and_actions_renderer.feedback.comments_count_summary_renderer.feedback.comment_rendering_instance.comments.total_count
        comet_sections = node.get("comet_sections", {})
        feedback_section = comet_sections.get("feedback", {})
        story = feedback_section.get("story", {})
        story_ufi_container = story.get("story_ufi_container", {})
        ufi_story = story_ufi_container.get("story", {})
        feedback_context = ufi_story.get("feedback_context", {})
        feedback_target = feedback_context.get("feedback_target_with_context", {})
        comet_ufi = feedback_target.get(
            "comet_ufi_summary_and_actions_renderer", {}
        ).get("feedback", {})
        comments_count_renderer = comet_ufi.get(
            "comments_count_summary_renderer", {}
        ).get("feedback", {})
        comment_count = (
            comments_count_renderer.get("comment_rendering_instance", {})
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
    if not node or node.get("__typename") != "Story":
        return False

    # Check for reel in story type or anywhere in node
    node_typename = node.get("__typename", "")
    if "reel" in node_typename.lower():
        return True

    # Check comet_sections for reel content
    comet_sections = node.get("comet_sections", {})
    content = comet_sections.get("content", {})

    content_typename = content.get("__typename", "")
    if "reel" in content_typename.lower():
        return True

    # Check attachments for video/reel content
    attachments = node.get("attachments", [])
    for attachment in attachments:
        # Check for video media type
        if "media" in attachment and attachment["media"].get("__typename") == "Video":
            return True

        # Check for reel substring in media object
        if "media" in attachment and "reel" in str(attachment["media"]).lower():
            return True

        # Check in styles > attachment > media for video or reel
        styles_media = (
            attachment.get("styles", {}).get("attachment", {}).get("media", {})
        )
        if styles_media.get("__typename") == "Video":
            return True
        if "reel" in str(styles_media).lower():
            return True

        # Check all_subattachments for videos or reels
        for subattachment in attachment.get("all_subattachments", {}).get("nodes", []):
            if (
                "media" in subattachment
                and subattachment["media"].get("__typename") == "Video"
            ):
                return True
            if (
                "media" in subattachment
                and "reel" in str(subattachment["media"]).lower()
            ):
                return True

    return False


def extract_media(node, post_id):
    """Extract photo and video URLs from a post node (no downloads)."""
    photos = []
    videos = []
    image_index = 0
    last_media_id = None

    attachments = node.get("attachments", [])

    for attachment in attachments:
        if "media" in attachment and attachment["media"].get("__typename") == "Photo":
            photo_data = (
                attachment.get("styles", {}).get("attachment", {}).get("media", {})
            )
            if "photo_image" in photo_data:
                image_index += 1
                last_media_id = attachment["media"].get("id")
                image_url = photo_data["photo_image"].get("uri")
                photos.append(
                    {
                        "id": last_media_id,
                        "url": image_url,
                        "width": photo_data["photo_image"].get("width"),
                        "height": photo_data["photo_image"].get("height"),
                    }
                )

        if "all_subattachments" in attachment:
            for subattachment in attachment.get("all_subattachments", {}).get(
                "nodes", []
            ):
                if (
                    "media" in subattachment
                    and subattachment["media"].get("__typename") == "Photo"
                ):
                    image_index += 1
                    photo_data = subattachment.get("media", {})
                    last_media_id = photo_data.get("id")
                    if "image" in photo_data:
                        image_url = photo_data["image"].get("uri")
                        photos.append(
                            {
                                "id": last_media_id,
                                "url": image_url,
                                "width": photo_data["image"].get("width"),
                                "height": photo_data["image"].get("height"),
                            }
                        )

        if "media" in attachment and attachment["media"].get("__typename") == "Video":
            video_data = attachment.get("media", {})
            videos.append(
                {
                    "id": video_data.get("id"),
                    "url": video_data.get("playable_url"),
                    "thumbnail": video_data.get("preferred_thumbnail", {})
                    .get("image", {})
                    .get("uri"),
                }
            )

    if image_index == 5 and last_media_id:
        for extra in fetch_remaining_image_urls(last_media_id, post_id, image_index):
            photos.append({"id": None, "url": extra["url"]})

    return {"photos": photos, "videos": videos}


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


def extract_post_data(node, group_name=None):
    """Extract relevant data from a post node."""
    if not node or node.get("__typename") != "Story":
        return None

    content_story = node.get("comet_sections", {}).get("content", {}).get("story", {})
    message = (content_story.get("message") or {}).get("text", "")

    post_id = node.get("post_id")
    if not post_id:
        return None

    comment_count = extract_comment_count(node)

    if not group_name:
        group_name = extract_group_name(node)

    if group_name:
        name_folder = (
            "".join(
                c for c in group_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            or "Unknown"
        )
    else:
        name_folder = "Unknown"

    ts = extract_post_timestamp(node)
    created_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else None

    media = extract_media(node, post_id)
    post_data = {
        "post_id": post_id,
        "message": message,
        "comment_count": comment_count,
        "group_name": group_name,
        "permalink": node.get("permalink_url", ""),
        "created_time": created_time,
        "photos": media["photos"],
        "videos": media["videos"],
    }

    return post_data


def fetch_posts(limit=10, min_comments=0, batch_size=10, on_batch_complete=None):
    """Fetch posts from Facebook group.

    Args:
        limit: Maximum number of posts to fetch.
        min_comments: Minimum comments required (0 = no filter).
        batch_size: Posts per batch before calling on_batch_complete.
        on_batch_complete: Callback(batch_posts, total_so_far, limit).
    """
    global GROUP_NAME
    all_posts = []
    batch_posts = []
    cursor = None
    page_num = 1

    while len(all_posts) < limit:
        print(f"\nFetching page {page_num}...")

        variables = {
            "count": 3,
            "cursor": cursor,
            "feedLocation": "GROUP",
            "feedType": "DISCUSSION",
            "feedbackSource": 0,
            "filterTopicId": None,
            "focusCommentID": None,
            "privacySelectorRenderLocation": "COMET_STREAM",
            "renderLocation": "group",
            "scale": 2,
            "stream_initial_count": 1,
            "useDefaultActor": False,
            "id": GROUP_ID,
        }

        payload = {
            "av": "0",
            "__user": "0",
            "__a": "1",
            "doc_id": DOC_ID,
            "variables": json.dumps(variables),
        }

        max_empty_retries = 3
        empty_retry_count = 0
        data = []

        while empty_retry_count < max_empty_retries:
            try:
                r = retry_request(GRAPHQL_URL, HEADERS, payload, PROXIES)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                break

            data = parse_fb_response(r.text)

            if data and len(data) > 0:
                break
            else:
                empty_retry_count += 1
                if empty_retry_count < max_empty_retries:
                    print(
                        f"  Empty response, retrying ({empty_retry_count}/{max_empty_retries})..."
                    )
                    time.sleep(2)
                else:
                    print(
                        f"  Empty response after {max_empty_retries} attempts, skipping page"
                    )

        if not data or len(data) == 0:
            print("No data received after retries, stopping pagination")
            break

        posts_found = 0
        next_cursor = None

        for item in data:
            if not isinstance(item, dict):
                continue

            node = item.get("node", {})
            node_typename = node.get("__typename")

            story_nodes = []
            if node_typename == "Story":
                story_nodes.append(node)
            elif node_typename == "Group":
                edges = node.get("group_feed", {}).get("edges", [])
                for edge in edges:
                    edge_node = edge.get("node", {})
                    if edge_node.get("__typename") == "Story":
                        story_nodes.append(edge_node)

            for story_node in story_nodes:
                if is_reel_or_video_post(story_node):
                    continue

                comment_count = extract_comment_count(story_node)
                if min_comments > 0 and comment_count < min_comments:
                    continue

                if not GROUP_NAME:
                    GROUP_NAME = extract_group_name(story_node)
                    if GROUP_NAME:
                        print(f"Group name: {GROUP_NAME}")

                temp_post_id = story_node.get("post_id")
                temp_group_name = GROUP_NAME or extract_group_name(story_node)
                if temp_group_name:
                    temp_name_folder = (
                        "".join(
                            c
                            for c in temp_group_name
                            if c.isalnum() or c in (" ", "-", "_")
                        ).strip()
                        or "Unknown"
                    )
                    if post_already_exists(
                        temp_post_id, "group_post", temp_name_folder
                    ):
                        print(f"  Skipping already scraped post: {temp_post_id}")
                        continue

                post_data = extract_post_data(story_node, GROUP_NAME)
                if post_data:
                    batch_posts.append(post_data)
                    all_posts.append(post_data)
                    posts_found += 1
                    print(f"  Found post: {post_data['post_id']}")

                    if (
                        batch_size > 0
                        and len(batch_posts) >= batch_size
                        and on_batch_complete
                    ):
                        on_batch_complete(batch_posts, len(all_posts), limit)
                        batch_posts = []

                    if len(all_posts) >= limit:
                        break

            if len(all_posts) >= limit:
                break

            if "page_info" in item:
                page_info = item["page_info"]
                if page_info.get("has_next_page"):
                    next_cursor = page_info.get("end_cursor")

        print(f"Found {posts_found} posts on this page")

        if not next_cursor or len(all_posts) >= limit:
            print("No more pages or reached limit. Stopping.")
            break

        cursor = next_cursor
        page_num += 1
        time.sleep(2)

    if batch_posts and on_batch_complete:
        on_batch_complete(batch_posts, len(all_posts), limit)

    return all_posts


if __name__ == "__main__":
    count = int(input("How many posts to fetch? "))
    print(f"\nFetching {count} posts from group {GROUP_ID}...")
    posts = fetch_posts(count)
    with open("group_posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(posts)} posts to group_posts.json")
