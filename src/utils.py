import re


def normalize_text(text):
    """Normalize text for deduplication: lowercase, remove punctuation, extra spaces"""
    if not isinstance(text, str):
        return str(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
