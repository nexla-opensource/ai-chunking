import re
from typing import List


def extract_image_urls(text: str) -> List[str]:
    """Extract markdown image URLs from text"""
    # Match markdown image syntax ![alt](url)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(image_pattern, text)
    return [url for _, url in matches]
