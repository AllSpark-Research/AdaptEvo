"""图片 URL → 本地 cache 路径转换（所有工具共用）。"""

import hashlib
import os
from typing import Any, Dict, List


def url_to_cache_path(url: str, cache_dir: str) -> str:
    """URL → 本地路径: {cache_dir}/{md5[:2]}/{md5[2:4]}/{md5}.jpg"""
    base = url.split("?")[0]
    h = hashlib.md5(base.encode()).hexdigest()
    return os.path.join(cache_dir, h[:2], h[2:4], f"{h}.jpg")


def replace_urls_with_cache(images: list, cache_dir: str) -> list:
    """仅将已有本地副本的 URL 替换为 cache 路径。"""
    if not images or not cache_dir:
        return images
    converted = []
    for url in images:
        if isinstance(url, str) and url.startswith("http"):
            path = url_to_cache_path(url, cache_dir)
            converted.append(path if os.path.exists(path) else url)
        else:
            converted.append(url)
    return converted


def get_image_cache_dir(args: Dict[str, Any]) -> str:
    """从 args 或环境变量获取 image_cache_dir。"""
    from ..data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return ""
    return args.get("_image_cache_dir") or os.environ.get("IMAGE_CACHE_DIR", "")


def apply_image_cache(result: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """对工具返回结果统一做 URL → 本地路径替换。"""
    cache_dir = get_image_cache_dir(args)
    if not cache_dir:
        return result
    for key in ("display_images", "all_images", "images"):
        if key in result and isinstance(result[key], list):
            result[key] = replace_urls_with_cache(result[key], cache_dir)
    return result
