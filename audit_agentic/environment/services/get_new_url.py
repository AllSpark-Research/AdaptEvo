"""Optional deployment-owned media URL resolver; no bundled service credentials."""

import os

import requests


def try_transfer_blocked_url(url):
    endpoint = os.environ.get("AUDIT_MEDIA_RESOLVER_URL", "").strip()
    if not endpoint:
        return url
    headers = {}
    token = os.environ.get("AUDIT_MEDIA_RESOLVER_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(endpoint, json={"url": url}, headers=headers,
                             timeout=10, allow_redirects=False)
    response.raise_for_status()
    return response.json().get("url") or url
