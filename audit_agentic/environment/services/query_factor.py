"""Deployment-owned evidence service adapter without embedded authentication."""

import os

import requests


def query_factor(factor_id: str, params: dict, fetch_field: str = None,
                 session: requests.Session = None) -> dict:
    endpoint = os.environ.get("AUDIT_FACTOR_SERVICE_URL", "").strip()
    if not endpoint:
        raise RuntimeError("Configure AUDIT_FACTOR_SERVICE_URL or supply cached evidence")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("AUDIT_FACTOR_SERVICE_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"factor_id": factor_id, "params": params}
    if fetch_field:
        payload["fetch_field"] = fetch_field
    response = (session or requests).post(endpoint, json=payload, headers=headers,
                                         timeout=30, allow_redirects=False)
    response.raise_for_status()
    return response.json()


def extract_result(api_response: dict):
    return (api_response.get("data") or {}).get("result")
