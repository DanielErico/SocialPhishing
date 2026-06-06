"""
feature_extractor.py  (backend copy)
=====================================
Extracts 8 numerical features from a raw URL string for the phishing classifier.
This file is a self-contained copy of the module from /model/ so the backend
can be deployed independently without the full training pipeline.

Features extracted
------------------
1.  url_length          -- Total character count of the URL
2.  domain_length       -- Character count of the domain only
3.  subdomain_count     -- Number of subdomains (dots before TLD)
4.  has_https           -- 1 if scheme is HTTPS, else 0
5.  has_ip_address      -- 1 if hostname is a raw IPv4/IPv6 address, else 0
6.  suspicious_keywords -- Count of: login, verify, secure, update, account, bank
7.  special_char_count  -- Count of @, -, _, ~ in the full URL
8.  path_depth          -- Number of non-empty path segments after the domain
"""

import socket
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = ["login", "verify", "secure", "update", "account", "bank"]
SPECIAL_CHARS = ["@", "-", "_", "~"]

# Ordered feature column names -- must match the order used during training
FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "subdomain_count",
    "has_https",
    "has_ip_address",
    "suspicious_keywords",
    "special_char_count",
    "path_depth",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ip_address(hostname: str) -> bool:
    """Return True if hostname is a valid IPv4 or IPv6 address."""
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, hostname)
            return True
        except (socket.error, OSError):
            continue
    return False


def _count_subdomains(hostname: str) -> int:
    """
    Count subdomains in *hostname*.
    Uses a heuristic to detect two-part TLDs (co.uk, com.au, etc.).
    """
    if not hostname or _is_ip_address(hostname):
        return 0
    parts = hostname.split(".")
    if len(parts) <= 2:
        return 0
    # Detect two-part TLD: e.g. 'co' in 'co.uk' has length <= 3
    tld_parts = 3 if len(parts) >= 3 and len(parts[-2]) <= 3 else 2
    return max(0, len(parts) - tld_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract all 8 features from *url* and return them as a dict.

    Tolerant of malformed URLs -- returns all-zero dict on parse failure.
    """
    features = {col: 0 for col in FEATURE_COLUMNS}

    if not url or not isinstance(url, str):
        return features

    url = url.strip()
    features["url_length"] = len(url)

    try:
        parsed = urlparse(url)
    except Exception:
        return features

    hostname = parsed.hostname or ""

    features["domain_length"] = len(hostname)
    features["subdomain_count"] = _count_subdomains(hostname)
    features["has_https"] = 1 if parsed.scheme.lower() == "https" else 0
    features["has_ip_address"] = 1 if _is_ip_address(hostname) else 0

    url_lower = url.lower()
    features["suspicious_keywords"] = sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)
    features["special_char_count"] = sum(url.count(ch) for ch in SPECIAL_CHARS)

    path = parsed.path or ""
    features["path_depth"] = len([seg for seg in path.split("/") if seg])

    return features


def features_to_vector(features: dict) -> list:
    """Return an ordered list of feature values matching FEATURE_COLUMNS."""
    return [features[col] for col in FEATURE_COLUMNS]
