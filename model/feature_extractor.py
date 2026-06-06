"""
feature_extractor.py
====================
Extracts a fixed set of 8 numerical features from a raw URL string.

Features
--------
1.  url_length          – Total character count of the URL
2.  domain_length       – Character count of the domain only
3.  subdomain_count     – Number of subdomains (dots minus TLD separator)
4.  has_https           – 1 if scheme is HTTPS, else 0
5.  has_ip_address      – 1 if the hostname is an IPv4/IPv6 address, else 0
6.  suspicious_keywords – Count of suspicious keyword occurrences in the URL
7.  special_char_count  – Count of @, -, _, ~ characters in the full URL
8.  path_depth          – Number of non-empty path segments after the domain

Usage
-----
    from feature_extractor import extract_features
    features = extract_features("https://login-verify.suspicious.com/secure/account")
    # Returns a dict of the 8 features above
"""

import re
import socket
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords that commonly appear in phishing URLs
SUSPICIOUS_KEYWORDS = ["login", "verify", "secure", "update", "account", "bank"]

# Special characters frequently abused in obfuscated phishing URLs
SPECIAL_CHARS = ["@", "-", "_", "~"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_ip_address(hostname: str) -> bool:
    """
    Returns True if *hostname* is a valid IPv4 or IPv6 address.
    Uses socket.inet_pton which handles both address families correctly.
    """
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, hostname)
            return True
        except (socket.error, OSError):
            continue
    return False


def _count_subdomains(hostname: str) -> int:
    """
    Counts the number of subdomains in *hostname*.

    Strategy: split the hostname by '.' and subtract the root domain parts.
    e.g.  "login.mail.example.co.uk" → ["login", "mail", "example", "co", "uk"]
    Known two-part TLDs (co.uk, com.au, …) have 2 TLD parts, so subdomain_count = 5 - 2 - 1 = 2.

    For simplicity we assume:
      - Single-part TLD  (e.g. .com)  → parts - 2  (TLD + root domain)
      - Two-part TLD     (e.g. .co.uk)→ parts - 3
    We detect a two-part TLD heuristically: if second-to-last part len ≤ 3.
    """
    if not hostname or _is_ip_address(hostname):
        return 0

    parts = hostname.split(".")
    if len(parts) <= 2:
        # Just "domain.tld" — no subdomains
        return 0

    # Heuristic: detect two-part TLD like co.uk, com.au, org.uk
    if len(parts) >= 3 and len(parts[-2]) <= 3:
        tld_parts = 3  # TLD has 2 parts
    else:
        tld_parts = 2  # Standard single-part TLD

    subdomain_count = max(0, len(parts) - tld_parts)
    return subdomain_count


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract the 8 URL features used by the phishing classifier.

    Parameters
    ----------
    url : str
        The raw URL string to analyse. Must be a non-empty string.

    Returns
    -------
    dict
        A dictionary with keys:
        url_length, domain_length, subdomain_count, has_https,
        has_ip_address, suspicious_keywords, special_char_count, path_depth

    Notes
    -----
    - All counts are clamped to 0 minimum to avoid negative values on
      edge-case URLs (e.g. data: URIs, blank hostnames).
    - The function is intentionally tolerant of malformed URLs and will
      return a feature dict of all zeros rather than raising an exception.
    """
    # Initialise all features to 0 (safe defaults)
    features = {
        "url_length": 0,
        "domain_length": 0,
        "subdomain_count": 0,
        "has_https": 0,
        "has_ip_address": 0,
        "suspicious_keywords": 0,
        "special_char_count": 0,
        "path_depth": 0,
    }

    if not url or not isinstance(url, str):
        return features

    url = url.strip()

    # 1. url_length — total character count
    features["url_length"] = len(url)

    try:
        parsed = urlparse(url)
    except Exception:
        return features

    # Extract hostname (netloc may contain port; strip it)
    hostname = parsed.hostname or ""  # urlparse normalises to lowercase

    # 2. domain_length — length of just the hostname portion
    features["domain_length"] = len(hostname)

    # 3. subdomain_count
    features["subdomain_count"] = _count_subdomains(hostname)

    # 4. has_https — 1 if scheme is exactly "https"
    features["has_https"] = 1 if parsed.scheme.lower() == "https" else 0

    # 5. has_ip_address — 1 if hostname resolves to a raw IP
    features["has_ip_address"] = 1 if _is_ip_address(hostname) else 0

    # 6. suspicious_keywords — count of keyword hits in the full lowercased URL
    url_lower = url.lower()
    features["suspicious_keywords"] = sum(
        url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS
    )

    # 7. special_char_count — count of @, -, _, ~ across the entire URL
    features["special_char_count"] = sum(url.count(ch) for ch in SPECIAL_CHARS)

    # 8. path_depth — number of non-empty path segments
    path = parsed.path or ""
    features["path_depth"] = len([seg for seg in path.split("/") if seg])

    return features


# ---------------------------------------------------------------------------
# Feature vector ordering (matches training column order exactly)
# ---------------------------------------------------------------------------

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


def features_to_vector(features: dict) -> list:
    """
    Convert a features dict to an ordered list matching FEATURE_COLUMNS.
    Use this when building the numpy array for model.predict().
    """
    return [features[col] for col in FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Quick smoke-test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("https://www.google.com", "Expected: Safe"),
        ("http://192.168.1.1/login/account/verify", "Expected: Phishing"),
        ("https://secure-login.bank-verify.suspicious.example.com/update/account", "Expected: Phishing"),
        ("https://amazon.co.uk/dp/B08H93ZRK9", "Expected: Safe"),
        ("http://verify-your-account@phishsite.ru/secure/login.php", "Expected: Phishing"),
    ]

    print(f"{'URL':<65} {'Features'}")
    print("-" * 120)
    for url, note in test_cases:
        f = extract_features(url)
        print(f"{url[:64]:<65} {note}")
        for k, v in f.items():
            print(f"    {k:<25}: {v}")
        print()
