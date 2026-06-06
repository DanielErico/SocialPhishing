"""
dataset_loader.py
=================
Loads and merges URL datasets into a single, balanced Pandas DataFrame
ready for feature extraction and model training.

Sources
-------
1. PhishTank  — verified_online.csv  (phishing URLs, label = "Phishing")
2. Safe URLs  — Synthetically generated from Alexa/Tranco-style known-good
                domains.  A curated list of ~500 real-world safe domains is
                used to construct realistic safe URLs with random paths.

Output
------
A DataFrame with exactly two columns:
    url   : str  – the raw URL string
    label : str  – one of  "Phishing", "Suspicious", "Safe"

Label mapping
-------------
All PhishTank records  → "Phishing"
All generated safe URLs → "Safe"
"Suspicious" is NOT present in the training set at this stage; the backend
assigns "Suspicious" when the model confidence falls in the mid-range
(0.40–0.70). This is a post-processing label applied at inference time.

Usage
-----
    from dataset_loader import load_dataset
    df = load_dataset()
    print(df['label'].value_counts())
"""

import os
import random
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit these paths if needed
# ---------------------------------------------------------------------------

# Absolute path to the PhishTank CSV file provided by the user
PHISHTANK_CSV_PATH = r"C:\Users\DANNY\Documents\SocialFish\Phishtank CSV\verified_online.csv"

# Reproducibility seed (must match train_model.py)
RANDOM_SEED = 42

# Number of safe URLs to generate (aim to balance the classes)
# We will match the number of phishing URLs (capped at 60 000)
MAX_PHISHING_ROWS = 60_000


# ---------------------------------------------------------------------------
# Safe URL generation helpers
# ---------------------------------------------------------------------------

# A representative sample of well-known, legitimate domains across categories
SAFE_DOMAINS = [
    # Search & portals
    "google.com", "google.co.uk", "bing.com", "yahoo.com", "duckduckgo.com",
    # E-commerce
    "amazon.com", "amazon.co.uk", "ebay.com", "etsy.com", "shopify.com",
    "bestbuy.com", "walmart.com", "target.com", "aliexpress.com",
    # Social media
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
    "reddit.com", "pinterest.com", "tiktok.com", "snapchat.com",
    # Technology
    "github.com", "stackoverflow.com", "microsoft.com", "apple.com",
    "developer.apple.com", "docs.microsoft.com", "cloud.google.com",
    "aws.amazon.com", "azure.microsoft.com", "python.org", "npmjs.com",
    # News
    "bbc.co.uk", "bbc.com", "cnn.com", "theguardian.com", "nytimes.com",
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "economist.com",
    # Finance & banking (legitimate)
    "hsbc.co.uk", "barclays.co.uk", "lloydsbank.com", "natwest.com",
    "paypal.com", "wise.com", "monzo.com", "revolut.com", "chase.com",
    "citibank.com", "bankofamerica.com", "wellsfargo.com",
    # Government & education
    "gov.uk", "usa.gov", "irs.gov", "nhs.uk", "edu.au",
    "mit.edu", "stanford.edu", "ox.ac.uk", "cam.ac.uk", "ucl.ac.uk",
    # Streaming & media
    "netflix.com", "youtube.com", "spotify.com", "twitch.tv",
    "hulu.com", "disneyplus.com", "primevideo.com",
    # Productivity & cloud
    "docs.google.com", "drive.google.com", "office.com", "outlook.com",
    "notion.so", "trello.com", "slack.com", "zoom.us", "teams.microsoft.com",
    # Other
    "wikipedia.org", "wikimedia.org", "archive.org", "cloudflare.com",
    "medium.com", "substack.com", "wordpress.com",
    # Domains with hyphens (to prevent hyphen bias in SVM)
    "t-mobile.com", "coca-cola.com", "scikit-learn.org", "rolls-royce.com",
    "merriam-webster.com", "daily-mail.co.uk", "google-groups.com",
    "github-pages.com", "web-hosting.com", "co-op.co.uk",
    # Long domains (to prevent domain length bias in SVM/RF)
    "nationalgeographic.com", "entertainmentweekly.com",
    "constructionequipment.com", "interactivemetaphysics.com",
    "californiastateuniversity.edu", "d-internconnect.com",
    "in-hub.d-internconnect.com"
]

# Realistic path components found on legitimate websites
SAFE_PATH_SEGMENTS = [
    "", "about", "contact", "products", "services", "blog", "news",
    "help", "faq", "support", "pricing", "features", "docs", "api",
    "en-us", "en-gb", "home", "shop", "cart", "search", "category",
    "2024", "2025", "articles", "resources", "download", "releases",
    "dp/B08XYZ1234", "issues/123", "pull/456", "wiki/Main_Page",
    "s?q=python+tutorial", "watch?v=dQw4w9WgXcQ",
]

# Query string fragments used on real sites
SAFE_QUERY_STRINGS = [
    "", "", "", "",            # Weighted towards no query string
    "?ref=homepage", "?source=nav", "?hl=en", "?lang=en",
    "?page=1", "?sort=newest", "?category=tech",
]


# Realistic subdomain prefixes for legitimate sites
SUBDOMAIN_PREFIXES = [
    "mail", "blog", "news", "help", "support", "status", "api", "dev", 
    "test", "portal", "hub", "connect", "internal", "register", "admin",
    "docs", "drive", "my", "login", "secure"
]

def _generate_safe_url(rng: random.Random) -> str:
    """Generate a plausible safe URL from the curated domain/path lists."""
    scheme = "https"  # Legitimate sites use HTTPS
    domain = rng.choice(SAFE_DOMAINS)
    
    # 30% chance to add 1-2 realistic subdomains (possibly hyphenated)
    if rng.random() < 0.30:
        num_subs = rng.randint(1, 2)
        subs = rng.choices(SUBDOMAIN_PREFIXES, k=num_subs)
        sub_str = ""
        for s in subs:
            if sub_str:
                # Randomly join subdomains with dot or hyphen
                sub_str += rng.choice([".", "-"])
            sub_str += s
        domain = sub_str + "." + domain
    else:
        # Add www. prefix to 70% of safe URLs to make subdomain distribution realistic
        if rng.random() < 0.70:
            domain = "www." + domain

    # Build a 0-3 segment path
    depth = rng.randint(0, 3)
    segments = rng.choices(SAFE_PATH_SEGMENTS, k=depth)
    # Filter empty segments (they were included to add weight to short paths)
    segments = [s for s in segments if s]
    path = "/" + "/".join(segments) if segments else ""
    query = rng.choice(SAFE_QUERY_STRINGS)
    return f"{scheme}://{domain}{path}{query}"


# ---------------------------------------------------------------------------
# PhishTank loader
# ---------------------------------------------------------------------------

def _load_phishtank(csv_path: str, max_rows: int) -> pd.DataFrame:
    """
    Load the PhishTank verified_online.csv file.

    Relevant columns:
        url   – the phishing URL
        verified – should be "yes" for all rows in verified_online.csv
        online   – "yes" means still live (we keep all regardless)

    Returns a DataFrame with columns [url, label] where label = "Phishing".
    """
    print(f"[dataset_loader] Loading PhishTank data from:\n  {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"PhishTank CSV not found at: {csv_path}\n"
            "Please update PHISHTANK_CSV_PATH in dataset_loader.py."
        )

    df = pd.read_csv(
        csv_path,
        usecols=["url"],          # Only the URL column is needed
        dtype={"url": str},
        encoding="utf-8",
    )

    # Drop rows with missing or blank URLs
    df = df.dropna(subset=["url"])
    df = df[df["url"].str.strip() != ""]

    # Deduplicate
    df = df.drop_duplicates(subset=["url"])

    # Cap to avoid excessively large training sets
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_SEED)

    df = df.reset_index(drop=True)
    df["label"] = "Phishing"
    print(f"[dataset_loader] PhishTank: {len(df):,} phishing URLs loaded.")
    return df[["url", "label"]]


# ---------------------------------------------------------------------------
# Safe URL generation
# ---------------------------------------------------------------------------

def _generate_safe_urls(n: int) -> pd.DataFrame:
    """
    Generate *n* synthetic safe URLs from the curated domain list.

    Returns a DataFrame with columns [url, label] where label = "Safe".
    """
    print(f"[dataset_loader] Generating {n:,} synthetic safe URLs …")
    rng = random.Random(RANDOM_SEED)
    urls = [_generate_safe_url(rng) for _ in range(n)]
    df = pd.DataFrame({"url": urls, "label": "Safe"})
    print(f"[dataset_loader] Safe URLs generated: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    phishtank_path: str = PHISHTANK_CSV_PATH,
    max_phishing: int = MAX_PHISHING_ROWS,
) -> pd.DataFrame:
    """
    Load and merge all URL sources into a single, balanced DataFrame.

    Parameters
    ----------
    phishtank_path : str
        Path to the PhishTank verified_online.csv file.
    max_phishing : int
        Maximum number of phishing rows to include (caps very large CSVs).

    Returns
    -------
    pd.DataFrame
        Columns: ['url', 'label']
        Labels:  'Phishing', 'Safe'
        The DataFrame is shuffled with random_state=RANDOM_SEED.
    """
    # 1. Load phishing URLs
    phishing_df = _load_phishtank(phishtank_path, max_phishing)

    # 2. Generate matching number of safe URLs for class balance
    safe_df = _generate_safe_urls(len(phishing_df))

    # 3. Combine and shuffle
    combined = pd.concat([phishing_df, safe_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n[dataset_loader] Final dataset: {len(combined):,} total rows")
    print(combined["label"].value_counts().to_string())
    print()
    return combined


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_dataset()
    print("\nSample rows:")
    print(df.sample(10, random_state=42)[["url", "label"]].to_string(index=False))
