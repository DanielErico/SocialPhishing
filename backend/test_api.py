import time, urllib.request, json, sys

time.sleep(3)
base = "http://localhost:8000"

def post_classify(url, html_features=None):
    payload = {"url": url}
    if html_features:
        payload["html_features"] = html_features
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/classify", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def get_endpoint(path):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as r:
        return json.loads(r.read())

# Health check
print("--- GET /health ---")
print(json.dumps(get_endpoint("/health"), indent=2))

tests = [
    ("https://www.google.com", None, "Expected: Safe (Whitelisted)"),
    ("https://amazon.co.uk/dp/B08H93ZRK9", None, "Expected: Safe (Whitelisted)"),
    
    # Standard phishing URL (with forms)
    ("http://192.168.1.1/login/account/verify", 
     {"has_password_input": 1, "has_form": 1, "external_links_ratio": 0.05, "brand_mismatch": 0}, 
     "Expected: Phishing"),
     
    # High-threat phishing URL (with forms and brand spoofing)
    ("https://secure-login.bank-verify.attacker.com/update", 
     {"has_password_input": 1, "has_form": 1, "external_links_ratio": 0.75, "brand_mismatch": 1}, 
     "Expected: Phishing"),
     
    # Test Decision Fusion: Phishing-looking URL but has NO forms (should be demoted)
    ("https://secure-login.bank-verify.attacker.com/update", 
     {"has_password_input": 0, "has_form": 0, "external_links_ratio": 0.05, "brand_mismatch": 0}, 
     "Expected: Safe/Suspicious (Demoted due to no forms)"),
     
    # Test Decision Fusion: Random URL with login fields and brand mismatch (should be promoted to Phishing)
    ("https://some-unknown-registration-portal.org/home", 
     {"has_password_input": 1, "has_form": 1, "external_links_ratio": 0.85, "brand_mismatch": 1}, 
     "Expected: Phishing (Promoted due to brand mismatch + password fields)")
]

print()
for url, html_feats, note in tests:
    r = post_classify(url, html_feats)
    label = r["label"]
    conf = r["confidence"]
    p_ph = r["phishing_probability"]
    lat = r["latency_ms"]
    print(note)
    print("  URL     : " + url[:70])
    print("  Features: " + str(html_feats))
    print("  Label   : " + label + "  (conf=" + str(conf) + ", p_phishing=" + str(p_ph) + ")")
    print("  Latency : " + str(lat) + "ms")
    print()
