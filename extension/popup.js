/**
 * popup.js — Extension Popup Logic
 * =================================
 * Retrieves the current active tab's URL, injects a script to extract HTML features,
 * sends them to the background service worker, and updates the popup UI with an
 * interactive verdict, reasoning card, and security analysis grid.
 */

"use strict";

// Helper — map classification label to CSS classes, badge characters, and summaries
const STATE_MAP = {
  Safe: {
    cardClass: "conclusion-card-safe",
    badge: "S",
    title: "Safe Website",
    desc: (pct) => `SocialPhishing verified this URL as safe (${pct}% confidence).`
  },
  Suspicious: {
    cardClass: "conclusion-card-suspicious",
    badge: "!",
    title: "Suspicious Site",
    desc: (pct) => `Caution: URL patterns show potential phishing signals (${pct}% confidence).`
  },
  Phishing: {
    cardClass: "conclusion-card-phishing",
    badge: "P",
    title: "Phishing Threat!",
    desc: (pct) => `Danger: URL is highly likely designed to steal credentials (${pct}% confidence).`
  }
};

// Evaluate individual feature threat level for visual analysis grid
function evaluateFeature(name, value) {
  let status = "safe"; // "safe" | "suspicious" | "phishing"
  let description = "";

  switch (name) {
    case "has_https":
      status = value === 1 ? "safe" : "phishing";
      description = value === 1 ? "Site uses secure HTTPS encryption" : "Unencrypted connection (HTTP)";
      break;
    case "has_ip_address":
      status = value === 1 ? "phishing" : "safe";
      description = value === 1 ? "Uses raw IP instead of domain name" : "Uses normal domain name resolution";
      break;
    case "url_length":
      if (value < 75) {
        status = "safe";
        description = "URL length is standard and compact";
      } else if (value < 120) {
        status = "suspicious";
        description = "Moderately long URL (possible padding)";
      } else {
        status = "phishing";
        description = "Suspiciously long URL (obfuscation signal)";
      }
      break;
    case "domain_length":
      if (value < 30) {
        status = "safe";
        description = "Domain length is normal";
      } else {
        status = "suspicious";
        description = "Domain name is long (possible typo-squatting)";
      }
      break;
    case "subdomain_count":
      if (value <= 1) {
        status = "safe";
        description = "Standard subdomain structure";
      } else if (value === 2) {
        status = "suspicious";
        description = "Multiple subdomains present";
      } else {
        status = "phishing";
        description = "Excessive subdomains (common phishing indicator)";
      }
      break;
    case "suspicious_keywords":
      status = value === 0 ? "safe" : "phishing";
      description = value === 0 
        ? "No phishing terms (login, bank, secure) found" 
        : `Contains ${value} suspicious keyword(s)`;
      break;
    case "special_char_count":
      if (value === 0) {
        status = "safe";
        description = "No separator characters";
      } else if (value <= 2) {
        status = "suspicious";
        description = "Some special characters (@, -, _, ~)";
      } else {
        status = "phishing";
        description = "High density of separators or @ symbol";
      }
      break;
    case "path_depth":
      if (value < 4) {
        status = "safe";
        description = "Standard folder depth";
      } else {
        status = "suspicious";
        description = "Unusually deep sub-directory structure";
      }
      break;
  }

  return { status, description };
}

// Generate human-readable summary of classification factors
function generateReasoning(label, urlFeatures, htmlFeatures) {
  const reasons = [];

  // 1. URL Analysis
  if (urlFeatures.has_https === 0) {
    reasons.push("Uses an unencrypted <strong>HTTP</strong> connection, which is insecure.");
  }
  if (urlFeatures.has_ip_address === 1) {
    reasons.push("Uses a raw <strong>IP address</strong> instead of a domain name.");
  }
  if (urlFeatures.suspicious_keywords > 0) {
    reasons.push("Contains keywords like 'login' or 'verify' commonly used for brand spoofing.");
  }
  if (urlFeatures.subdomain_count >= 3) {
    reasons.push(`Contains an unusually high subdomain count (${urlFeatures.subdomain_count}) to obscure the true domain.`);
  }
  if (urlFeatures.special_char_count >= 3) {
    reasons.push("High density of special symbols (like hyphens) resembling fake typo-squatted domains.");
  }

  // 2. HTML Content Analysis
  if (htmlFeatures) {
    if (htmlFeatures.brand_mismatch === 1 && htmlFeatures.has_password_input === 1) {
      reasons.push("🚨 <strong>Critical</strong>: Page requests passwords and claims to be a popular brand, but is hosted on a mismatching domain!");
    } else if (htmlFeatures.has_password_input === 1) {
      reasons.push("🔑 Page contains a **password entry field**, demanding credentials. Ensure you trust the address.");
    }
    
    if (htmlFeatures.has_password_input === 0 && htmlFeatures.has_form === 0) {
      reasons.push("ℹ️ Page has **no inputs or form fields** (cannot collect passwords or log in).");
    }
    
    if (htmlFeatures.external_links_ratio > 0.60) {
      reasons.push(`Loads a high percentage of external assets (${(htmlFeatures.external_links_ratio * 100).toFixed(0)}%), typical of cloned phishing pages.`);
    }
  }

  if (label === "Safe") {
    if (reasons.length === 0) {
      return "<strong>Verification Summary:</strong> The website conforms to standard safe templates. It uses HTTPS encryption, contains no brand mismatches, has no login inputs, and uses clean domain names.";
    } else {
      // Return safe verdict but list any warnings
      const nonCriticalReasons = reasons.filter(r => !r.includes("Critical"));
      return `<strong>Verification Summary:</strong> The site matches safe parameters overall:
              <ul>${nonCriticalReasons.map(r => `<li>${r}</li>`).join("")}</ul>`;
    }
  } else {
    // Phishing or Suspicious
    const header = label === "Phishing"
      ? "<strong>Threat Reason Summary:</strong> Flagged due to patterns strongly associated with phishing sites:"
      : "<strong>Suspicious Signals:</strong> Caution advised. Flagged due to the following anomalies:";

    if (reasons.length > 0) {
      return `${header}<ul>${reasons.map(r => `<li>${r}</li>`).join("")}</ul>`;
    } else {
      // Fallback
      return `${header} Although individual indicators look normal, the combination of a custom subdomain, domain length, and minor special characters closely matches templates of phishing portals.`;
    }
  }
}

// Map technical feature names to human-readable labels
const FEATURE_LABELS = {
  url_length: "URL Length",
  domain_length: "Domain Name Length",
  subdomain_count: "Subdomains",
  has_https: "HTTPS Protocol",
  has_ip_address: "Raw IP Address",
  suspicious_keywords: "Sensitive Keywords",
  special_char_count: "Special Symbols",
  path_depth: "URL Path Depth"
};

// Extract HTML features from the tab's DOM via scripting injection
async function getTabHtmlFeatures(tabId) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: () => {
        const hasPasswordInput = document.querySelector('input[type="password"]') ? 1 : 0;
        const hasForm = document.querySelector('form') ? 1 : 0;
        
        const currentHost = window.location.hostname.toLowerCase();
        let totalLinks = 0;
        let externalLinks = 0;
        
        const elements = document.querySelectorAll('a, link, script, img');
        elements.forEach(el => {
          const srcOrHref = el.href || el.src;
          if (srcOrHref) {
            try {
              const url = new URL(srcOrHref, window.location.href);
              if (url.protocol.startsWith('http')) {
                totalLinks++;
                const linkHost = url.hostname.toLowerCase();
                if (linkHost !== currentHost && !linkHost.endsWith('.' + currentHost)) {
                  externalLinks++;
                }
              }
            } catch (_) {}
          }
        });
        
        const externalLinksRatio = totalLinks > 0 ? parseFloat((externalLinks / totalLinks).toFixed(4)) : 0.0;
        
        const title = (document.title || "").toLowerCase();
        const brands = ["paypal", "google", "microsoft", "netflix", "bankofamerica", "chase", "hsbc", "amazon", "facebook", "twitter", "linkedin", "apple"];
        let hasBrandInTitle = false;
        for (const b of brands) {
          if (title.includes(b)) {
            hasBrandInTitle = true;
            break;
          }
        }
        
        let brandMismatch = 0;
        if (hasBrandInTitle) {
          const hasBrandInHost = brands.some(b => currentHost.includes(b));
          if (!hasBrandInHost) {
            brandMismatch = 1;
          }
        }
        
        return {
          has_password_input: hasPasswordInput,
          has_form: hasForm,
          external_links_ratio: externalLinksRatio,
          brand_mismatch: brandMismatch
        };
      }
    });
    
    if (results && results[0] && results[0].result) {
      return results[0].result;
    }
  } catch (_) {}
  return null;
}

// ---------------------------------------------------------------------------
// Run classification and update UI
// ---------------------------------------------------------------------------
async function analyzeActiveTab() {
  const card = document.getElementById("conclusion-card");
  const badge = document.getElementById("conclusion-badge");
  const title = document.getElementById("conclusion-title");
  const desc = document.getElementById("conclusion-desc");
  const urlDisplay = document.getElementById("url-display");
  const reasonDiv = document.getElementById("conclusion-reason");
  const grid = document.getElementById("analysis-grid");

  try {
    // 1. Get active tab URL
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.url) {
      title.textContent = "Unable to Scan";
      desc.textContent = "Please navigate to a valid web page first.";
      urlDisplay.textContent = "N/A";
      reasonDiv.style.display = "none";
      grid.innerHTML = '<div class="analysis-placeholder">Cannot extract page details.</div>';
      return;
    }

    urlDisplay.textContent = tab.url;

    // Guard: ignore chrome:// pages and other non-http URLs
    if (!tab.url.startsWith("http://") && !tab.url.startsWith("https://")) {
      card.className = "conclusion-card";
      badge.textContent = "–";
      title.textContent = "Local Browser Page";
      desc.textContent = "Chrome system pages cannot be analyzed.";
      reasonDiv.style.display = "none";
      grid.innerHTML = '<div class="analysis-placeholder">System protocol is safe by default.</div>';
      return;
    }

    // 2. Extract HTML features via script injection
    const htmlFeatures = await getTabHtmlFeatures(tab.id);

    // 3. Classify via background script
    const response = await chrome.runtime.sendMessage({
      type: "SP_CLASSIFY_URL",
      url: tab.url,
      html_features: htmlFeatures
    });

    if (response && response.ok) {
      const result = response.result;
      const label = result.label; // Safe | Suspicious | Phishing
      const confidence = result.confidence;
      const confPct = (confidence * 100).toFixed(1);

      // 4. Update conclusion card
      const state = STATE_MAP[label] || STATE_MAP.Suspicious;
      card.className = `conclusion-card ${state.cardClass}`;
      badge.textContent = state.badge;
      title.textContent = state.title;
      desc.textContent = state.desc(confPct);

      // 5. Update reasoning summary
      reasonDiv.innerHTML = generateReasoning(label, result.features, htmlFeatures);
      reasonDiv.style.display = "block";

      // 6. Update features grid
      grid.innerHTML = ""; // Clear placeholders
      const features = result.features;

      // Render URL features
      Object.entries(features).forEach(([featName, featValue]) => {
        const readableLabel = FEATURE_LABELS[featName] || featName;
        const evaluation = evaluateFeature(featName, featValue);
        
        let displayValue = featValue;
        if (featName === "has_https") displayValue = featValue === 1 ? "Yes" : "No";
        if (featName === "has_ip_address") displayValue = featValue === 1 ? "Yes" : "No";

        const item = document.createElement("div");
        item.className = "analysis-item";
        item.innerHTML = `
          <div class="analysis-info-left">
            <div class="analysis-label">${readableLabel}</div>
            <div class="analysis-detail">${evaluation.description}</div>
          </div>
          <div class="analysis-right">
            <span class="analysis-value">${displayValue}</span>
            <span class="analysis-dot dot-${evaluation.status}"></span>
          </div>
        `;
        grid.appendChild(item);
      });

      // Render HTML content features (if available)
      if (htmlFeatures) {
        const htmlFeatsRender = [
          {
            label: "HTML Password Fields",
            value: htmlFeatures.has_password_input === 1 ? "Found" : "None",
            status: htmlFeatures.has_password_input === 1 ? "suspicious" : "safe",
            detail: htmlFeatures.has_password_input === 1 ? "Requests password input" : "No password inputs detected"
          },
          {
            label: "HTML Form Fields",
            value: htmlFeatures.has_form === 1 ? "Yes" : "No",
            status: htmlFeatures.has_form === 1 ? "suspicious" : "safe",
            detail: htmlFeatures.has_form === 1 ? "Contains form submission containers" : "No forms detected"
          },
          {
            label: "External Asset Ratio",
            value: `${(htmlFeatures.external_links_ratio * 100).toFixed(1)}%`,
            status: htmlFeatures.external_links_ratio > 0.60 ? "phishing" : (htmlFeatures.external_links_ratio > 0.30 ? "suspicious" : "safe"),
            detail: `Proportion of stylesheet/link targets that point externally`
          },
          {
            label: "Brand Name Spoofing",
            value: htmlFeatures.brand_mismatch === 1 ? "Mismatch" : "None",
            status: htmlFeatures.brand_mismatch === 1 ? "phishing" : "safe",
            detail: htmlFeatures.brand_mismatch === 1 ? "Title contains brand, domain does not" : "No brand mismatches found"
          }
        ];

        htmlFeatsRender.forEach(hf => {
          const item = document.createElement("div");
          item.className = "analysis-item";
          item.innerHTML = `
            <div class="analysis-info-left">
              <div class="analysis-label">${hf.label}</div>
              <div class="analysis-detail">${hf.detail}</div>
            </div>
            <div class="analysis-right">
              <span class="analysis-value">${hf.value}</span>
              <span class="analysis-dot dot-${hf.status}"></span>
            </div>
          `;
          grid.appendChild(item);
        });
      }
    } else {
      // Backend error or offline
      title.textContent = "Error Analyzing Page";
      desc.textContent = response ? response.error : "Failed to connect to the classification backend.";
      reasonDiv.style.display = "none";
      grid.innerHTML = '<div class="analysis-placeholder">Ensure the remote API server is online and running.</div>';
    }
  } catch (err) {
    title.textContent = "Connection Failure";
    desc.textContent = "Extension background service is offline.";
    reasonDiv.style.display = "none";
    grid.innerHTML = '<div class="analysis-placeholder">Please reload the extension.</div>';
  }
}

// ---------------------------------------------------------------------------
// Check backend connectivity and update status dot
// ---------------------------------------------------------------------------
async function checkBackend() {
  const dot = document.getElementById("status-dot");
  try {
    const res = await fetch("https://socialphishing-api.onrender.com/health", {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      dot.className = "status-dot status-active";
      dot.title = "Backend connected";
    } else {
      dot.className = "status-dot status-warning";
      dot.title = "Backend returned an error";
    }
  } catch {
    dot.className = "status-dot status-offline";
    dot.title = "Backend offline — start the local server or check your Render deployment";
  }
}

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  analyzeActiveTab();
  checkBackend();
});
