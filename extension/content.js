/**
 * content.js — SocialPhishing Detector Content Script
 * ====================================================
 * Analyzes the current webpage's URL and HTML features locally,
 * sends them to the FastAPI backend, and injects a warning banner
 * if classified as Phishing.
 */

"use strict";

const BACKEND_URL = "http://localhost:8000/classify";
const NS = "sp-page-warning";

// Helper — decide if a URL is worth classifying
function isClassifiableURL(href) {
  if (!href) return false;
  if (href.startsWith("chrome://") || href.startsWith("chrome-extension://") || href.startsWith("about:")) return false;
  if (!href.startsWith("http://") && !href.startsWith("https://")) return false;
  if (href.startsWith(BACKEND_URL)) return false;
  return true;
}

// Ask backend service worker to classify a URL
async function classifyURL(url, htmlFeatures = null) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({
      type: "SP_CLASSIFY_URL",
      url: url,
      html_features: htmlFeatures
    }, (response) => {
      if (chrome.runtime.lastError) {
        resolve(null);
        return;
      }
      if (response && response.ok) {
        resolve(response.result);
      } else {
        resolve(null);
      }
    });
  });
}

// Extract HTML features locally for privacy and performance
function extractHtmlFeatures() {
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

// Injects the warning banner at the top of the page
function injectWarningBanner(confidence) {
  if (document.getElementById(`${NS}-banner`)) return;

  const banner = document.createElement("div");
  banner.id = `${NS}-banner`;
  banner.style.cssText = `
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    background-color: #ef4444 !important;
    color: white !important;
    text-align: center !important;
    padding: 12px 24px !important;
    font-family: system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    z-index: 2147483647 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
  `;

  const confPct = (confidence * 100).toFixed(1);
  banner.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; text-align: left;">
      <span style="font-size: 20px;">⚠️</span>
      <span><strong>WARNING:</strong> SocialPhishing has classified this website as a phishing threat (${confPct}% confidence). We recommend leaving immediately.</span>
    </div>
    <button id="${NS}-close-btn" style="
      background: rgba(255,255,255,0.2);
      border: 1px solid rgba(255,255,255,0.4);
      color: white;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: bold;
      transition: background 0.2s;
      white-space: nowrap;
      margin-left: 15px;
    ">Dismiss</button>
  `;

  document.body.appendChild(banner);

  // Shift body down to not obscure content
  const originalPaddingTop = parseInt(window.getComputedStyle(document.body).paddingTop || 0);
  document.body.style.paddingTop = (originalPaddingTop + 45) + "px";

  document.getElementById(`${NS}-close-btn`).addEventListener("click", () => {
    banner.remove();
    document.body.style.paddingTop = originalPaddingTop + "px";
  });
}

// Main page analysis
async function analyzePage() {
  const currentUrl = window.location.href;
  if (!isClassifiableURL(currentUrl)) return;

  const htmlFeatures = extractHtmlFeatures();
  const result = await classifyURL(currentUrl, htmlFeatures);
  if (result && result.label === "Phishing") {
    injectWarningBanner(result.confidence);
  }
}

// Run after DOM is available
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", analyzePage);
} else {
  analyzePage();
}
