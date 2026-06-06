/**
 * background.js — SocialPhishing Service Worker
 * ===============================================
 * Handles all HTTP requests to the classification backend, including
 * forwarding HTML structural features alongside the URL.
 */

"use strict";

const BACKEND_URL = "https://socialphishing-api.onrender.com/classify";

// In-memory URL cache (fast path; reset if service worker is evicted)
const urlCache = new Map();

// Helper — fetch a URL classification from the backend
async function fetchClassification(url, htmlFeatures = null) {
  // If we have features, bypass simple URL cache to allow dynamic updates based on page state
  if (!htmlFeatures && urlCache.has(url)) {
    return { ok: true, result: urlCache.get(url), cached: true };
  }

  // Check chrome.storage.session (only if no HTML features are provided)
  if (!htmlFeatures) {
    try {
      const stored = await chrome.storage.session.get(url);
      if (stored[url]) {
        urlCache.set(url, stored[url]); // Warm in-memory cache
        return { ok: true, result: stored[url], cached: true };
      }
    } catch (_) { /* session storage unavailable — continue to API */ }
  }

  // Call the backend API
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const payload = { url };
    if (htmlFeatures) {
      payload.html_features = htmlFeatures;
    }

    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) return { ok: false, error: `HTTP ${response.status}` };

    const result = await response.json();

    // Cache the result if we didn't use dynamic HTML features (to keep URL-only cache accurate)
    if (!htmlFeatures) {
      urlCache.set(url, result);
      chrome.storage.session.set({ [url]: result }).catch(() => {});
    }

    return { ok: true, result, cached: false };
  } catch (err) {
    const msg = err.name === "AbortError" ? "Request timed out" : err.message;
    return { ok: false, error: msg };
  }
}

// Message listener — handles messages from content.js and popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "SP_CLASSIFY_URL") {
    fetchClassification(message.url, message.html_features).then((response) => {
      sendResponse(response);
    });
    return true; // Keep message channel open for async response
  }
  return true;
});
