# Options Screener v6.0
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import pytz
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading as _threading

from pattern_detection import detect_double_bottom, detect_double_top, detect_break_and_retest
from backtester import run_backtest

st.set_page_config(page_title="PaidButPressured", page_icon="📡", layout="centered", initial_sidebar_state="expanded")

# ── PWA MANIFEST + SERVICE WORKER INJECTION ───────────────────────────────────
st.markdown("""
<link rel="manifest" href="data:application/json;charset=utf-8,%7B%22name%22%3A%22PaidButPressured%22%2C%22short_name%22%3A%22PBP%22%2C%22description%22%3A%22Options%20Screener%20by%20PaidButPressured%22%2C%22start_url%22%3A%22%2F%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%23060c14%22%2C%22theme_color%22%3A%22%2300e5aa%22%2C%22orientation%22%3A%22portrait%22%2C%22icons%22%3A%5B%7B%22src%22%3A%22https%3A%2F%2Fraw.githubusercontent.com%2Fyungweb%2Foptions-screener%2Fmain%2Ficon-192.png%22%2C%22sizes%22%3A%22192x192%22%2C%22type%22%3A%22image%2Fpng%22%7D%2C%7B%22src%22%3A%22https%3A%2F%2Fraw.githubusercontent.com%2Fyungweb%2Foptions-screener%2Fmain%2Ficon-512.png%22%2C%22sizes%22%3A%22512x512%22%2C%22type%22%3A%22image%2Fpng%22%7D%5D%7D">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="PaidButPressured">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#D4AF37">
<style>
  /* PWA fullscreen feel - hide Streamlit chrome on mobile */
  @media (display-mode: standalone) {
    header[data-testid="stHeader"] { background: transparent !important; }
    .stDeployButton { display: none; }
    #MainMenu { display: none; }
    footer { display: none; }
  }
</style>
<script>
  // Register service worker for offline/install support
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/app/static/sw.js').catch(function() {
        // SW registration failed silently - app still works
      });
    });
  }
  // Show install prompt on supported browsers
  let deferredPrompt;
  window.addEventListener('beforeinstallprompt', (e) => {
    deferredPrompt = e;
    // Show a subtle install banner
    const banner = document.createElement('div');
    banner.id = 'pwa-install-banner';
    banner.innerHTML = `
      <div style="position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
                  background:#1A1A1D;border:1px solid #D4AF37;border-radius:12px;
                  padding:12px 20px;z-index:9999;display:flex;align-items:center;gap:12px;
                  box-shadow:0 4px 20px rgba(0,229,170,0.2);max-width:320px;width:90%">
        <span style="font-size:1.2rem">📡</span>
        <div>
          <div style="color:#F5F5F5;font-size:0.8rem;font-weight:700">Add to Home Screen</div>
          <div style="color:#A1A1A6;font-size:0.7rem">Install PaidButPressured as an app</div>
        </div>
        <button onclick="installPWA()" style="background:#D4AF37;color:#0B0B0C;border:none;
                border-radius:8px;padding:6px 14px;font-weight:700;font-size:0.75rem;cursor:pointer">
          Install
        </button>
        <button onclick="document.getElementById('pwa-install-banner').remove()"
                style="background:transparent;border:none;color:#A1A1A6;cursor:pointer;font-size:1rem">✕</button>
      </div>
    `;
    document.body.appendChild(banner);
  });
  function installPWA() {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(() => {
        deferredPrompt = null;
        const b = document.getElementById('pwa-install-banner');
        if (b) b.remove();
      });
    }
  }
</script>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── ENVIRONMENT VARIABLES (must be before check_auth) ────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_UID          = os.environ.get("ADMIN_UID", "158a9910")
ADMIN_EMAIL        = os.environ.get("ADMIN_EMAIL", "")
MAKE_WEBHOOK_URL   = os.environ.get("MAKE_WEBHOOK_URL", "https://hook.us2.make.com/k4yp47rg33vdinypxzb3tl7ch6j4u229")
POLYGON_API_KEY    = os.environ.get("POLYGON_API_KEY", "")
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")
FMP_API_KEY        = os.environ.get("FMP_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
APP_PASSWORD       = os.environ.get("APP_PASSWORD", "")
SUPABASE_URL       = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY       = os.environ.get("SUPABASE_KEY", "")

# ── TERMS OF SERVICE TEXT ────────────────────────────────────────────────────
TOS_TEXT = """
**TERMS OF SERVICE & RISK DISCLOSURE**
*PaidButPressured Options Screener - Last updated March 2026*

---

**1. NOT FINANCIAL ADVICE**
PaidButPressured Options Screener ("the Service") is provided for **educational and informational purposes only**. Nothing on this platform constitutes financial, investment, legal, or tax advice. All signals, scores, alerts, and paper trading results are generated by automated algorithms and do not represent personalized investment recommendations.

**2. NO GUARANTEE OF RESULTS**
Options trading involves substantial risk of loss and is not appropriate for all investors. Past performance of signals, patterns, or paper trades does not guarantee future results. You may lose some or all of your invested capital. The Service makes no representation that any signal will result in a profit.

**3. YOUR RESPONSIBILITY**
By using this Service you acknowledge that:
- You are solely responsible for all trading decisions you make
- You will conduct your own research before placing any trade
- You understand the risks associated with options trading
- You are not relying on this Service as your primary source of investment guidance
- You have read and understood this entire agreement

**4. NO BROKER RELATIONSHIP**
PaidButPressured is not a registered investment advisor, broker-dealer, or financial institution. Use of this Service does not create any fiduciary duty or advisory relationship between you and PaidButPressured.

**5. ACCURACY OF DATA**
Market data, prices, and signals are sourced from third-party providers and may be delayed, inaccurate, or unavailable. PaidButPressured is not responsible for any errors in data or signal generation.

**6. LIMITATION OF LIABILITY**
To the maximum extent permitted by law, PaidButPressured shall not be liable for any direct, indirect, incidental, special, or consequential damages arising from your use of this Service or any trading decisions made based on information provided herein.

**7. SUBSCRIPTION & REFUNDS**
Subscription fees are non-refundable. Access may be revoked at any time for violation of these terms.

**8. CHANGES TO TERMS**
These terms may be updated at any time. Continued use of the Service constitutes acceptance of any revised terms.

*By checking the box below and entering the app, you confirm that you have read, understood, and agree to all terms above.*
"""

# ── LOGIN WALL + TOS ──────────────────────────────────────────────────────────
def check_auth():
    """
    Supabase Auth gate — email/password login.
    Persists session via access token stored in query params.
    Auto-refreshes token to prevent mid-session logouts.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.session_state.tos_agreed    = True
        st.session_state.authenticated = True
        st.session_state.user_email    = "dev@local"
        st.session_state.user_id       = "dev"
        return

    # Already authenticated this session — try token refresh to keep alive
    if st.session_state.get("authenticated") and st.session_state.get("user_email"):
        # Refresh token every ~10 minutes to prevent expiry
        _last_refresh = st.session_state.get("_last_token_refresh")
        _now = datetime.now()
        if _last_refresh is None or (_now - _last_refresh).total_seconds() > 600:
            try:
                from supabase import create_client
                sb = create_client(SUPABASE_URL, SUPABASE_KEY)
                _rt = st.session_state.get("_refresh_token")
                if _rt:
                    resp = sb.auth.refresh_session(_rt)
                    if resp and resp.session:
                        st.session_state._access_token   = resp.session.access_token
                        st.session_state._refresh_token  = resp.session.refresh_token
                        st.session_state._last_token_refresh = _now
            except Exception:
                pass
        return

    # Try to restore session from stored tokens
    _access  = st.session_state.get("_access_token")
    _refresh = st.session_state.get("_refresh_token")
    if _access and _refresh:
        try:
            from supabase import create_client
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            resp = sb.auth.set_session(_access, _refresh)
            if resp and resp.user:
                st.session_state.authenticated        = True
                st.session_state.tos_agreed           = True
                st.session_state.user_email           = resp.user.email
                st.session_state.user_id              = resp.user.id
                st.session_state.is_admin             = (resp.user.email == ADMIN_EMAIL)
                st.session_state.watchlist_loaded     = False
                st.session_state._last_token_refresh  = datetime.now()
                return
        except Exception:
            pass  # tokens expired — fall through to login screen

    # ── Auth screen ───────────────────────────────────────────────────────────
    st.markdown("""
<style>
.auth-wrap { max-width:400px; margin:60px auto; padding:32px 36px;
             background:#1A1A1D; border:1px solid #2A2A2D; border-radius:16px; }
.auth-title { font-size:1.4rem; font-weight:700; color:#F5F5F5;
              text-align:center; letter-spacing:2px; margin-bottom:4px; }
.auth-sub   { font-size:0.75rem; color:#A1A1A6; text-align:center; margin-bottom:24px; }
</style>
<div class="auth-wrap">
  <div class="auth-title">📡 PAIDBUTPRESSURED</div>
  <div class="auth-sub">Options Screener · Member Access</div>
</div>
""", unsafe_allow_html=True)

    col = st.columns([1,4,1])[1]
    with col:
        _mode = st.radio("", ["Sign In", "Create Account"],
                         horizontal=True, label_visibility="collapsed")
        email    = st.text_input("Email", placeholder="your@email.com",
                                 label_visibility="collapsed")
        password = st.text_input("Password", type="password",
                                 placeholder="Password (min 6 chars)",
                                 label_visibility="collapsed")

        if _mode == "Sign In":
            if st.button("Sign In →", type="primary", use_container_width=True):
                if not email or not password:
                    st.error("Enter your email and password")
                else:
                    try:
                        from supabase import create_client
                        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
                        resp = sb.auth.sign_in_with_password({"email": email, "password": password})
                        if resp.user:
                            st.session_state.authenticated        = True
                            st.session_state.tos_agreed           = True
                            st.session_state.user_email           = resp.user.email
                            st.session_state.user_id              = resp.user.id
                            st.session_state.is_admin             = (resp.user.email == ADMIN_EMAIL)
                            st.session_state.watchlist_loaded     = False
                            st.session_state._access_token        = resp.session.access_token
                            st.session_state._refresh_token       = resp.session.refresh_token
                            st.session_state._last_token_refresh  = datetime.now()
                            st.rerun()
                        else:
                            st.error("Sign in failed — check your email and password")
                    except Exception as e:
                        st.error("Sign in error: %s" % str(e)[:100])

        else:  # Create Account
            # Show TOS before signup
            with st.expander("📋 Read Terms of Service before signing up", expanded=False):
                st.markdown(TOS_TEXT)
            agreed = st.checkbox("I agree to the Terms of Service and Risk Disclosure")
            if st.button("Create Account →", type="primary",
                         use_container_width=True, disabled=not agreed):
                if not email or not password:
                    st.error("Enter your email and password")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        from supabase import create_client
                        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
                        resp = sb.auth.sign_up({"email": email, "password": password})
                        if resp.user:
                            st.session_state.authenticated        = True
                            st.session_state.tos_agreed           = True
                            st.session_state.user_email           = resp.user.email
                            st.session_state.user_id              = resp.user.id
                            st.session_state.is_admin             = (resp.user.email == ADMIN_EMAIL)
                            st.session_state.watchlist_loaded     = False
                            if resp.session:
                                st.session_state._access_token       = resp.session.access_token
                                st.session_state._refresh_token      = resp.session.refresh_token
                                st.session_state._last_token_refresh = datetime.now()
                            st.success("Account created! Welcome to PaidButPressured.")
                            st.rerun()
                        else:
                            st.error("Signup failed — try a different email")
                    except Exception as e:
                        err = str(e)
                        if "already registered" in err.lower() or "duplicate" in err.lower():
                            st.error("Email already registered — use Sign In instead")
                        else:
                            st.error("Signup error: %s" % err[:100])

        st.markdown(
            "<div style='text-align:center;margin-top:16px;font-size:0.68rem;color:#4a5568'>"
            "⚠️ Options trading involves substantial risk. Not financial advice."
            "</div>", unsafe_allow_html=True
        )
    st.stop()


def check_onboarding():
    """
    Shows onboarding flow for first-time users.
    Tracks completion via Supabase onboarding_complete flag.
    Skip button available on every step.
    """
    # Already completed onboarding this session
    if st.session_state.get("onboarding_complete"):
        return

    # Check Supabase for onboarding status
    user_id = st.session_state.get("user_id")
    if user_id:
        try:
            sb = get_supabase()
            if sb:
                res = sb.table("user_data").select("preferences").eq("user_id", str(user_id)).execute()
                if res.data:
                    import json as _j
                    prefs = _j.loads(res.data[0].get("preferences", "{}"))
                    if prefs.get("onboarding_complete"):
                        st.session_state.onboarding_complete = True
                        return
        except Exception:
            pass

    # Initialize step
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1

    step = st.session_state.onboarding_step

    def complete_onboarding():
        st.session_state.onboarding_complete = True
        # Save to Supabase
        if user_id:
            try:
                sb = get_supabase()
                if sb:
                    import json as _j
                    res = sb.table("user_data").select("preferences").eq("user_id", str(user_id)).execute()
                    prefs = {}
                    if res.data:
                        prefs = _j.loads(res.data[0].get("preferences", "{}"))
                    prefs["onboarding_complete"] = True
                    sb.table("user_data").upsert({
                        "user_id": str(user_id),
                        "preferences": _j.dumps(prefs),
                        "updated_at": datetime.now(tz=pytz.UTC).isoformat()
                    }).execute()
            except Exception:
                pass

    # Onboarding UI
    st.markdown("""
<style>
.ob-wrap { max-width:480px; margin:40px auto; padding:32px 36px;
           background:#1A1A1D; border:1px solid #2A2A2D; border-radius:16px; }
.ob-step { font-size:0.65rem; color:#A1A1A6; letter-spacing:3px;
           text-align:center; margin-bottom:8px; }
.ob-title { font-size:1.4rem; font-weight:700; color:#F5F5F5;
            text-align:center; margin-bottom:8px; }
.ob-body { font-size:0.85rem; color:#A1A1A6; text-align:center;
           line-height:1.8; margin-bottom:24px; }
.ob-badge { display:inline-block; padding:4px 14px; border-radius:20px;
            font-size:0.75rem; font-weight:700; margin:4px; }
</style>
""", unsafe_allow_html=True)

    col = st.columns([1, 6, 1])[1]
    with col:
        if step == 1:
            st.markdown("""
<div class='ob-wrap'>
  <div class='ob-step'>STEP 1 OF 4</div>
  <div class='ob-title'>Welcome to PaidButPressured 📡</div>
  <div class='ob-body'>
    A real-time options screener built for active traders.<br><br>
    We scan the market, filter out the noise, and surface only the 
    highest-conviction setups — with clear entry, target, and stop levels.<br><br>
    <b style='color:#D4AF37'>Let's show you how it works.</b>
  </div>
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Skip Tutorial", use_container_width=True, key="ob_skip_1"):
                    complete_onboarding()
                    st.rerun()
            with c2:
                if st.button("Let's Go →", type="primary", use_container_width=True, key="ob_next_1"):
                    st.session_state.onboarding_step = 2
                    st.rerun()

        elif step == 2:
            st.markdown("""
<div class='ob-wrap'>
  <div class='ob-step'>STEP 2 OF 4</div>
  <div class='ob-title'>How to Scan 🔍</div>
  <div class='ob-body'>
    Open the <b style='color:#F5F5F5'>SCAN tab</b> and pick a sector from the dropdown.<br><br>
    Start with <b style='color:#D4AF37'>My Watchlist</b> for tickers you already follow,
    or choose a sector like <b style='color:#D4AF37'>Tech & Semis</b> or 
    <b style='color:#D4AF37'>High Momentum</b>.<br><br>
    Hit <b style='color:#F5F5F5'>RUN SCAN</b> and let the engine do the work.
  </div>
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Back", use_container_width=True, key="ob_back_2"):
                    st.session_state.onboarding_step = 1
                    st.rerun()
            with c2:
                if st.button("Got it →", type="primary", use_container_width=True, key="ob_next_2"):
                    st.session_state.onboarding_step = 3
                    st.rerun()

        elif step == 3:
            st.markdown("""
<div class='ob-wrap'>
  <div class='ob-step'>STEP 3 OF 4</div>
  <div class='ob-title'>Reading Signals 🚦</div>
  <div class='ob-body'>
    Every signal lands in one of three buckets:<br><br>
    <span class='ob-badge' style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E'>
      🟢 GO NOW
    </span>
    Entry confirmed. Highest conviction. Act now.<br><br>
    <span class='ob-badge' style='background:#D4AF3722;color:#D4AF37;border:1px solid #D4AF37'>
      🟡 WATCHING
    </span>
    Building conviction. Wait for confirmation.<br><br>
    <span class='ob-badge' style='background:#C1121F22;color:#C1121F;border:1px solid #C1121F'>
      🔴 ON DECK
    </span>
    Setup developing. Not ready yet.
  </div>
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Back", use_container_width=True, key="ob_back_3"):
                    st.session_state.onboarding_step = 2
                    st.rerun()
            with c2:
                if st.button("Makes sense →", type="primary", use_container_width=True, key="ob_next_3"):
                    st.session_state.onboarding_step = 4
                    st.rerun()

        elif step == 4:
            st.markdown("""
<div class='ob-wrap'>
  <div class='ob-step'>STEP 4 OF 4</div>
  <div class='ob-title'>Watch Queue 👁</div>
  <div class='ob-body'>
    See a WATCHING signal you like?<br><br>
    Hit the <b style='color:#D4AF37'>Watch button</b> on the signal card and we'll 
    track the entry timing for you.<br><br>
    Check the <b style='color:#F5F5F5'>WATCH QUEUE tab</b> to see when your 
    signal confirms — and get ready to enter.<br><br>
    <b style='color:#D4AF37'>You're ready. Let's find some setups.</b>
  </div>
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Back", use_container_width=True, key="ob_back_4"):
                    st.session_state.onboarding_step = 3
                    st.rerun()
            with c2:
                if st.button("Start Scanning →", type="primary", use_container_width=True, key="ob_finish"):
                    complete_onboarding()
                    st.rerun()

    st.stop()

check_auth()
check_onboarding()  # Show first-time tutorial
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@600;700&family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', 'Barlow', sans-serif; }
body, .stApp { background: #0B0B0C; color: #F5F5F5; }
.stSidebar { background: #1A1A1D !important; border-right: 1px solid #2A2A2D; }

/* Hide Streamlit branding but keep sidebar toggle */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Gold gradient buttons */
.stButton>button {
    background: linear-gradient(90deg, #D4AF37, #F6E27A) !important;
    color: #0B0B0C !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Block container padding */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.big-price { font-size: 2rem; font-weight: 700; }
.section-title { color: #D4AF37; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; letter-spacing: 2px; margin: 20px 0 8px; border-bottom: 1px solid #2A2A2D; padding-bottom: 4px; }
.metric-card { background: #111827; border: 1px solid #2A2A2D; border-radius: 8px; padding: 14px; margin: 4px 0; transition: all 0.2s ease; }
.metric-card:hover { border-color: #D4AF37; transform: translateY(-2px); }
.rank-best   { background: #1A1500; border: 2px solid #D4AF37; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-better { background: #0a1a0a; border: 2px solid #40c070; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-good   { background: #141a0a; border: 2px solid #F6E27A; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-badge  { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; padding: 3px 10px; border-radius: 20px; display: inline-block; margin-bottom: 8px; }
.badge-best   { background: #D4AF3722; color: #D4AF37; }
.badge-better { background: #40c07022; color: #40c070; }
.badge-good   { background: #F6E27A22; color: #F6E27A; }
.conf-num-best   { font-size: 2.2rem; font-weight: 700; color: #D4AF37; }
.conf-num-better { font-size: 2.2rem; font-weight: 700; color: #40c070; }
.conf-num-good   { font-size: 2.2rem; font-weight: 700; color: #F6E27A; }
.factor-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.82rem; }
.dot-green  { width: 8px; height: 8px; background: #D4AF37; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-red    { width: 8px; height: 8px; background: #C1121F; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-yellow { width: 8px; height: 8px; background: #F6E27A; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.trade-box  { background: #111827; border-radius: 8px; padding: 14px; margin-top: 10px; border-left: 3px solid #D4AF37; }
.trade-box.bear { border-left-color: #C1121F; }
.exit-rules { background: #0d1525; border: 1px solid #2A2A2D; border-radius: 8px; padding: 12px 14px; margin-top: 10px; font-size: 0.83rem; }
.gate-box   { background: #1A1A1D; border: 1px solid #2A2A2D; border-radius: 8px; padding: 12px 14px; margin-top: 8px; }
.ai-placeholder { background: #1A1A1D; border: 1px dashed #2A2A2D; border-radius: 8px; padding: 14px; margin-top: 10px; color: #A1A1A6; font-size: 0.83rem; text-align: center; }
.conflict-warn { background: #1a150a; border: 1px solid #F6E27A; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #F6E27A; }
.market-open   { background: #1A1500; border: 1px solid #D4AF37; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #D4AF37; font-size: 0.85rem; }
.market-closed { background: #1a1010; border: 1px solid #C1121F; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #C1121F; font-size: 0.85rem; }
.market-pre    { background: #1a150a; border: 1px solid #F6E27A; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #F6E27A; font-size: 0.85rem; }
.divergence-bull { background: #1A1500; border: 1px solid #D4AF37; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #D4AF37; }
.divergence-bear { background: #1a0610; border: 1px solid #C1121F; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #C1121F; }
</style>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# env vars defined at top of file
# Default watchlist - users can customize this in the app
DEFAULT_WATCHLIST = ["SPY", "QQQ", "IWM"]
WATCHLIST = DEFAULT_WATCHLIST  # overridden at runtime by session state

# Full scan universe - 120 most liquid options tickers across all sectors
SCAN_UNIVERSE = [
    # Mega cap / Index ETFs
    "SPY","QQQ","IWM","DIA","AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","BRK-B",
    # Semis
    "AMD","INTC","AVGO","QCOM","MU","AMAT","LRCX","KLAC","MRVL","CRDO","SMCI","ARM","TSM",
    # Tech / Cloud / Cyber
    "PLTR","SNOW","DDOG","NET","CRWD","ZS","PANW","FTNT","OKTA","S","XYZ","COIN","VRT","WDC",
    "APP","AXON","MSTR","NBIS","ZETA","AAOI",
    # Large cap growth
    "NFLX","UBER","LYFT","ABNB","SHOP","MELI","BABA","PDD","SE","GRAB",
    "RBLX","U","TTWO","EA",
    # Financials
    "JPM","BAC","GS","MS","C","WFC","BLK","V","MA","PYPL","AXP",
    "SOFI","AFRM","HOOD",
    # Healthcare / Biotech
    "UNH","JNJ","PFE","MRNA","BNTX","ABBV","LLY","BMY","GILD","REGN","BIIB",
    # Energy / Power
    "XOM","CVX","OXY","SLB","HAL","MPC","PSX","VST","CEG","GEV",
    # Consumer
    "WMT","TGT","COST","HD","LOW","NKE","LULU","MCD","SBUX","CMG",
    "DKNG","CELH","HIMS",
    # Industrial / EV / Defense
    "GE","CAT","DE","BA","LMT","RTX","RIVN","LCID","F","GM",
    # Speculative / High momentum
    "ASTS","RDW","IREN",
    # ETF sectors
    "XLK","XLF","XLE","XLV","XLY","XLI","GLD","SLV","TLT","HYG","IBIT",
]
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))  # deduplicate

# ── Sector scan lists ─────────────────────────────────────────────────────────
SECTOR_LISTS = {
    "My Watchlist":     [],  # populated from session state at runtime
    "Tech & Semis":     ["NVDA","AMD","INTC","AVGO","QCOM","MU","AMAT","LRCX","KLAC","MRVL",
                         "ARM","TSM","PLTR","SNOW","DDOG","NET","CRWD","ZS","PANW","FTNT",
                         "OKTA","S","APP","AXON","WDC","SMCI"],
    "Mega Cap":         ["AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA","NVDA","BRK-B","SPY","QQQ","IWM","DIA"],
    "Financials":       ["JPM","BAC","GS","MS","C","WFC","BLK","V","MA","PYPL","AXP","SOFI","AFRM","HOOD","XYZ","COIN"],
    "Healthcare":       ["UNH","JNJ","PFE","MRNA","BNTX","ABBV","LLY","BMY","GILD","REGN","BIIB"],
    "Energy & Power":   ["XOM","CVX","OXY","SLB","HAL","MPC","PSX","VST","CEG","GEV"],
    "Consumer":         ["WMT","TGT","COST","HD","LOW","NKE","LULU","MCD","SBUX","CMG","DKNG","CELH","HIMS"],
    "High Momentum":    ["PLTR","TSLA","COIN","MSTR","ASTS","NVDA","AMD","HOOD","SOFI","AFRM",
                         "RIVN","RDW","IREN","NBIS","ZETA","AAOI","CRDO","LCID","GRAB","SE"],
    "Industrial & EV":  ["GE","CAT","DE","BA","LMT","RTX","RIVN","LCID","F","GM"],
    "ETF Sectors":      ["XLK","XLF","XLE","XLV","XLY","XLI","GLD","SLV","TLT","HYG","IBIT","SPY","QQQ","IWM"],
    "Full Universe":    [],  # populated from SCAN_UNIVERSE at runtime
}

# Sector ETF map for sector alignment check
SECTOR_ETF = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","INTC":"XLK","AVGO":"XLK",
    "QCOM":"XLK","MU":"XLK","AMAT":"XLK","LRCX":"XLK","KLAC":"XLK","MRVL":"XLK",
    "CRDO":"XLK","SMCI":"XLK","ARM":"XLK","TSM":"XLK","WDC":"XLK",
    "PLTR":"XLK","SNOW":"XLK","DDOG":"XLK","NET":"XLK","CRWD":"XLK","ZS":"XLK",
    "PANW":"XLK","FTNT":"XLK","OKTA":"XLK","S":"XLK","NBIS":"XLK","VRT":"XLK",
    "AAOI":"XLK","ASTS":"XLK","ZETA":"XLK","IREN":"XLK",
    "XYZ":"XLF","COIN":"XLF","HOOD":"XLF","PYPL":"XLF","V":"XLF","MA":"XLF",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","C":"XLF","WFC":"XLF",
    "BLK":"XLF","AXP":"XLF",
    "UNH":"XLV","JNJ":"XLV","PFE":"XLV","MRNA":"XLV","BNTX":"XLV","ABBV":"XLV",
    "LLY":"XLV","BMY":"XLV","GILD":"XLV","REGN":"XLV","BIIB":"XLV",
    "XOM":"XLE","CVX":"XLE","OXY":"XLE","SLB":"XLE","HAL":"XLE","MPC":"XLE","PSX":"XLE",
    "WMT":"XLY","TGT":"XLY","COST":"XLY","HD":"XLY","LOW":"XLY","NKE":"XLY",
    "LULU":"XLY","MCD":"XLY","SBUX":"XLY","CMG":"XLY","AMZN":"XLY",
    "NFLX":"XLY","UBER":"XLY","LYFT":"XLY","ABNB":"XLY","RBLX":"XLY",
    "GE":"XLI","CAT":"XLI","DE":"XLI","BA":"XLI","LMT":"XLI","RTX":"XLI",
    "RIVN":"XLI","LCID":"XLI","F":"XLI","GM":"XLI",
}
TIMEFRAMES = {
    "5 Min":  ("minute", 5,  2),
    "15 Min": ("minute", 15, 5),
    "1 Hour": ("hour",   1,  14),
    "4 Hour": ("hour",   4,  30),
    "Daily":  ("day",    1,  90),
}

def get_market_status():
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    wd  = now.weekday()
    t   = now.time()
    from datetime import time as dtime
    if wd >= 5: return "closed", "Market Closed - Weekend"
    if   t < dtime(4,  0): return "closed", "Market Closed - Opens 4:00 AM ET"
    elif t < dtime(9, 30): return "pre",    "Pre-Market Hours - Regular session opens 9:30 AM ET"
    elif t < dtime(16, 0): return "open",   "Market Open - Regular Session Until 4:00 PM ET"
    elif t < dtime(20, 0): return "after",  "After-Hours Trading - Until 8:00 PM ET"
    else:                  return "closed", "Market Closed - Pre-market opens 4:00 AM ET"

# ── Thread-safe TTL cache ─────────────────────────────────────────────────────
# @st.cache_data cannot be called safely from background threads (ThreadPoolExecutor).
# Doing so fires "missing ScriptRunContext" warnings and can cause instability.
# Functions in the scan worker path use _thread_cache instead - pure Python,
# no Streamlit context required, thread-safe via a single lock.
import time as _time_mod
_THREAD_CACHE      = {}
_THREAD_CACHE_LOCK = _threading.Lock()

def _thread_cache(ttl=300):
    """
    Decorator: simple TTL memoize safe to call from any thread.
    Stores results in a module-level dict keyed by (func_name, args).
    Expired entries are evicted on the next call for that key.
    """
    def decorator(fn):
        def wrapper(*args):
            key = (fn.__name__,) + args
            now = _time_mod.time()
            with _THREAD_CACHE_LOCK:
                entry = _THREAD_CACHE.get(key)
                if entry and (now - entry[0]) < ttl:
                    return entry[1]
            result = fn(*args)
            with _THREAD_CACHE_LOCK:
                _THREAD_CACHE[key] = (now, result)
            return result
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator
# ─────────────────────────────────────────────────────────────────────────────

# ── DataFrame column helpers ──────────────────────────────────────────────────
# yfinance v0.2.x returns MultiIndex columns for single-ticker downloads.
# e.g. df["close"] returns a DataFrame with shape (n,1) instead of a Series.
# _col() safely squeezes any column to a 1D float Series regardless of structure.

def _col(df, name):
    """Return column `name` from df as a guaranteed 1D float Series."""
    c = df[name]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.astype(float)

def _clean_df(df):
    """
    Normalize a yfinance DataFrame so all columns are flat (non-MultiIndex)
    and contain float values. Safe to call multiple times (idempotent).
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
# ─────────────────────────────────────────────────────────────────────────────

# ── yfinance session manager ──────────────────────────────────────────────────
# Yahoo Finance rotates crumb/auth tokens. A stale session causes 401 Unauthorized.
# We keep one module-level session and refresh it on any 401/crumb error.
_YF_SESSION_LOCK = _threading.Lock()
_yf_session      = None

def _get_yf_session():
    """Return a live yfinance session, refreshing on demand."""
    global _yf_session
    import yfinance as yf
    import requests
    with _YF_SESSION_LOCK:
        if _yf_session is None:
            _yf_session = requests.Session()
            _yf_session.headers.update({"User-Agent": "Mozilla/5.0"})
    return _yf_session

def _polygon_download(ticker, period, interval):
    """
    Fetch OHLCV from Polygon.io REST API.
    Converts yfinance-style period/interval to Polygon multiplier/timespan.
    Returns a DataFrame with columns: datetime, open, high, low, close, volume
    or None on failure.
    """
    api_key = POLYGON_API_KEY
    if not api_key:
        return None
    try:
        import requests as _req
        # Map interval → Polygon timespan
        tf_map = {
            "1m":  (1,  "minute"), "2m":  (2,  "minute"), "5m":  (5,  "minute"),
            "15m": (15, "minute"), "30m": (30, "minute"),
            "1h":  (1,  "hour"),   "2h":  (2,  "hour"),   "4h":  (4,  "hour"),
            "1d":  (1,  "day"),    "1wk": (1,  "week"),
        }
        mult, span = tf_map.get(interval, (1, "day"))

        # Map period → days back
        period_days = {
            "1d": 1, "2d": 2, "5d": 5, "7d": 7,
            "14d": 14, "30d": 30, "60d": 60, "90d": 90,
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        }
        days = period_days.get(period, 30)
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        from_str = start_dt.strftime("%Y-%m-%d")
        to_str   = end_dt.strftime("%Y-%m-%d")

        url = (
            "https://api.polygon.io/v2/aggs/ticker/%s/range/%s/%s/%s/%s"
            "?adjusted=true&sort=asc&limit=50000&apiKey=%s"
            % (ticker.upper(), mult, span, from_str, to_str, api_key)
        )
        r = _req.get(url, timeout=4)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None

        df = pd.DataFrame(results)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["datetime","open","high","low","close","volume"]].dropna().reset_index(drop=True)
    except Exception:
        return None


def _finnhub_download(ticker, period, interval):
    """
    Fetch OHLCV from Finnhub REST API.
    Free tier: 60 calls/minute. Retries once on 429.
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        import requests as _req, time as _t
        res_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "4h": "D", "1d": "D", "1wk": "W"
        }
        resolution = res_map.get(interval, "D")
        period_days = {
            "1d": 1, "2d": 2, "5d": 5, "7d": 7,
            "14d": 14, "30d": 30, "60d": 60, "90d": 90,
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        }
        days    = period_days.get(period, 30)
        to_ts   = int(_t.time())
        from_ts = to_ts - days * 86400
        url = (
            "https://finnhub.io/api/v1/stock/candle"
            "?symbol=%s&resolution=%s&from=%s&to=%s&token=%s"
            % (ticker.upper(), resolution, from_ts, to_ts, FINNHUB_API_KEY)
        )
        for attempt in range(2):
            r = _req.get(url, timeout=5)
            if r.status_code == 429:
                _t.sleep(2)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            if data.get("s") != "ok" or not data.get("t"):
                return None
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["t"], unit="s"),
                "open":     data["o"],
                "high":     data["h"],
                "low":      data["l"],
                "close":    data["c"],
                "volume":   data["v"],
            })
            return df.dropna().reset_index(drop=True)
        return None
    except Exception:
        return None
def _finnhub_price(ticker):
    """Get real-time quote from Finnhub."""
    if not FINNHUB_API_KEY:
        return None
    try:
        import requests as _req
        r = _req.get(
            "https://finnhub.io/api/v1/quote?symbol=%s&token=%s"
            % (ticker.upper(), FINNHUB_API_KEY),
            timeout=4
        )
        if r.status_code != 200:
            return None
        data = r.json()
        price = data.get("c")  # current price
        return round(float(price), 2) if price else None
    except Exception:
        return None


def _fmp_download(ticker, period, interval):
    """
    Fetch OHLCV from Financial Modeling Prep (FMP) Premium.
    Uses stable endpoints: /stable/historical-chart/{interval} for intraday
    and /stable/historical-price-eod/full for daily.
    """
    if not FMP_API_KEY:
        return None
    try:
        import requests as _req
        interval_map = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1hour", "4h": "4hour", "1d": "1day", "1wk": "1week"
        }
        fmp_interval = interval_map.get(interval, "1day")
        period_days  = {
            "1d":1,"2d":2,"5d":5,"7d":7,"14d":14,"30d":30,
            "60d":60,"90d":90,"1mo":30,"3mo":90,"6mo":180,"1y":365,
        }
        days    = period_days.get(period, 30)
        from_dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_dt   = datetime.now().strftime("%Y-%m-%d")

        if fmp_interval in ["1day", "1week"]:
            # Daily endpoint
            url = (
                "https://financialmodelingprep.com/stable/historical-price-eod/full"
                "?symbol=%s&from=%s&to=%s&apikey=%s"
                % (ticker.upper(), from_dt, to_dt, FMP_API_KEY)
            )
            r = _req.get(url, timeout=8)
            if r.status_code != 200: return None
            data = r.json()
            # Stable EOD returns {"symbol":..., "historical":[...]}
            if isinstance(data, dict):
                hist = data.get("historical", [])
            elif isinstance(data, list):
                hist = data
            else:
                return None
            if not hist: return None
            df = pd.DataFrame(hist)
        else:
            # Intraday endpoint — /stable/historical-chart/5min?symbol=AAPL
            url = (
                "https://financialmodelingprep.com/stable/historical-chart/%s"
                "?symbol=%s&from=%s&to=%s&apikey=%s"
                % (fmp_interval, ticker.upper(), from_dt, to_dt, FMP_API_KEY)
            )
            r = _req.get(url, timeout=8)
            if r.status_code != 200: return None
            data = r.json()
            if not data or not isinstance(data, list): return None
            df = pd.DataFrame(data)

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing: return None

        return df[required].dropna().reset_index(drop=True)
    except Exception:
        return None


def _fmp_debug(ticker, interval="5m", period="5d"):
    """Diagnostic — tests correct FMP stable/historical-chart endpoint."""
    if not FMP_API_KEY:
        return {"error": "No FMP_API_KEY set"}
    try:
        import requests as _req
        from_dt = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        to_dt   = datetime.now().strftime("%Y-%m-%d")
        url = (
            "https://financialmodelingprep.com/stable/historical-chart/5min"
            "?symbol=%s&from=%s&to=%s&apikey=%s"
            % (ticker.upper(), from_dt, to_dt, FMP_API_KEY)
        )
        r = _req.get(url, timeout=8)
        data = r.json()
        return {
            "endpoint": "stable/historical-chart/5min",
            "status_code": r.status_code,
            "url": url.replace(FMP_API_KEY, "***"),
            "response_type": type(data).__name__,
            "record_count": len(data) if isinstance(data, list) else "not a list",
            "keys_if_dict": list(data.keys()) if isinstance(data, dict) else None,
            "first_record": data[0] if isinstance(data, list) and data else data,
        }
    except Exception as e:
        return {"error": str(e)}


def _fmp_price(ticker):
    """Get real-time quote from FMP."""
    if not FMP_API_KEY:
        return None
    try:
        import requests as _req
        r = _req.get(
            "https://financialmodelingprep.com/api/v3/quote-short/%s?apikey=%s"
            % (ticker.upper(), FMP_API_KEY),
            timeout=4
        )
        if r.status_code != 200: return None
        data = r.json()
        if data and isinstance(data, list):
            price = data[0].get("price", 0)
            return round(float(price), 2) if price else None
    except Exception:
        pass
    return None


def _yf_download(ticker, period, interval, **kwargs):
    """
    FMP-first data fetcher. Premium = 750 calls/min, intraday, no rate limits.
    Falls back to yfinance, then Finnhub, then Polygon.
    """
    # 1. FMP Premium - best coverage, no rate issues, intraday included
    if FMP_API_KEY:
        df = _fmp_download(ticker, period, interval)
        if df is not None and not df.empty:
            return df

    # 2. yfinance fallback
    try:
        import yfinance as yf
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True,
            threads=False, session=session, **kwargs
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 3. Finnhub fallback
    if FINNHUB_API_KEY:
        df = _finnhub_download(ticker, period, interval)
        if df is not None and not df.empty:
            return df

    # 4. Polygon last resort
    if POLYGON_API_KEY:
        df = _polygon_download(ticker, period, interval)
        if df is not None and not df.empty:
            return df

    return None

@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, multiplier, timespan, days_back):
    try:
        intervals = {"minute": "5m", "hour": "1h", "day": "1d"}
        interval  = intervals.get(timespan, "1h")
        period    = f"{min(days_back, 59)}d" if timespan == "minute" else f"{days_back}d"
        df = _yf_download(ticker, period=period, interval=interval, prepost=True)
        if df is None or df.empty:
            return _demo_data(ticker)
        df = df.reset_index()
        df = _clean_df(df)
        df = df.rename(columns={"datetime": "timestamp", "date": "timestamp"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    except:
        return _demo_data(ticker)

def _demo_data(ticker, bars=200):
    np.random.seed(hash(ticker)%999)
    prices = {"PLTR":118,"NBIS":98,"VRT":92,"CRDO":68,"GOOGL":175,
              "AAOI":22,"ASTS":28,"ZETA":19,"SPY":570,"QQQ":490,
              "NVDA":138,"TSLA":320,"AAPL":228}.get(ticker,100)
    dates = pd.date_range(end=datetime.now(), periods=bars, freq="1h")
    close = [prices]
    for _ in range(bars-1):
        close.append(close[-1]*(1+np.random.normal(0,0.012)))
    close = np.array(close)
    hi  = close*(1+np.abs(np.random.normal(0,0.008,bars)))
    lo  = close*(1-np.abs(np.random.normal(0,0.008,bars)))
    op  = lo+np.random.uniform(0,1,bars)*(hi-lo)
    vol = np.random.randint(500000,3000000,bars)
    return pd.DataFrame({"timestamp":dates,"open":op,"high":hi,"low":lo,"close":close,"volume":vol})

@_thread_cache(ttl=30)
def fetch_current_price(ticker):
    # FMP real-time quote - accurate pre/during/after market
    if FMP_API_KEY:
        price = _fmp_price(ticker)
        if price:
            return price
    # Finnhub fallback
    if FINNHUB_API_KEY:
        price = _finnhub_price(ticker)
        if price:
            return price
    # yfinance last resort
    try:
        df = _yf_download(ticker, period="1d", interval="1m")
        if df is None or df.empty: return None
        df = _clean_df(df.reset_index())
        return round(float(_col(df,"close").iloc[-1]), 2)
    except:
        return None

# ETFs and indices never have earnings - skip immediately to avoid 404s
_ETF_TICKERS = {
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLI",
    "GLD","SLV","TLT","HYG","VXX","UVXY","SQQQ","TQQQ","SPXU","SPXL",
}

@_thread_cache(ttl=3600)
def check_earnings(ticker):
    if ticker in _ETF_TICKERS:
        return None
    # Use Finnhub earnings calendar if available
    if FINNHUB_API_KEY:
        try:
            import requests as _req
            from_d = datetime.now().strftime("%Y-%m-%d")
            to_d   = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            r = _req.get(
                "https://finnhub.io/api/v1/calendar/earnings"
                "?from=%s&to=%s&symbol=%s&token=%s"
                % (from_d, to_d, ticker.upper(), FINNHUB_API_KEY),
                timeout=4
            )
            if r.status_code == 200:
                data = r.json()
                earnings = data.get("earningsCalendar", [])
                if earnings:
                    next_date = pd.Timestamp(earnings[0]["date"]).date()
                    days_away = (next_date - date.today()).days
                    return days_away if 0 <= days_away <= 14 else None
        except Exception:
            pass
    return None

@_thread_cache(ttl=300)
def fetch_iv_rank(ticker):
    try:
        hist = _yf_download(ticker, period="1y", interval="1d")
        if hist is not None: hist = hist.reset_index()
        if hist is None or hist.empty or len(hist) < 30: return None, None
        hist = _clean_df(hist)
        close_col = "close" if "close" in hist.columns else "Close"
        closes = _col(hist, close_col) if close_col in hist.columns else None
        if closes is None: return None, None
        log_ret    = np.log(closes / closes.shift(1)).dropna()
        rolling_hv = log_ret.rolling(20).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()
        current_hv = float(rolling_hv.iloc[-1])
        hv_low     = float(rolling_hv.min())
        hv_high    = float(rolling_hv.max())
        if hv_high == hv_low: return 50, current_hv
        iv_rank = int((current_hv - hv_low) / (hv_high - hv_low) * 100)
        return iv_rank, current_hv
    except Exception:
        return None, None

def calc_rsi(close, period=14):
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def estimate_delta(price, strike, dte, iv=0.45, is_call=True):
    T = max(dte/365, 0.001)
    try:
        d1  = (math.log(price/strike) + (0.05 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
        nd1 = 1 / (1 + math.exp(-1.7*d1))
        return nd1 if is_call else nd1 - 1
    except:
        return 0.5

# US market holidays — options expiring on these days shift to prior trading day
_MARKET_HOLIDAYS = {
    date(2026, 1,  1),  # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4,  3),  # Good Friday 2026 ← this was causing April 3 vs April 2 bug
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7,  3),  # Independence Day (observed)
    date(2026, 9,  7),  # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
    date(2027, 1,  1),  # New Year's Day 2027
    date(2027, 4, 18),  # Good Friday 2027
}

def get_expiration_date(dte_target):
    today   = date.today()
    d       = today
    fridays = []
    while len(fridays) < 16:
        d += timedelta(days=1)
        if d.weekday() == 4:  # Friday
            # If this Friday is a market holiday, options expire Thursday instead
            exp = d - timedelta(days=1) if d in _MARKET_HOLIDAYS else d
            fridays.append(exp)
    valid = [f for f in fridays if (f - today).days >= 5]
    return min(valid, key=lambda f: abs((f - (today + timedelta(days=dte_target))).days))

def estimate_move_timeframe(pattern_label):
    if "Double" in pattern_label:  est_days = 21
    elif "Break" in pattern_label: est_days = 14
    else:                          est_days = 10
    return est_days, int(est_days * 1.5)

@_thread_cache(ttl=600)
def fetch_real_strikes(ticker, expiration_str):
    """
    Fetch actually listed strikes from FMP options chain for a given expiration.
    Returns sorted list of floats, or None if unavailable.
    expiration_str format: 'YYYY-MM-DD'
    """
    if not FMP_API_KEY or not ticker:
        return None
    try:
        import requests as _req
        url = (
            "https://financialmodelingprep.com/api/v3/options/%s"
            "?apikey=%s" % (ticker.upper(), FMP_API_KEY)
        )
        r = _req.get(url, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        strikes = sorted(set(
            float(item["strike"])
            for item in data
            if item.get("expiration") == expiration_str and item.get("strike")
        ))
        return strikes if len(strikes) >= 3 else None
    except Exception:
        return None

def snap_to_chain(price, raw, real_strikes=None):
    """
    Snap raw strike to nearest actually listed strike.
    Uses real FMP chain if available, falls back to exchange increment logic.
    """
    if real_strikes:
        return min(real_strikes, key=lambda s: abs(s - raw))
    # Fallback increment logic — conservative (wider increments = safer)
    if price < 25:
        increment = 0.50
    elif price < 50:
        increment = 1.0
    elif price < 100:
        increment = 1.0
    elif price < 150:
        # Check if raw snaps cleanly to $5 — if so use $1, else use $5 for safety
        snap5 = round(round(raw / 5.0) * 5.0, 2)
        snap1 = round(round(raw / 1.0) * 1.0, 2)
        increment = 1.0 if snap1 == snap5 else 5.0
    elif price < 500:
        increment = 5.0
    else:
        increment = 10.0
    return round(round(raw / increment) * increment, 2)

def calc_trade(entry, stop, target, direction, days_to_exp, account, risk_pct, current_price, iv=0.45, atr=None, trade_style="swing", ticker=""):
    import math as _math
    # Guard against NaN/None prices (market closed, no data)
    def _clean(v, fallback=0.0):
        try:
            f = float(v)
            return fallback if (_math.isnan(f) or _math.isinf(f)) else f
        except (TypeError, ValueError):
            return fallback

    current_price = _clean(current_price, 100.0)
    entry  = _clean(entry,  current_price)
    stop   = _clean(stop,   current_price * 0.97)
    target = _clean(target, current_price * 1.05)

    is_call    = direction == "bullish"
    exp_date   = get_expiration_date(days_to_exp)
    actual_dte = max((exp_date - date.today()).days, 1)

    # Strike selection: Quick = ATM, Swing = slight OTM
    if trade_style == "quick":
        raw_strike = current_price
    else:
        raw_strike = current_price * 1.02 if is_call else current_price * 0.98

    # Snap to real available strike — fetch actual chain from FMP first
    exp_str     = exp_date.strftime("%Y-%m-%d")
    real_strikes = fetch_real_strikes(ticker, exp_str) if FMP_API_KEY else None
    strike = snap_to_chain(current_price, raw_strike, real_strikes)

    # IV adjustment: quick trades use higher IV estimate (short-dated premiums are inflated)
    iv_adj  = min(iv * 1.3, 0.80) if actual_dte <= 7 else iv
    # ATM option premium approximation: price * IV * sqrt(DTE/365) * ~0.4 for ATM
    # But 0.4 is too high for high-priced stocks - use 0.38 for quick (ATM) 0.25 for swing (OTM)
    otm_discount = 0.38 if trade_style == "quick" else 0.22
    premium = round(current_price * iv_adj * (max(actual_dte, 1)/365)**0.5 * otm_discount, 2)
    premium = max(premium, 0.05)
    breakeven = (strike + premium) if is_call else (strike - premium)

    # ── Target sanity check ───────────────────────────────────────────────────
    # Use the pattern's measured move as-is. Only apply a sanity cap so we never
    # show a target that requires an unrealistic price move.
    # Cap: target cannot be more than 20% away from current price for stocks,
    # or more than 4x ATR away. Whichever is less restrictive.
    max_move_pct = 0.20  # 20% max move
    if atr and atr > 0:
        # Allow up to 6x ATR as the measured move (generous for double bottoms)
        atr_cap_pct = (atr * 6) / current_price
        max_move_pct = max(max_move_pct, min(atr_cap_pct, 0.35))

    if is_call:
        max_target   = round(current_price * (1 + max_move_pct), 2)
        stock_target = min(target, max_target)
        # CALL target must always be ABOVE current price
        if stock_target <= current_price:
            stock_target = round(current_price * 1.05, 2)
    else:
        min_target   = round(current_price * (1 - max_move_pct), 2)
        stock_target = max(target, min_target)
        # PUT target must always be BELOW current price
        if stock_target >= current_price:
            stock_target = round(current_price * 0.95, 2)

    # ATR-based move probability
    move_needed = abs(stock_target - current_price)
    atr_multiples = round(move_needed / atr, 1) if atr and atr > 0 else None
    if atr_multiples is not None:
        if atr_multiples <= 2.0:   target_realistic = "Likely"
        elif atr_multiples <= 4.0: target_realistic = "Possible"
        else:                       target_realistic = "Ambitious"
    else:
        target_realistic = "Unknown"

    # Move pct for display
    move_pct = round((move_needed / current_price) * 100, 1)

    delta     = estimate_delta(current_price, strike, actual_dte, iv, is_call)
    abs_delta = abs(delta)
    max_loss_per     = premium * 100
    contracts        = max(1, int((account * risk_pct) / max_loss_per)) if max_loss_per > 0 else 1
    position_dollars = round(max_loss_per * contracts, 2)
    pct_of_account   = round((position_dollars / account) * 100, 1) if account > 0 else 0

    # R:R on the stock move (pattern level)
    rr_stock = round(abs(stock_target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0

    # R:R on the option — use premium % targets (how traders actually think)
    # Quick: target +30%, stop -20% = 1.5x RR
    # Swing: target +50%, stop -20% = 2.5x RR
    profit_target_pct = 0.30 if trade_style == "quick" else 0.50
    stop_loss_pct     = 0.20
    rr_option         = round(profit_target_pct / stop_loss_pct, 2)

    # Dollar profit estimate based on premium % target
    option_gain_per_share = premium * profit_target_pct
    profit_per   = round(option_gain_per_share * 100, 2)
    total_profit = round(profit_per * contracts, 2)

    # Also compute dollar-based profit estimate for display
    option_gain_per_share = premium * profit_target_pct
    profit_per   = round(option_gain_per_share * 100, 2)
    total_profit = round(profit_per * contracts, 2)

    return {
        "type": "CALL" if is_call else "PUT",
        "strike": strike, "premium": premium, "breakeven": round(breakeven, 2),
        "max_loss": position_dollars, "contracts": contracts,
        "position_dollars": position_dollars, "pct_of_account": pct_of_account,
        "profit_at_target": total_profit,
        "target": round(stock_target, 2), "stop": round(stop, 2), "entry": round(entry, 2),
        "rr": rr_stock, "rr_option": rr_option,
        "delta": round(abs_delta, 2), "delta_ok": 0.35 <= abs_delta <= 0.85,
        "expiration": exp_date.strftime("%b %d, %Y"), "actual_dte": actual_dte,
        "exit_take_half": round(premium * 2.0, 2),
        "exit_stop_stock": round(stop, 2),
        "move_pct": move_pct,
        "atr_multiples": atr_multiples,
        "target_realistic": target_realistic,
    }

def detect_rsi_divergence(df):
    if len(df) < 30: return None
    close = df["close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100/(1+(gain/loss)))
    rc = close.iloc[-20:].values
    rr = rsi.iloc[-20:].values
    plows, rlows, phighs, rhighs = [], [], [], []
    for i in range(2, len(rc)-2):
        if rc[i] < rc[i-1] and rc[i] < rc[i+1]: plows.append((i,rc[i]));  rlows.append((i,rr[i]))
        if rc[i] > rc[i-1] and rc[i] > rc[i+1]: phighs.append((i,rc[i])); rhighs.append((i,rr[i]))
    if len(plows)>=2 and len(rlows)>=2:
        p1,p2 = plows[-2][1],plows[-1][1]; r1,r2 = rlows[-2][1],rlows[-1][1]
        if p2<p1 and r2>r1: return {"type":"bullish","label":"Bullish RSI Divergence","detail":f"Price lower low (${p2:.2f}) but RSI higher low ({r2:.0f})"}
    if len(phighs)>=2 and len(rhighs)>=2:
        p1,p2 = phighs[-2][1],phighs[-1][1]; r1,r2 = rhighs[-2][1],rhighs[-1][1]
        if p2>p1 and r2<r1: return {"type":"bearish","label":"Bearish RSI Divergence","detail":f"Price higher high (${p2:.2f}) but RSI lower high ({r2:.0f})"}
    return None

def run_seven_point_gate(df, sig, opt, iv_rank, earnings_days, dte_used):
    is_bull = sig["direction"] == "bullish"
    price   = float(df["close"].iloc[-1])
    _, dte_rec = estimate_move_timeframe(sig["pattern_label"])

    iv_ok     = iv_rank is not None and iv_rank < 60
    iv_label  = "Volatility Environment: Favorable" if iv_ok else ("Volatility Environment: Elevated" if iv_rank is not None else "Volatility Environment: Unavailable")

    avg_vol   = float(df["volume"].iloc[-20:].mean())
    cur_vol   = float(df["volume"].iloc[-3:].mean())
    # Block zero volume entirely — market closed or no activity on contract
    if cur_vol == 0 or avg_vol == 0:
        vol_ok    = False
        vol_label = "Volume: No Activity"
    elif cur_vol > avg_vol * 1.1:
        vol_ok    = True
        vol_label = "Volume: Confirming"
    else:
        vol_ok    = False
        vol_label = "Volume: Not Confirming"

    div       = detect_rsi_divergence(df)
    div_ok    = div is not None and div["type"] == ("bullish" if is_bull else "bearish")
    div_label = "Momentum Divergence: Confirmed" if div_ok else "Momentum Divergence: Not Detected"

    entry_dist = abs(opt["entry"] - price) / price * 100
    neck_ok    = entry_dist < 3.0
    neck_label = "Entry Timing: Valid" if neck_ok else "Entry Timing: Stale"

    rr_ok    = opt["rr"] >= 2.0
    rr_label = "Risk/Reward: Acceptable" if rr_ok else "Risk/Reward: Insufficient"

    dte_ok    = dte_used >= dte_rec
    dte_label = "Expiration: Adequate" if dte_ok else "Expiration: Too Short"

    earn_ok    = earnings_days is None or earnings_days > 7
    earn_label = "Earnings Risk: Clear" if earn_ok else "Earnings Risk: BLOCKED"

    gates = {
        "Volatility":      {"pass": iv_ok,   "label": iv_label,   "critical": False},
        "Volume":          {"pass": vol_ok,  "label": vol_label,  "critical": False},
        "Momentum":        {"pass": div_ok,  "label": div_label,  "critical": False},
        "Entry Timing":    {"pass": neck_ok, "label": neck_label, "critical": True},
        "Risk/Reward":     {"pass": rr_ok,   "label": rr_label,   "critical": True},
        "Expiration":      {"pass": dte_ok,  "label": dte_label,  "critical": False},
        "Earnings":        {"pass": earn_ok, "label": earn_label, "critical": True},
    }
    passed          = sum(1 for g in gates.values() if g["pass"])
    critical_pass   = all(g["pass"] for g in gates.values() if g["critical"])
    non_crit_pass   = sum(1 for k,g in gates.items() if not g["critical"] and g["pass"])
    elevate         = critical_pass and non_crit_pass >= 2
    return gates, passed, elevate

# ── Entry confirmation candles ────────────────────────────────────────────────
def check_entry_confirmation(df, direction):
    """
    Checks last candles to see if price is moving in signal direction.
    Calls: need 2 consecutive green candles, each close higher than previous.
    Puts:  need 2 consecutive red candles, each close lower than previous.
    Returns status: CONFIRMED / WAITING / AGAINST
    """
    if len(df) < 4:
        return {"confirmed": False, "status": "WAITING", "candles": [], "message": "Not enough data"}

    recent = df.tail(5)
    is_bull = direction == "bullish"

    candle_dirs = []
    for _, row in recent.iterrows():
        if float(row["close"]) > float(row["open"]):   candle_dirs.append("green")
        elif float(row["close"]) < float(row["open"]): candle_dirs.append("red")
        else:                                            candle_dirs.append("doji")

    c1 = recent.iloc[-2]
    c2 = recent.iloc[-1]

    if is_bull:
        both_green    = float(c1["close"]) > float(c1["open"]) and float(c2["close"]) > float(c2["open"])
        higher_closes = float(c2["close"]) > float(c1["close"])
        confirmed     = both_green and higher_closes
        last_green    = candle_dirs[-1] == "green"
        if confirmed:
            status  = "CONFIRMED"
            message = f"2 bullish candles confirmed - buyers in control. Entry window open near ${float(c2['close']):.2f}"
        elif last_green:
            status  = "WAITING"
            message = "1 of 2 bullish candles printed. Need 1 more green candle closing higher."
        else:
            status  = "AGAINST"
            message = "Price still dropping. Signal valid but entry is early - wait for 2 consecutive green candles."
    else:
        both_red     = float(c1["close"]) < float(c1["open"]) and float(c2["close"]) < float(c2["open"])
        lower_closes = float(c2["close"]) < float(c1["close"])
        confirmed    = both_red and lower_closes
        last_red     = candle_dirs[-1] == "red"
        if confirmed:
            status  = "CONFIRMED"
            message = f"2 bearish candles confirmed - sellers in control. Entry window open near ${float(c2['close']):.2f}"
        elif last_red:
            status  = "WAITING"
            message = "1 of 2 bearish candles printed. Need 1 more red candle closing lower."
        else:
            status  = "AGAINST"
            message = "Price still climbing. Signal valid but entry is early - wait for 2 consecutive red candles."

    return {"confirmed": confirmed, "status": status, "candles": candle_dirs, "message": message}

# ── Watch queue ───────────────────────────────────────────────────────────────
WATCH_TIMEOUT_MINS = 30

def init_watch_queue():
    if "watch_queue" not in st.session_state:
        st.session_state.watch_queue = {}

def init_auto_scan():
    if "auto_scan_enabled"  not in st.session_state: st.session_state.auto_scan_enabled  = False
    if "auto_scan_results"  not in st.session_state: st.session_state.auto_scan_results   = None
    if "auto_scan_last_run" not in st.session_state: st.session_state.auto_scan_last_run  = None
    if "auto_scan_go_now"   not in st.session_state: st.session_state.auto_scan_go_now    = []
    if "auto_scan_prev_go"  not in st.session_state: st.session_state.auto_scan_prev_go   = []
    if "auto_scan_watching" not in st.session_state: st.session_state.auto_scan_watching  = []
    if "auto_scan_on_deck"  not in st.session_state: st.session_state.auto_scan_on_deck   = []
    if "auto_scan_mkt"      not in st.session_state: st.session_state.auto_scan_mkt       = "neutral"
    if "auto_scan_settings" not in st.session_state: st.session_state.auto_scan_settings  = {
        "scan_list": "watchlist", "max_premium": 5.0, "style": "both"
    }
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []  # loaded from Supabase after functions defined
    if "user_watchlist"      not in st.session_state: st.session_state.user_watchlist       = list(DEFAULT_WATCHLIST)
    if "watchlist_loaded"    not in st.session_state: st.session_state.watchlist_loaded     = False
    if "onboarding_complete" not in st.session_state: st.session_state.onboarding_complete  = False
    if "onboarding_step"     not in st.session_state: st.session_state.onboarding_step      = 1
    if "user_id"             not in st.session_state: st.session_state.user_id              = None
    if "tos_agreed"          not in st.session_state: st.session_state.tos_agreed           = False

init_auto_scan()
# init_user_watchlist() called later after function is defined

def _save_watch_queue_db():
    """Persist watch queue to Supabase. Converts datetimes to ISO strings."""
    user_id = st.session_state.get("user_id")
    if not user_id: return
    import json as _j
    wq = st.session_state.get("watch_queue", {})
    serializable = {}
    for k, item in wq.items():
        _item = dict(item)
        for ts_field in ["added_at", "last_checked"]:
            if isinstance(_item.get(ts_field), datetime):
                _item[ts_field] = _item[ts_field].isoformat()
        serializable[k] = _item
    save_user_data(user_id, watch_queue=serializable)

def add_to_watch_queue(ticker, direction, sig, opt):
    init_watch_queue()
    key = f"{ticker}_{direction}"
    if key not in st.session_state.watch_queue:
        st.session_state.watch_queue[key] = {
            "ticker":    ticker,
            "direction": direction,
            "action":    "CALL" if direction == "bullish" else "PUT",
            "style":     sig.get("trade_style", "swing"),
            "strike":    opt.get("strike", 0),
            "entry":     opt.get("entry", 0),
            "target":    opt.get("target", 0),
            "stop":      opt.get("stop", 0),
            "pattern":   sig.get("pattern_label", sig.get("pattern", "Signal")),
            "confidence":sig.get("confidence", 0),
            "added_at":  datetime.now(),
            "last_checked": None,
            "status":    "WAITING",
            "message":   "Watching for 2 confirmation candles...",
            "alerted":   False,
        }
        _save_watch_queue_db()

def remove_from_watch_queue(key):
    init_watch_queue()
    if key in st.session_state.watch_queue:
        del st.session_state.watch_queue[key]
    # Always save even if key wasn't found — clears any stale Supabase data
    _save_watch_queue_db()

def clear_watch_queue():
    """Nuke entire watch queue from session and Supabase."""
    st.session_state.watch_queue = {}
    user_id = st.session_state.get("user_id")
    if user_id:
        save_user_data(user_id, watch_queue={})

def run_background_watch_checks(tf_mult, tf_span, tf_days):
    """
    Runs on every Watch Queue tab refresh.
    Fetches fresh data bypassing cache. Uses signal's actual timeframe.
    """
    init_watch_queue()
    queue = st.session_state.watch_queue
    any_new_confirm = False
    to_remove = []

    for key, item in queue.items():
        elapsed = (datetime.now() - item["added_at"]).total_seconds() / 60
        timeout = 30 if item.get("style", "swing") == "quick" else 240
        if elapsed > timeout:
            to_remove.append(key)
            continue
        try:
            style = item.get("style", "swing")
            interval, period = ("5m", "2d") if style == "quick" else ("1h", "14d")

            raw = _yf_download(item["ticker"], period=period, interval=interval)
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                item["message"] = "Data unavailable - retrying..."
                continue

            if hasattr(raw, 'reset_index'):
                raw = raw.reset_index()
            fresh_df = _clean_df(raw)
            for col in ["datetime", "date", "timestamp", "index"]:
                if col in fresh_df.columns:
                    fresh_df = fresh_df.rename(columns={col: "timestamp"})
                    break

            conf = check_entry_confirmation(fresh_df, item["direction"])
            was_confirmed_before = item["status"] == "CONFIRMED"
            item["status"]       = conf["status"]
            item["message"]      = conf["message"]
            item["candles"]      = conf.get("candles", [])
            item["last_checked"] = datetime.now()

            if conf["confirmed"] and not was_confirmed_before and not item["alerted"]:
                item["alerted"]   = True
                any_new_confirm   = True
        except Exception:
            item["message"] = "Data fetch failed - retrying..."

    for key in to_remove:
        del queue[key]

    st.session_state.watch_queue = queue
    return any_new_confirm
def get_trend(df):
    close=df["close"]; high=df["high"]; low=df["low"]
    price=float(close.iloc[-1])
    ema20=float(close.ewm(span=20).mean().iloc[-1])
    tp   =(high+low+close)/3
    vwap =float((tp*df["volume"]).cumsum().iloc[-1]/df["volume"].cumsum().iloc[-1])
    rsi  =calc_rsi(close)
    recent   = df.tail(10)
    up_vol   = float(recent[recent["close"]>=recent["open"]]["volume"].mean() or 0)
    down_vol = float(recent[recent["close"]< recent["open"]]["volume"].mean() or 0)
    hl = [float(high.iloc[i]) for i in range(-10,0)]
    ll = [float(low.iloc[i])  for i in range(-10,0)]
    lower_highs = len(hl)>=9 and hl[-1]<hl[-5]<hl[-9]
    higher_lows = len(ll)>=9 and ll[-1]>ll[-5]>ll[-9]
    bear={"below_ema":{"pass":price<ema20,"label":"Trend Filter: Aligned" if price<ema20 else "Trend Filter: Against"},
          "below_vwap":{"pass":price<vwap,"label":"Intraday Bias: Aligned" if price<vwap else "Intraday Bias: Against"},
          "rsi_high":  {"pass":rsi>55,    "label":f"RSI elevated ({rsi:.0f})"},
          "down_vol":  {"pass":down_vol>up_vol,"label":"Heavier volume on down bars"},
          "lower_highs":{"pass":lower_highs,"label":"Lower highs forming"}}
    bull={"above_ema": {"pass":price>ema20,"label":"Trend Filter: Aligned" if price>ema20 else "Trend Filter: Against"},
          "above_vwap":{"pass":price>vwap, "label":"Intraday Bias: Aligned" if price>vwap else "Intraday Bias: Against"},
          "rsi_low":   {"pass":rsi<45,     "label":f"RSI low ({rsi:.0f})"},
          "up_vol":    {"pass":up_vol>down_vol,"label":"Heavier volume on up bars"},
          "higher_lows":{"pass":higher_lows,"label":"Higher lows forming"}}
    bear_score = sum(1 for f in bear.values() if f["pass"])
    bull_score = sum(1 for f in bull.values() if f["pass"])
    if bear_score >= bull_score: return "bearish",bear_score,bear,ema20,vwap,rsi
    return "bullish",bull_score,bull,ema20,vwap,rsi

def detect_market_regime(df):
    """
    Determines if market is TRENDING or CHOPPY using ATR expansion and directional consistency.
    Trending = ATR expanding + price making consistent directional moves.
    Choppy   = ATR contracting or price reversing frequently.
    Returns: regime ("trending"/"choppy"), strength (0-100)
    """
    if len(df) < 30: return "unknown", 50
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    # ATR trend: compare recent 7-bar ATR to prior 14-bar ATR
    tr = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    atr_recent = float(tr.iloc[-7:].mean())
    atr_prior  = float(tr.iloc[-21:-7].mean())
    atr_expanding = atr_recent > atr_prior * 1.1
    # Directional consistency: how many of last 10 bars close in same direction
    last10 = df.tail(10)
    bull_bars = int((last10["close"] > last10["open"]).sum())
    bear_bars = 10 - bull_bars
    directional = max(bull_bars, bear_bars)  # 5=choppy, 10=strong trend
    consistency_score = int((directional - 5) / 5 * 100)  # 0-100
    if atr_expanding and directional >= 7:
        regime = "trending"
        strength = min(100, int(consistency_score * 1.2))
    elif not atr_expanding and directional <= 6:
        regime = "choppy"
        strength = max(0, 100 - consistency_score)
    else:
        regime = "trending" if directional >= 7 else "choppy"
        strength = consistency_score
    return regime, strength

@_thread_cache(ttl=300)
@_thread_cache(ttl=300)
def check_liquidity(ticker):
    """
    Check options liquidity via Polygon options snapshot API.
    Returns: (liquid bool, avg_volume, avg_oi, message)
    Liquid = avg volume >= 50 AND avg OI >= 100 on nearest expiry calls.
    """
    if not POLYGON_API_KEY:
        return True, 0, 0, "Verify OI manually"
    try:
        import requests as _req
        # Get options contracts for this ticker - nearest expiry, calls
        url = (
            "https://api.polygon.io/v3/snapshot/options/%s"
            "?limit=25&apiKey=%s" % (ticker.upper(), POLYGON_API_KEY)
        )
        r = _req.get(url, timeout=4)
        if r.status_code != 200:
            return True, 0, 0, "Liquidity unavailable"
        data = r.json()
        results = data.get("results", [])
        if not results:
            return False, 0, 0, "No options data found"

        # Pull volume and OI from snapshot
        volumes = []
        ois     = []
        for item in results:
            day  = item.get("day", {})
            det  = item.get("details", {})
            vol  = day.get("volume", 0) or 0
            oi   = item.get("open_interest", 0) or 0
            volumes.append(float(vol))
            ois.append(float(oi))

        avg_vol = round(sum(volumes) / len(volumes), 0) if volumes else 0
        avg_oi  = round(sum(ois)     / len(ois),     0) if ois     else 0
        liquid  = avg_vol >= 50 and avg_oi >= 100

        if liquid:
            msg = "Vol %.0f · OI %.0f" % (avg_vol, avg_oi)
        elif avg_oi < 100:
            msg = "⚠️ Low OI (%.0f) - wide spreads likely" % avg_oi
        else:
            msg = "⚠️ Low volume (%.0f) - hard to exit" % avg_vol

        return liquid, avg_vol, avg_oi, msg
    except Exception as e:
        return True, 0, 0, "Liquidity check error"

def score_setup(df, setup):
    """
    Confidence scoring - base layer (50 pts max).
    Final score = base + TF confluence + extra confluence = 50-100.
    """
    close  = _col(df, "close"); high = _col(df, "high"); low = _col(df, "low")
    vol    = _col(df, "volume")
    price  = float(close.iloc[-1])
    is_bull = setup.direction == "bullish"
    ema20  = float(close.ewm(span=20).mean().iloc[-1])
    tp     = (high + low + close) / 3
    vwap_num = float((tp * vol).cumsum().iloc[-1])
    vwap_den = float(vol.cumsum().iloc[-1])
    vwap   = vwap_num / vwap_den if vwap_den > 0 else price
    rsi    = calc_rsi(close)
    avg_vol = float(vol.iloc[-20:].mean())
    cur_vol = float(vol.iloc[-1])

    rsi_div = detect_rsi_divergence(df)
    rsi_div_match = rsi_div is not None and (
        (is_bull and rsi_div.get("type") == "bullish") or
        (not is_bull and rsi_div.get("type") == "bearish")
    )
    vol_expanding = cur_vol > avg_vol * 1.3
    vol_present   = cur_vol > avg_vol * 1.1

    factors = {
        "Pattern":{"pass":True,          "label":"Pattern confirmed"},
        "RSI Div":{"pass":rsi_div_match, "label":"Price Divergence: Confirmed" if rsi_div_match else "Price Divergence: Not Detected"},
        "Volume": {"pass":vol_expanding, "label":"Volume: Confirmed" if vol_expanding else ("Volume: Present" if vol_present else "Volume: Insufficient")},
        "EMA":    {"pass":(price>ema20 if is_bull else price<ema20),"label":"Trend Filter: Aligned" if (price>ema20 if is_bull else price<ema20) else "Trend Filter: Against"},
        "VWAP":   {"pass":(price>vwap  if is_bull else price<vwap), "label":"Intraday Bias: Aligned" if (price>vwap if is_bull else price<vwap) else "Intraday Bias: Against"},
    }

    # Base score: each factor = 10pts, max 50
    raw_score  = sum(1 for f in factors.values() if f["pass"])
    base_score = raw_score * 10  # 0-50

    return factors, raw_score, base_score, rsi, vwap, ema20

def calc_quick_levels(price, direction, atr):
    """
    For QUICK trades: ATR-based levels with proper fallback.
    Target = 1.0x ATR, Stop = 0.5x ATR.
    Fallback: uses 1.5% of price if ATR is None/zero/unreasonably small.
    """
    min_atr = price * 0.005
    if not atr or atr <= 0 or atr < min_atr:
        atr = max(price * 0.015, min_atr)
    is_bull = direction == "bullish"
    entry   = round(price, 2)
    target  = round(price + atr * 1.0, 2) if is_bull else round(price - atr * 1.0, 2)
    stop    = round(price - atr * 0.5, 2) if is_bull else round(price + atr * 0.5, 2)
    if is_bull and stop >= price:
        stop = round(price * 0.97, 2)
    if not is_bull and stop <= price:
        stop = round(price * 1.03, 2)
    return entry, target, stop


@_thread_cache(ttl=60)
def _fetch_tf(ticker, interval, period):
    """
    Module-level cached data fetcher for multi-timeframe data.
    Handles both yfinance (index-based datetime) and Finnhub (datetime column) formats.
    """
    try:
        df = _yf_download(ticker, period=period, interval=interval, prepost=True)
        if df is None or df.empty:
            return None

        # Reset index to bring date/datetime from index to column (yfinance)
        df = df.reset_index()
        df = _clean_df(df)

        # Normalize timestamp column — could be datetime, date, or already timestamp
        for col in ["datetime", "date", "timestamp", "index"]:
            if col in df.columns:
                df = df.rename(columns={col: "timestamp"})
                break

        # Must have timestamp column at this point
        if "timestamp" not in df.columns:
            return None

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None

        return df[required].dropna().reset_index(drop=True)
    except Exception:
        return None


def fetch_multi_tf(ticker, trade_style):
    """
    Fetches the correct timeframes automatically based on trade style.
    Quick: 5min (primary) + 15min (confirmation)
    Swing: 1hr (primary) + 4hr (confirmation) + Daily (trend anchor)
    Returns dict of {label: df}
    """
    if trade_style == "quick":
        tf5  = _fetch_tf(ticker, "5m",  "5d")
        tf15 = _fetch_tf(ticker, "15m", "5d")
        return {
            "5min":  tf5  if tf5  is not None and len(tf5)  > 20 else None,
            "15min": tf15 if tf15 is not None and len(tf15) > 20 else None,
        }
    else:
        tf1h = _fetch_tf(ticker, "1h", "30d")
        tf1d = _fetch_tf(ticker, "1d", "90d")
        return {
            "1hr":   tf1h if tf1h is not None and len(tf1h) > 20 else None,
            "4hr":   tf1h if tf1h is not None and len(tf1h) > 40 else None,  # reuse 1h data
            "daily": tf1d if tf1d is not None and len(tf1d) > 20 else None,
        }

def detect_squeeze(df, direction):
    """
    Bollinger Band / Keltner Channel squeeze detector.
    Three states:
      "firing"  - was in squeeze, just broke out in signal direction (BEST - enter now)
      "squeeze" - compression active, building energy, watching for break
      "none"    - normal volatility, no edge from squeeze

    A firing squeeze is one of the strongest options entry signals because:
      - IV is low during compression = cheap premium
      - Volatility expansion after break inflates option value fast
    """
    if df is None or len(df) < 25:
        return "none", 0

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    is_bull = direction == "bullish"

    # Bollinger Bands (20, 2)
    sma    = close.rolling(20).mean()
    std    = close.rolling(20).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2

    # True Range + ATR (20)
    import pandas as pd
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()

    # Keltner Channels (20 SMA ± 1.5 ATR)
    upper_kc = sma + atr * 1.5
    lower_kc = sma - atr * 1.5

    # Squeeze = BB inside KC
    in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    # Compression ratio - how tight is the squeeze? (0-100, higher = tighter)
    bb_width  = float((upper_bb - lower_bb).iloc[-1])
    kc_width  = float((upper_kc - lower_kc).iloc[-1])
    compression = round(max(0, min(100, (1 - bb_width / kc_width) * 100)), 1) if kc_width > 0 else 0

    curr_squeeze = bool(in_squeeze.iloc[-1])
    prev_squeeze = bool(in_squeeze.iloc[-2]) if len(in_squeeze) >= 2 else False

    # Firing = was in squeeze last bar, now broke out in signal direction
    curr_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    broke_up   = curr_close > prev_close and curr_close > float(sma.iloc[-1])
    broke_down = curr_close < prev_close and curr_close < float(sma.iloc[-1])

    if prev_squeeze and not curr_squeeze:
        # Just exited squeeze - check direction
        if (is_bull and broke_up) or (not is_bull and broke_down):
            return "firing", compression
        else:
            return "none", compression  # broke wrong direction, skip

    if curr_squeeze:
        return "squeeze", compression

    return "none", compression


def check_vwap_confluence(df_5min, direction):
    """
    Quick trade extra confluence: VWAP reclaim/rejection on 5min.
    For calls: previous candle closed BELOW vwap, current candle closes ABOVE = actual reclaim
    For puts:  previous candle closed ABOVE vwap, current candle closes BELOW = actual rejection
    Returns: (passes: bool, label: str)
    """
    if df_5min is None or len(df_5min) < 5:
        return False, "5min data unavailable"
    close = _col(df_5min, "close")
    high  = _col(df_5min, "high")
    low   = _col(df_5min, "low")
    vol   = _col(df_5min, "volume")
    tp    = (high + low + close) / 3
    vwap_num = float((tp * vol).cumsum().iloc[-1])
    vwap_den = float(vol.cumsum().iloc[-1])
    vwap  = vwap_num / vwap_den if vwap_den > 0 else float(close.iloc[-1])
    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    is_bull = direction == "bullish"
    if is_bull:
        reclaim = prev < vwap and price > vwap
        holding = prev > vwap and price > vwap
        passes  = reclaim or holding
        if reclaim:
            label = "5min Intraday: Strong Reclaim ✅"
        elif holding:
            label = "5min Intraday: Holding Bullish"
        else:
            label = "5min Intraday: Waiting for Reclaim"
    else:
        rejection = prev > vwap and price < vwap
        holding   = prev < vwap and price < vwap
        passes    = rejection or holding
        if rejection:
            label = "5min Intraday: Strong Rejection ✅"
        elif holding:
            label = "5min Intraday: Holding Bearish"
        else:
            label = "5min Intraday: Waiting for Rejection"
    return passes, label
    is_bull = direction == "bullish"
    if is_bull:
        # Actual reclaim: prev closed below, current closed above
        reclaim = prev < vwap and price > vwap
        # Also accept: holding above VWAP with prev also above (momentum continuation)
        holding = prev > vwap and price > vwap
        passes  = reclaim or holding
        if reclaim:
            label = "5min Intraday: Strong Reclaim ✅"
        elif holding:
            label = "5min Intraday: Holding Bullish"
        else:
            label = "5min Intraday: Waiting for Reclaim"
    else:
        # Actual rejection: prev closed above, current closed below
        rejection = prev > vwap and price < vwap
        holding   = prev < vwap and price < vwap
        passes    = rejection or holding
        if rejection:
            label = "5min Intraday: Strong Rejection ✅"
        elif holding:
            label = "5min Intraday: Holding Bearish"
        else:
            label = "5min Intraday: Waiting for Rejection"
    return passes, label

def check_ema50_slope(df_daily, direction):
    """
    Swing trade extra confluence: 50 EMA slope on Daily.
    Slope = (current EMA50 - EMA50 5 bars ago) / EMA50 5 bars ago
    Calls need rising slope, Puts need falling slope.
    Returns: (passes: bool, label: str)
    """
    if df_daily is None or len(df_daily) < 55:
        return False, "Daily data unavailable for EMA50"
    close  = df_daily["close"].astype(float)
    ema50  = close.ewm(span=50).mean()
    current = float(ema50.iloc[-1])
    prior   = float(ema50.iloc[-6])
    slope_pct = (current - prior) / prior * 100
    is_bull = direction == "bullish"
    if is_bull:
        passes = slope_pct > 0
        label  = "Trend Slope: Rising" if passes else "Trend Slope: Falling"
    else:
        passes = slope_pct < 0
        label  = "Trend Slope: Falling" if passes else "Trend Slope: Rising"
    return passes, label

def check_tf_trend_agreement(dfs, direction):
    """
    Checks how many timeframes agree with the signal direction.
    Fixed: properly extracts scalar float from potentially multi-column DataFrame.
    Returns (agreeing_count, total_checked, details_list)
    """
    details  = []
    agreeing = 0
    for label, df in dfs.items():
        if df is None:
            continue
        try:
            if isinstance(df.columns, pd.MultiIndex):
                close = df["close"].iloc[:, 0].astype(float)
            else:
                close = df["close"].astype(float)
            if len(close) < 21:
                continue
            ema20     = close.ewm(span=20).mean()
            ema20_val = float(ema20.iloc[-1].item() if hasattr(ema20.iloc[-1], 'item') else ema20.iloc[-1])
            price_val = float(close.iloc[-1].item() if hasattr(close.iloc[-1], 'item') else close.iloc[-1])
            trend  = "bullish" if price_val > ema20_val else "bearish"
            agrees = trend == direction
            if agrees:
                agreeing += 1
            details.append({
                "tf":     label,
                "trend":  trend,
                "agrees": agrees,
                "ema20":  round(ema20_val, 2),
                "price":  round(price_val, 2),
            })
        except Exception:
            continue
    return agreeing, len(details), details

def build_multi_tf_candidates(ticker, toggles, account, risk_pct,
                               dte, trade_style, atr=None):
    """
    Automatically fetches the right timeframes and builds candidates
    with multi-TF confluence baked in.
    Quick:  15min primary pattern + 5min trend + 5min VWAP
    Swing:  1hr primary pattern + 4hr trend + Daily EMA50 slope
    """
    tfs = fetch_multi_tf(ticker, trade_style)

    # Pick primary df for pattern detection
    if trade_style == "quick":
        _15m = tfs.get("15min"); _5m = tfs.get("5min")
        primary_df = _15m if _15m is not None else _5m
        confirm_df = _5m
    else:
        _1h  = tfs.get("1hr"); _4h = tfs.get("4hr"); _1d = tfs.get("daily")
        primary_df = _1h
        confirm_df = _4h if _4h is not None else _1d
        daily_df   = _1d

    if primary_df is None:
        return [], tfs   # no data

    # Build candidates using primary timeframe
    cands = build_candidates(primary_df, ticker, toggles, account, risk_pct,
                             dte, trade_style=trade_style, atr=atr)

    # Enhance each candidate with multi-TF confluence
    for c in cands:
        direction = c["direction"]
        tf_agreement, tf_total, tf_details = check_tf_trend_agreement(tfs, direction)

        if trade_style == "quick":
            # Extra confluence: 5min VWAP
            extra_pass, extra_label = check_vwap_confluence(confirm_df, direction)
            extra_name = "5min VWAP"
        else:
            # Extra confluence: Daily EMA50 slope
            extra_pass, extra_label = check_ema50_slope(
                tfs.get("daily"), direction)
            extra_name = "Daily EMA50"

        # ── Clean 50-100 final score ─────────────────────────────────────────────
        # Base (from score_setup): 0-50 pts  (each of 5 factors = 10 pts)
        # TF confluence:           0-30 pts  (each agreeing TF = 10 pts, max 3 TFs = 30)
        # Extra confluence:        0-20 pts  (VWAP reclaim or EMA50 slope)
        # Total max = 100, min shown = 50
        base = c["confidence"]  # already 50-95 from build_candidates

        # TF layer: up to 30 pts from agreeing timeframes
        if tf_total > 0:
            tf_pts = int((tf_agreement / tf_total) * 30)
        else:
            tf_pts = 15  # neutral if no TF data

        # Extra confluence layer: 20 pts if passes, 0 if not
        extra_pts = 20 if extra_pass else 0

        # Combine and clamp to 50-100
        raw_final = base + tf_pts + extra_pts
        # Normalize so max possible (50+30+20=100) maps cleanly
        # But base is already 50-95, so we need to scale down
        # Simpler: score = 50 + (factors/5)*25 + (tfs/total)*15 + extra*10
        factor_pts = c.get("score", 0)  # 0-5 raw factors passing
        score_50_100 = (
            50
            + int((factor_pts / 5) * 25)
            + (int((tf_agreement / tf_total) * 15) if tf_total > 0 else 8)
            + (10 if extra_pass else 0)
        )
        c["confidence"] = min(100, max(50, score_50_100))
        c["tf_details"]   = tf_details
        c["tf_agreement"] = tf_agreement
        c["tf_total"]     = tf_total
        c["extra_confluence"] = {
            "name":  extra_name,
            "pass":  extra_pass,
            "label": extra_label,
        }
        c["primary_tf"] = "15min" if trade_style == "quick" else "1hr"
        c["confirm_tfs"] = list(tfs.keys())

    return cands, tfs

def build_candidates(df, ticker, toggles, account, risk_pct, dte, trade_style="swing", atr=None):
    trend_dir,trend_score,trend_factors,t_ema,t_vwap,t_rsi = get_trend(df)
    _raw_price = df["close"].iloc[-1]
    price = float(_raw_price.iloc[0] if hasattr(_raw_price, "iloc") else _raw_price)
    regime, regime_strength = detect_market_regime(df)
    is_quick = trade_style == "quick"
    # Define regime_bonus once here so it's always available even if no patterns found
    regime_bonus = 5 if regime == "trending" else -5
    candidates = []
    raw = []
    if toggles["db"]: raw += [s for s in detect_double_bottom(df,ticker,rr_min=2.0) if s.confirmed]
    if toggles["dt"]: raw += [s for s in detect_double_top(df,ticker,rr_min=2.0)    if s.confirmed]
    if toggles["br"]: raw += [s for s in detect_break_and_retest(df,ticker,rr_min=2.0) if s.confirmed]

    for setup in raw:
        if abs(setup.entry_price - price) / price > 0.05: continue
        factors, raw_score, weighted_conf, rsi, vwap, ema20 = score_setup(df, setup)
        conflict = setup.direction != trend_dir and trend_score >= 3

        # TF alignment: handled in build_multi_tf_candidates, use base score here
        final_conf = max(50, min(95, 50 + weighted_conf))

        if is_quick:
            q_entry, q_target, q_stop = calc_quick_levels(price, setup.direction, atr)
        else:
            q_entry, q_target, q_stop = setup.entry_price, setup.target, setup.stop_loss

        if conflict:
            t_entry  = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
            t_stop   = round(price*1.02,2) if trend_dir=="bearish" else round(price*0.98,2)
            if is_quick:
                _, t_target, t_stop = calc_quick_levels(price, trend_dir, atr)
                t_entry = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
            else:
                t_target = round(price*0.96,2) if trend_dir=="bearish" else round(price*1.04,2)
            conf_val = max(50, min(90, 50 + int(trend_score/5*50)))
            candidates.append({"source":"trend_override","direction":trend_dir,
                "confidence":conf_val,"score":trend_score,"factors":trend_factors,
                "conflict":True,"conflict_pattern":setup.pattern,
                "entry":t_entry,"stop":t_stop,"target":t_target,
                "pattern_label":"Trend Override","rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
                "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})
        else:
            candidates.append({"source":"pattern","direction":setup.direction,
                "confidence":final_conf,"score":raw_score,"factors":factors,"conflict":False,
                "entry":q_entry,"stop":q_stop,"target":q_target,
                "pattern_label":setup.pattern.replace("Double","Double ").replace("BreakRetest","Break & Retest"),
                "rsi":rsi,"vwap":vwap,"ema20":ema20,"rr":setup.rr_ratio,
                "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})

    if trend_score >= 3:
        t_entry  = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
        if is_quick:
            _, t_target, t_stop = calc_quick_levels(price, trend_dir, atr)
            t_entry = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
        else:
            t_stop   = round(price*1.02,2) if trend_dir=="bearish" else round(price*0.98,2)
            t_target = round(price*0.96,2) if trend_dir=="bearish" else round(price*1.04,2)
        trend_conf = max(50, min(90, 50 + int(trend_score/5*50)))
        candidates.append({"source":"trend","direction":trend_dir,
            "confidence":trend_conf,"score":trend_score,"factors":trend_factors,"conflict":False,
            "entry":t_entry,"stop":t_stop,"target":t_target,
            "pattern_label":f"{'Bearish' if trend_dir=='bearish' else 'Bullish'} Trend",
            "rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
            "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})

    seen = {}
    for c in sorted(candidates, key=lambda x:x["confidence"], reverse=True):
        k = f"{c['direction']}_{c['pattern_label']}"
        if k not in seen: seen[k] = c
    return sorted(seen.values(), key=lambda x:x["confidence"], reverse=True)[:3]

def load_journal():
    if "trade_journal" not in st.session_state: st.session_state.trade_journal = []
    return st.session_state.trade_journal

def log_trade(ticker, sig, opt, gates_passed, gates_total, elevate):
    journal = load_journal()
    journal.append({
        "Date":        datetime.now().strftime("%m/%d %H:%M"),
        "Ticker":      ticker,
        "Action":      "CALL" if sig["direction"]=="bullish" else "PUT",
        "Pattern":     sig["pattern_label"],
        "Strike":      f"${opt['strike']:.2f}",
        "Entry":       f"${opt['entry']:.2f}",
        "Target":      f"${opt['target']:.2f}",
        "Stop":        f"${opt['stop']:.2f}",
        "Premium":     f"${opt['premium']:.2f}",
        "Contracts":   opt["contracts"],
        "Max Loss":    f"${opt['max_loss']:.0f}",
        "Pot. Profit": f"${opt['profit_at_target']:,.0f}",
        "Confidence":  f"{sig['confidence']}%",
        "Gate Score":  f"{gates_passed}/{gates_total}",
        "Elevated":    "YES" if elevate else "no",
        "Expiry":      opt["expiration"],
        "Result":      "Open",
        "P&L $":       "",
    })
    st.session_state.trade_journal = journal[-200:]

def get_journal_stats():
    journal = load_journal()
    if not journal: return {}
    stats = {}
    for t in journal:
        tk = t["Ticker"]
        if tk not in stats: stats[tk] = {"total":0,"wins":0,"losses":0,"open":0,"calls":0,"puts":0}
        stats[tk]["total"] += 1
        r = t.get("Result","Open")
        if r=="Open":          stats[tk]["open"]   += 1
        elif "Win"  in r:      stats[tk]["wins"]   += 1
        elif "Loss" in r:      stats[tk]["losses"] += 1
        if t["Action"]=="CALL": stats[tk]["calls"] += 1
        else:                   stats[tk]["puts"]  += 1
    return stats

def build_share_text(ticker, sig, opt, gates_passed, gates_total, elevate, market_status):
    direction = "CALL" if sig["direction"]=="bullish" else "PUT"
    elevated  = "YES - ALL GATES PASSED" if elevate else f"NO - {gates_passed}/7 gates"
    sep = "=" * 32
    return (f"OPTIONS SCREENER v6.0 SIGNAL\n{sep}\n"
            f"{ticker} - BUY {direction}\n"
            f"Pattern:   {sig['pattern_label']}\n"
            f"Conf:      {sig['confidence']}%\n"
            f"Gate:      {gates_passed}/7 | Elevated: {elevated}\n{sep}\n"
            f"Strike:    ${opt['strike']:.2f}\n"
            f"Premium:   ${opt['premium']:.2f}/share\n"
            f"Entry:     ${opt['entry']:.2f}\n"
            f"Target:    ${opt['target']:.2f}\n"
            f"Stop:      ${opt['stop']:.2f}\n"
            f"R:R:       {opt['rr']}x\n"
            f"Delta:     {opt['delta']:.2f}\n"
            f"Contracts: {opt['contracts']}\n"
            f"Position:  ${opt['position_dollars']:.0f} ({opt['pct_of_account']}% of acct)\n"
            f"Max Loss:  ${opt['max_loss']:.0f}\n"
            f"Profit:    ${opt['profit_at_target']:,.0f}\n"
            f"Expires:   {opt['expiration']}\n{sep}\n"
            f"EXIT RULES:\n"
            f"Take 50% when option hits ${opt['exit_take_half']:.2f} (100% gain)\n"
            f"Close 100% if stock closes beyond ${opt['exit_stop_stock']:.2f}\n{sep}\n"
            f"Market: {market_status}\n"
            f"Time:   {datetime.now().strftime('%m/%d/%Y %H:%M')}\n"
            f"NOT FINANCIAL ADVICE")

# ── AI Trade Brief ───────────────────────────────────────────────────────────
def get_ai_brief(ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status):
    """
    Calls Claude API with full signal context.
    Returns a structured verdict: rating, reasoning, key risk.
    """
    import urllib.request
    import json

    is_bull     = sig["direction"] == "bullish"
    action      = "CALL" if is_bull else "PUT"
    gate_lines  = "\n".join(["  - " + k + ": " + ("PASS" if v["pass"] else "FAIL") + " (" + v["label"] + ")" for k,v in gates.items()])
    div         = detect_rsi_divergence_text(sig)

    prompt = f"""You are an expert options trader reviewing a technical setup. Give a concise professional assessment.

TICKER: {ticker}
SIGNAL: BUY {action}
Pattern: {sig['pattern_label']}
Confidence Score: {sig['confidence']}%
Gate Score: {gates_passed}/7

PRICE DATA:
- Entry: ${opt['entry']:.2f}
- Strike: ${opt['strike']:.2f}
- Target: ${opt['target']:.2f}
- Stop: ${opt['stop']:.2f}
- R:R Ratio: {opt['rr']}x
- Delta: {opt['delta']:.2f}
- Premium: ${opt['premium']:.2f}
- Expiration: {opt['expiration']}

7-POINT GATE RESULTS:
{gate_lines}

ADDITIONAL CONTEXT:
- IV Rank: {iv_rank if iv_rank is not None else 'unavailable'}%
- Earnings: {'None within 14 days' if earnings_days is None else f'In {earnings_days} days - HIGH RISK'}
- Entry timing: {conf_status}

Respond in exactly this format, no extra text:
RATING: [Strong Setup / Moderate Setup / Weak Setup / Do Not Trade]
REASONING: [2-3 sentences on why the setup quality is good or bad based on the data above]
KEY RISK: [1 sentence on the single biggest risk to this trade]
EDGE: [1 sentence on what gives this trade its edge if taken]"""

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        if "invalid x-api-key" in err.lower() or "authentication" in err.lower():
            return "RATING: Unavailable\nREASONING: API key not set or invalid. Add ANTHROPIC_API_KEY to Railway environment variables.\nKEY RISK: N/A\nEDGE: N/A"
        return "RATING: Unavailable\nREASONING: API error %s - %s\nKEY RISK: N/A\nEDGE: N/A" % (e.code, err[:100])
    except Exception as e:
        return "RATING: Unavailable\nREASONING: Connection error - %s\nKEY RISK: N/A\nEDGE: N/A" % str(e)[:80]

def detect_rsi_divergence_text(sig):
    return sig.get("rsi_div", "not checked")

def parse_ai_brief(text):
    """Parse the structured AI response into parts."""
    lines  = text.strip().splitlines()
    parsed = {}
    for line in lines:
        if line.startswith("RATING:"):    parsed["rating"]    = line.replace("RATING:","").strip()
        elif line.startswith("REASONING:"): parsed["reasoning"] = line.replace("REASONING:","").strip()
        elif line.startswith("KEY RISK:"): parsed["risk"]      = line.replace("KEY RISK:","").strip()
        elif line.startswith("EDGE:"):    parsed["edge"]      = line.replace("EDGE:","").strip()
    return parsed

def render_signal_cards(candidates, ticker, dte, trade_style, key_prefix,
                        df, current_price, atr, iv_rank, earnings_days,
                        mstatus, mtext, account_size, risk_pct,
                        htf_trend, htf_rsi, htf_ema, liq_ok):
    """Renders signal cards for a given candidate list. Called once per column."""
    if not candidates:
        st.markdown(
            "<div style='background:#111827;border:1px solid #2A2A2D;border-radius:12px;"
            "padding:20px;text-align:center;color:#A1A1A6;font-size:0.85rem'>"
            "No signals found for this mode.</div>", unsafe_allow_html=True)
        return

    rank_labels   = ["BEST","BETTER","GOOD"]
    rank_classes  = ["rank-best","rank-better","rank-good"]
    badge_classes = ["badge-best","badge-better","badge-good"]
    conf_classes  = ["conf-num-best","conf-num-better","conf-num-good"]
    rank_icons    = ["🥇","🥈","🥉"]

    for i, sig in enumerate(candidates):
        rl = rank_labels[i]  if i<3 else f"#{i+1}"
        rc = rank_classes[i] if i<3 else "rank-good"
        bc = badge_classes[i]if i<3 else "badge-good"
        cc = conf_classes[i] if i<3 else "conf-num-good"
        ri = rank_icons[i]   if i<3 else ""

        is_bull   = sig["direction"] == "bullish"
        dir_color = "#D4AF37" if is_bull else "#C1121F"
        dir_label = "BUY CALL" if is_bull else "BUY PUT"

        # Trade style badge
        sig_style = trade_style  # passed as parameter
        if sig_style == "quick":
            style_badge = "<span style='background:#1a0a3a;color:#aa88ff;font-family:monospace;font-size:0.68rem;padding:2px 7px;border-radius:10px;margin-left:6px'>⚡ QUICK</span>"
        else:
            style_badge = "<span style='background:#0a1a2a;color:#A1A1A6;font-family:monospace;font-size:0.68rem;padding:2px 7px;border-radius:10px;margin-left:6px'>📅 SWING</span>"

        # Liquidity warning (silent fail - only shows if explicitly illiquid)
        liq_warn = "" if liq_ok else "<span style='color:#F6E27A;font-size:0.75rem;margin-left:8px'>⚠ Low liquidity</span>"

        # Regime indicator (1 line, subtle)
        sig_regime   = sig.get("regime","unknown")
        regime_icon  = "📈" if sig_regime=="trending" else "↔️" if sig_regime=="choppy" else ""
        regime_label = sig_regime.upper() if sig_regime != "unknown" else ""

        conflict_html = ""
        if sig.get("conflict"):
            pname = sig.get("conflict_pattern","pattern")
            conflict_html = f"<div class='conflict-warn'>Pattern {pname} found but trend overrides - showing {'PUT' if not is_bull else 'CALL'}.</div>"

        # Quick trade warning if market closed
        quick_warn_html = ""
        if sig_style == "quick" and mstatus != "open":
            session_name = {"pre": "Pre-Market", "after": "After-Hours", "closed": "Market Closed"}.get(mstatus, "Extended Hours")
            quick_warn_html = "<div style='background:#1a150a;border:1px solid #F6E27A;border-radius:6px;padding:8px 12px;margin-bottom:6px;color:#F6E27A;font-size:0.8rem'>⚡ %s - Quick trade levels based on latest price. Use for planning only.</div>" % session_name

        dots_html = ""
        for f in sig["factors"].values():
            dot = "dot-green" if f["pass"] else "dot-red"
            dots_html += f"<div class='factor-row'><span class='{dot}'></span><span style='color:{'#F5F5F5' if f['pass'] else '#A1A1A6'}'>{f['label']}</span></div>"

        # Multi-TF confluence rows
        tf_details = sig.get("tf_details", [])
        extra_conf = sig.get("extra_confluence", {})
        tf_html = ""
        if tf_details:
            tf_html += "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #2A2A2D'>"
            tf_html += "<div style='color:#A1A1A6;font-family:monospace;font-size:0.68rem;letter-spacing:1px;margin-bottom:4px'>TIMEFRAME CONFLUENCE</div>"
            for td in tf_details:
                dot = "dot-green" if td["agrees"] else "dot-red"
                c_color = "#F5F5F5" if td["agrees"] else "#A1A1A6"
                tf_html += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + c_color + ";font-size:0.78rem'><b>" + td["tf"].upper() + ":</b> " + td["trend"].upper() + "</span></div>"
            if extra_conf:
                dot = "dot-green" if extra_conf.get("pass") else "dot-yellow"
                c_color = "#F5F5F5" if extra_conf.get("pass") else "#A1A1A6"
                tf_html += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + c_color + ";font-size:0.78rem'><b>" + str(extra_conf.get("name","")) + ":</b> " + str(extra_conf.get("label","")) + "</span></div>"
            tf_html += "</div>"

        st.markdown(f"""
        {conflict_html}
        {quick_warn_html}
        <div class='{rc}'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                    <span class='rank-badge {bc}'>{ri} {rl}</span>{style_badge}{liq_warn}
                    <div style='font-size:1.1rem;font-weight:700;color:{dir_color};margin-top:4px'>{dir_label} - {ticker}</div>
                    <div style='color:#A1A1A6;font-size:0.82rem;margin-top:2px'>{sig['pattern_label']} &nbsp;<span style='font-size:0.75rem'>{regime_icon} {regime_label}</span></div>
                </div>
                <div style='text-align:right'>
                    <div class='{cc}'>{sig['confidence']}%</div>
                    <div style='font-size:0.7rem;font-family:monospace;margin-top:2px;color:#A1A1A6'>{"GO" if sig['confidence']>=90 else "STRONG" if sig['confidence']>=80 else "WATCH" if sig['confidence']>=70 else "WEAK" if sig['confidence']>=60 else "WAIT"}</div>
                </div>
            </div>
            <div style='margin-top:10px'>{dots_html}{tf_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if sig["confidence"] >= 60:
            opt = calc_trade(sig["entry"],sig["stop"],sig["target"],sig["direction"],dte,account_size,risk_pct,current_price,atr=atr,trade_style=trade_style,ticker=ticker)
            gates, gates_passed, elevate = run_seven_point_gate(df,sig,opt,iv_rank,earnings_days,opt["actual_dte"])
            est_days, dte_rec = estimate_move_timeframe(sig["pattern_label"])
            gate_color = "#D4AF37" if gates_passed>=6 else "#F6E27A" if gates_passed>=4 else "#C1121F"
            elev_badge = "<span style='background:#D4AF3722;color:#D4AF37;padding:2px 8px;border-radius:10px;font-size:0.72rem;margin-left:8px'>PRIME SETUP</span>" if elevate else ""

            # Fibonacci confluence for signals tab
            try:
                _sig_price = float(df["close"].iloc[-1]) if "close" in df.columns else None
                _fib_sig = detect_fibonacci_confluence(df, sig["direction"], _sig_price)
            except Exception:
                _fib_sig = {"confirmed": False, "boost": 0}

            gates_dots = ""
            for gname, gdata in gates.items():
                if gdata["critical"] and not gdata["pass"]: dot = "dot-red"
                elif gdata["pass"]:                          dot = "dot-green"
                else:                                        dot = "dot-yellow"
                g_color = "#F5F5F5" if gdata["pass"] else "#A1A1A6"
                # Sanitize label - remove any characters that could break HTML
                g_label = str(gdata["label"]).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;")
                gates_dots += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + g_color + ";font-size:0.78rem'>" + g_label + "</span></div>"

            # Build as plain string - no f-string so special chars in labels cant break it
            gate_html = (
                "<div class='gate-box'>"
                "<div style='display:flex;align-items:center;margin-bottom:8px'>"
                "<span style='color:" + gate_color + ";font-family:monospace;font-size:0.78rem;font-weight:700'>7-POINT GATE: " + str(gates_passed) + "/7 PASSED</span>"
                + elev_badge +
                "</div>"
                + gates_dots +
                "<div style='color:#A1A1A6;font-size:0.75rem;margin-top:6px'>Pattern needs ~" + str(est_days) + " days to play out | Recommended DTE: " + str(dte_rec) + "+ days</div>"
                "</div>"
            )
            st.markdown(gate_html, unsafe_allow_html=True)

            # ── HTF Confluence ────────────────────────────────────────────
            if htf_trend is not None:
                htf_agrees = htf_trend == sig["direction"]
                htf_color  = "#D4AF37" if htf_agrees else "#C1121F"
                htf_icon   = "✅" if htf_agrees else "⚠️"
                htf_label  = ("DAILY TREND CONFIRMS" if htf_agrees else "DAILY TREND CONFLICTS")
                htf_detail = "Daily chart agrees - higher timeframe is aligned." if htf_agrees else "Daily chart is moving the other way. Extra caution - counter-trend trade."
                htf_html = (
                    "<div style='background:#1A1A1D;border:1px solid " + htf_color + "33;border-radius:8px;padding:10px 14px;margin-top:6px'>"
                    "<div style='display:flex;align-items:center;gap:8px'>"
                    "<span>" + htf_icon + "</span>"
                    "<span style='color:" + htf_color + ";font-family:monospace;font-size:0.72rem;font-weight:700'>" + htf_label + "</span>"
                    "<span style='color:#A1A1A6;font-size:0.78rem;margin-left:4px'>Daily trend: " + htf_trend.upper() + " | RSI " + str(htf_rsi) + " | EMA20 $" + str(htf_ema) + "</span>"
                    "</div>"
                    "<div style='color:#A1A1A6;font-size:0.78rem;margin-top:4px'>" + htf_detail + "</div>"
                    "</div>"
                )
                st.markdown(htf_html, unsafe_allow_html=True)

            # ── Move probability ──────────────────────────────────────────
            tr_color = "#D4AF37" if opt["target_realistic"]=="Likely" else "#F6E27A" if opt["target_realistic"]=="Possible" else "#C1121F"
            atr_txt  = (str(opt["atr_multiples"]) + "x ATR needed") if opt["atr_multiples"] else ""
            move_html = (
                "<div style='background:#1A1A1D;border:1px solid #2A2A2D;border-radius:8px;padding:10px 14px;margin-top:6px'>"
                "<div style='display:flex;justify-content:space-between;align-items:center'>"
                "<span style='color:#A1A1A6;font-family:monospace;font-size:0.72rem'>MOVE REQUIRED</span>"
                "<span style='color:" + tr_color + ";font-weight:700;font-size:0.85rem'>" + opt["target_realistic"].upper() + "</span>"
                "</div>"
                "<div style='margin-top:4px;font-size:0.82rem'>"
                "Price needs to move <b style='color:#F5F5F5'>" + str(opt["move_pct"]) + "%</b>"
                + (" &nbsp;|&nbsp; <span style='color:#A1A1A6'>" + atr_txt + "</span>" if atr_txt else "") +
                "</div>"
                "<div style='color:#A1A1A6;font-size:0.75rem;margin-top:2px'>"
                + ("Likely = &le;2x ATR &nbsp; Possible = 2-4x ATR &nbsp; Ambitious = 4x+ ATR" if opt["atr_multiples"] else "") +
                "</div>"
                "</div>"
            )
            st.markdown(move_html, unsafe_allow_html=True)

            if not opt["delta_ok"]:
                st.markdown(f"<div style='background:#1a150a;border:1px solid #F6E27A;border-radius:6px;padding:8px 12px;margin-top:6px;color:#F6E27A;font-size:0.8rem'>Delta {opt['delta']:.2f} outside 0.35-0.85 ideal range</div>", unsafe_allow_html=True)

            delta_color = "#D4AF37" if opt["delta_ok"] else "#F6E27A"
            st.markdown(f"""
            <div class='trade-box {"" if is_bull else "bear"}'>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.88rem'>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>STRIKE</div><div style='font-size:1.2rem;font-weight:700;color:{dir_color}'>${opt['strike']:.2f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>PAY MAX</div><div style='font-weight:700'>${opt['premium']:.2f}/sh</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>ENTRY</div><div style='font-weight:700'>${opt['entry']:.2f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>DELTA</div><div style='font-weight:700;color:{delta_color}'>{opt['delta']:.2f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>EXIT TARGET</div><div style='font-weight:700;color:#D4AF37'>${opt['target']:.2f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>STOP OUT</div><div style='font-weight:700;color:#C1121F'>${opt['stop']:.2f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>R:R RATIO</div><div style='font-weight:700;color:#D4AF37'>{opt['rr']}x</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>MAX LOSS</div><div style='font-weight:700;color:#C1121F'>${opt['max_loss']:.0f}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>CONTRACTS</div><div style='font-size:1.2rem;font-weight:700;color:{dir_color}'>{opt['contracts']}</div></div>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>PROFIT AT TARGET</div><div style='font-size:1.2rem;font-weight:700;color:#D4AF37'>${opt['profit_at_target']:,.0f}</div></div>
                </div>
                <div style='margin-top:8px;padding-top:8px;border-top:1px solid #2A2A2D;display:flex;justify-content:space-between;align-items:center'>
                    <div><div style='color:#A1A1A6;font-size:0.72rem'>EXPIRES</div><div style='font-weight:700'>{opt['expiration']}</div></div>
                    <div style='text-align:right'><div style='color:#A1A1A6;font-size:0.72rem'>POSITION SIZE</div><div style='font-weight:700'>${opt['position_dollars']:.0f} <span style='color:#A1A1A6;font-size:0.75rem'>({opt['pct_of_account']}% of account)</span></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            _contracts = opt.get("contracts", 1)
            _can_split  = _contracts >= 2
            _take_action_q = (
                "Sell %s contract(s) at $%.2f/sh - lock in 100%% gain, let remaining %s ride with tight stop." % (_contracts // 2, opt['exit_take_half'], _contracts - _contracts // 2)
                if _can_split else
                "Close full position at $%.2f/sh (100%% gain). With 1 contract you exit all at once." % opt['exit_take_half']
            )
            _take_action_s = (
                "Sell %s contract(s) at $%.2f/sh - lock in 100%% gain, let remaining %s run with no stop." % (_contracts // 2, opt['exit_take_half'], _contracts - _contracts // 2)
                if _can_split else
                "Close full position at $%.2f/sh (100%% gain). With 1 contract you exit all at once." % opt['exit_take_half']
            )
            if sig_style == "quick":
                exit_hold = "Close within 20-60 minutes regardless of outcome. Do not hold into close."
                exit_take = _take_action_q
            else:
                exit_take = _take_action_s
                exit_hold = "Never hold through earnings. Never add to a losing position. Never let a winner turn into a loser."

            st.markdown(f"""
            <div class='exit-rules'>
                <div style='color:#D4AF37;font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:6px'>EXIT RULES - DECIDE BEFORE YOU ENTER</div>
                <div style='margin:4px 0'><b>{'Partial exit:' if _can_split else 'Full exit:'}</b> {exit_take}</div>
                <div style='margin:4px 0'><b>Close all</b> if {ticker} closes {'below' if is_bull else 'above'} <b style='color:#C1121F'>${opt['exit_stop_stock']:.2f}</b> - pattern failed, no questions asked.</div>
                <div style='margin:4px 0;color:#A1A1A6;font-size:0.8rem'>{exit_hold}</div>
            </div>
            """, unsafe_allow_html=True)

            # Entry timing check + watch queue - market hours only
            watch_key = f"{ticker}_{sig['direction']}"
            already_watching = watch_key in st.session_state.get("watch_queue", {})
            conf_status = "N/A"

            if True:  # entry check runs in all sessions
                conf_result = check_entry_confirmation(df, sig["direction"])
                conf_status = conf_result["status"]
                if mstatus != "open":
                    conf_status = conf_status + " (extended hrs)"
                if "CONFIRMED" in conf_status:
                    conf_bg = "#1A1500"; conf_border = "#D4AF37"; conf_color = "#D4AF37"; conf_icon = "✅"
                    # ── Fire Telegram + paper trade from signals tab ──────────
                    _sig_key = "signals_fired_%s_%s_%s" % (ticker, sig.get("direction",""), i)
                    if not st.session_state.get(_sig_key):
                        st.session_state[_sig_key] = True
                        _detail = sig.get("detail", {})
                        _signal_r = {
                            "ticker":       ticker,
                            "direction":    sig.get("direction","bullish"),
                            "action":       "CALL" if sig.get("direction")=="bullish" else "PUT",
                            "pattern":      sig.get("pattern_label", sig.get("pattern","Signal")),
                            "style":        sig.get("trade_style", trade_style),
                            "confidence":   sig.get("confidence", 60),
                            "gates_passed": gates_passed,
                            "signals_hit":  _detail.get("signals_hit", 0),
                            "signal_detail":_detail.get("signal_detail",[]),
                            "price":        round(float(df["close"].iloc[-1]), 2),
                            "iv_rank":      iv_rank,
                            "earn_days":    earnings_days,
                            "detail":       _detail,
                            "opt":          opt,
                            "sig":          sig,
                            "exh_confirmed":_detail.get("exhaustion_confirmed", False),
                            "exh_reasons":  _detail.get("exhaustion_reasons", []),
                            "rel_vol":      1.0,
                            "vol_spike":    False,
                            "block_detected": False,
                            "sq_state":     "none",
                            "sq_compression": 0,
                            "market_bias":  "neutral",
                            "sector_bias":  "neutral",
                            "elevate":      elevate,
                            "entry_status": "CONFIRMED",
                        }
                        _is_admin = (
                            st.session_state.get("is_admin", False) or
                            st.session_state.get("user_email", "") == ADMIN_EMAIL
                        )
                        # Only fire Telegram from scan tab where gates/exh are verified
                        # Signals tab just logs history and enters paper trade
                        save_signal_history(_signal_r)
                        if st.session_state.get("paper_auto_enabled", True):
                            paper_enter_trade(_signal_r)
                elif "WAITING" in conf_status:
                    conf_bg = "#1A1A1D"; conf_border = "#F6E27A"; conf_color = "#F6E27A"; conf_icon = "👁"
                else:
                    conf_bg = "#1a0a0a"; conf_border = "#C1121F"; conf_color = "#C1121F"; conf_icon = "⏳"

                candle_html = ""
                for c in conf_result.get("candles", []):
                    if c == "green":   candle_html += "<span style='color:#D4AF37'>&#9650;</span> "
                    elif c == "red":   candle_html += "<span style='color:#C1121F'>&#9660;</span> "
                    else:              candle_html += "<span style='color:#A1A1A6'>&#9644;</span> "

                st.markdown(f"""
                <div style='background:{conf_bg};border:1px solid {conf_border};border-radius:8px;padding:10px 14px;margin-top:8px'>
                    <div style='color:{conf_color};font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:4px'>ENTRY TIMING CHECK</div>
                    <div style='display:flex;align-items:center;gap:10px'>
                        <span style='font-size:1.1rem'>{conf_icon}</span>
                        <span style='font-weight:700;color:{conf_color}'>{conf_status}</span>
                        <span style='color:#A1A1A6;font-size:0.82rem'>Recent candles: {candle_html}</span>
                    </div>
                    <div style='color:#F5F5F5;font-size:0.82rem;margin-top:4px'>{conf_result["message"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Fibonacci confluence display in signals tab
                if _fib_sig.get("confirmed"):
                    _fl = _fib_sig.get("level", "")
                    _fp = _fib_sig.get("level_price", 0)
                    _fh = _fib_sig.get("swing_high", 0)
                    _fw = _fib_sig.get("swing_low", 0)
                    _fc = "#D4AF37" if _fl == "61.8%" else "#F6E27A"
                    st.markdown(
                        "<div style='background:#1A1A1D;border:1px solid %s;border-radius:8px;"
                        "padding:10px 14px;margin-top:8px'>"
                        "<div style='color:#A1A1A6;font-family:monospace;font-size:0.68rem;"
                        "letter-spacing:1px;margin-bottom:4px'>FIBONACCI CONFLUENCE</div>"
                        "<div style='font-size:0.9rem;font-weight:700;color:%s'>🔶 %s Retracement</div>"
                        "<div style='font-size:0.75rem;color:#A1A1A6;margin-top:2px'>"
                        "Level: $%.2f &nbsp;·&nbsp; Range: $%.2f — $%.2f</div>"
                        "</div>" % (_fc, _fc, _fl, _fp, _fw, _fh),
                        unsafe_allow_html=True
                    )

                # Show Watch button prominently for strong setups — no auto-adding
                if elevate and not already_watching and conf_status != "CONFIRMED":
                    st.markdown(f"<div style='background:#1A1A1D;border:1px solid #D4AF37;border-radius:6px;padding:6px 12px;margin-top:4px;color:#D4AF37;font-size:0.8rem'>🚨 {gates_passed}/7 gates — elite setup. Hit Watch to track entry timing.</div>", unsafe_allow_html=True)
                elif gates_passed >= 5 and not already_watching and conf_status != "CONFIRMED":
                    st.markdown(f"<div style='background:#1A1A1D;border:1px solid #F6E27A;border-radius:6px;padding:6px 12px;margin-top:4px;color:#F6E27A;font-size:0.8rem'>⚡ {gates_passed}/7 gates - strong setup. Hit Watch to track entry timing.</div>", unsafe_allow_html=True)


            if True:  # AI brief available in all sessions
                if ANTHROPIC_API_KEY:
                    ai_key = f"ai_result_{ticker}_{key_prefix}_{i}"
                    if st.button(f"🤖 Get AI Brief #{i+1}", key=f"{key_prefix}_ai_{i}"):
                        with st.spinner("Analyzing setup..."):
                            try:
                                ai_text   = get_ai_brief(ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status)
                                ai_parsed = parse_ai_brief(ai_text)
                                st.session_state[ai_key] = ai_parsed
                            except Exception as e:
                                st.session_state[ai_key] = {"error": str(e)}

                    if ai_key in st.session_state:
                        ai = st.session_state[ai_key]
                        if "error" in ai:
                            st.error(f"AI call failed: {ai['error']}")
                        else:
                            rating = ai.get("rating","")
                            if "Strong" in rating:     r_color = "#D4AF37"; r_bg = "#1A1500"; r_border = "#D4AF37"
                            elif "Moderate" in rating: r_color = "#F6E27A"; r_bg = "#1a150a"; r_border = "#F6E27A"
                            else:                      r_color = "#C1121F"; r_bg = "#1a0a0a"; r_border = "#C1121F"
                            st.markdown(f"""
                            <div style='background:{r_bg};border:1px solid {r_border};border-radius:8px;padding:14px;margin-top:8px'>
                                <div style='color:#A1A1A6;font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:6px'>AI TRADE BRIEF</div>
                                <div style='font-size:1.1rem;font-weight:700;color:{r_color};margin-bottom:10px'>🤖 {rating}</div>
                                <div style='margin:6px 0;font-size:0.85rem'><span style='color:#A1A1A6'>REASONING</span><br>{ai.get("reasoning","")}</div>
                                <div style='margin:6px 0;font-size:0.85rem'><span style='color:#C1121F'>KEY RISK</span><br>{ai.get("risk","")}</div>
                                <div style='margin:6px 0;font-size:0.85rem'><span style='color:#D4AF37'>EDGE</span><br>{ai.get("edge","")}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("<div class='ai-placeholder'>🤖 AI Trade Brief - Add ANTHROPIC_API_KEY in Railway to enable</div>", unsafe_allow_html=True)

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                share_text = build_share_text(ticker,sig,opt,gates_passed,7,elevate,mtext)
                st.download_button(f"📤 Share #{i+1}", data=share_text,
                    file_name=f"{ticker}_signal_{datetime.now().strftime('%m%d_%H%M')}.txt",
                    mime="text/plain", key=f"{key_prefix}_share_{i}")
            with bcol2:
                if not already_watching:
                    if st.button(f"👁 Watch #{i+1}", key=f"{key_prefix}_watch_{i}", use_container_width=True):
                        add_to_watch_queue(ticker, sig["direction"], sig, opt)
                        st.success("✅ Added to Watch Queue!")
                        # Don't rerun - keeps signal visible with success message
                else:
                    st.markdown(
                        "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:6px;"
                        "padding:6px 12px;font-size:0.75rem;color:#D4AF37;text-align:center'>"
                        "✅ In Watch Queue</div>",
                        unsafe_allow_html=True
                    )
                    if st.button(f"Remove #{i+1}", key=f"{key_prefix}_unwatch_{i}", use_container_width=True):
                        remove_from_watch_queue(watch_key)
                        st.rerun()

        if i < len(candidates) - 1:
            st.markdown("<hr style='border-color:#2A2A2D;margin:12px 0'>", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION SCAN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@_thread_cache(ttl=300)
def get_market_internals():
    """
    Checks SPY and QQQ trend to determine overall market bias.
    Returns: bias ("bullish"/"bearish"/"neutral"), strength (0-100)
    """
    try:
        results = {}
        for sym in ["SPY","QQQ"]:
            df = _yf_download(sym, period="5d", interval="15m")
            if df is None or df.empty: continue
            df = _clean_df(df)
            close = _col(df, "close")
            ema20 = float(close.ewm(span=20).mean().iloc[-1])
            ema50 = float(close.ewm(span=50).mean().iloc[-1])
            price = float(close.iloc[-1])
            rsi   = calc_rsi(close)
            results[sym] = {
                "above_ema20": price > ema20,
                "above_ema50": price > ema50,
                "rsi": rsi,
                "price": price,
                "ema20": round(ema20,2),
            }
        if not results: return "neutral", 50

        bull_signals = sum([
            results.get("SPY",{}).get("above_ema20", False),
            results.get("SPY",{}).get("above_ema50", False),
            results.get("QQQ",{}).get("above_ema20", False),
            results.get("QQQ",{}).get("above_ema50", False),
            results.get("SPY",{}).get("rsi",50) > 50,
            results.get("QQQ",{}).get("rsi",50) > 50,
        ])
        bear_signals = 6 - bull_signals

        if bull_signals >= 5:   return "bullish", int(bull_signals/6*100)
        elif bear_signals >= 5: return "bearish", int(bear_signals/6*100)
        else:                   return "neutral",  50
    except:
        return "neutral", 50

@_thread_cache(ttl=300)
@_thread_cache(ttl=300)
def get_sector_bias(sector_etf):
    """Returns trend direction of a sector ETF."""
    try:
        df = _yf_download(sector_etf, period="5d", interval="1h")
        if df is None or df.empty: return "neutral"
        df = _clean_df(df)
        close = _col(df, "close")
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        return "bullish" if price > ema20 else "bearish"
    except:
        return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION ENGINE
# Layer 1: Breadth Calculator
# Layer 2: Index Health Check  
# Layer 3: Rally Authenticity
# Layer 4: Regime Classifier
# Layer 5: Signal Adjuster
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_breadth_score(go_now, watching, on_deck):
    """
    Layer 1 — Breadth Calculator
    Analyzes directional bias across all scan results.
    Returns score from -100 (fully bearish) to +100 (fully bullish).
    Weighted by confidence and gate count.
    """
    bull_weight = 0.0
    bear_weight = 0.0
    
    def weight(r):
        conf  = r.get("confidence", 50) / 100
        gates = r.get("gates_passed", 3) / 7
        return conf * gates

    # GO NOW signals count most
    for r in go_now:
        w = weight(r) * 3.0  # 3x multiplier for GO NOW
        if r.get("direction") == "bullish":
            bull_weight += w
        else:
            bear_weight += w

    # WATCHING counts moderately
    for r in watching:
        w = weight(r) * 1.5
        if r.get("direction") == "bullish":
            bull_weight += w
        else:
            bear_weight += w

    # ON DECK counts lightly
    real_on_deck = [r for r in on_deck if not r.get("_rejected")]
    for r in real_on_deck:
        w = weight(r) * 0.5
        if r.get("direction") == "bullish":
            bull_weight += w
        else:
            bear_weight += w

    total = bull_weight + bear_weight
    if total == 0:
        return 0, 0, 0  # no signals

    bull_pct = bull_weight / total * 100
    bear_pct = bear_weight / total * 100
    score    = round(bull_pct - bear_pct)  # -100 to +100

    return score, round(bull_pct), round(bear_pct)


def check_index_health(ticker="SPY"):
    """
    Layer 2 — Index Health Check
    Analyzes SPY/QQQ/IWM for trend direction, volume, and momentum.
    Returns dict with health metrics.
    """
    try:
        # Get daily data for trend analysis
        df_daily = _fmp_download(ticker, "60d", "1d")
        if df_daily is None or len(df_daily) < 20:
            return {"status": "unknown", "trend_5d": "neutral", "trend_20d": "neutral",
                    "vol_ratio": 1.0, "rsi": 50, "above_20ema": None}

        close  = df_daily["close"].astype(float)
        volume = df_daily["volume"].astype(float)

        # 5-day vs 20-day trend
        ema5  = float(close.ewm(span=5).mean().iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])

        trend_5d  = "bullish" if price > ema5  else "bearish"
        trend_20d = "bullish" if price > ema20 else "bearish"

        # Volume — up day vol vs down day vol ratio (last 10 sessions)
        recent = df_daily.iloc[-10:].copy()
        recent["up"] = recent["close"] > recent["close"].shift(1)
        up_vol   = float(recent[recent["up"] == True]["volume"].mean()) if len(recent[recent["up"] == True]) > 0 else 1
        down_vol = float(recent[recent["up"] == False]["volume"].mean()) if len(recent[recent["up"] == False]) > 0 else 1
        vol_ratio = round(up_vol / down_vol, 2) if down_vol > 0 else 1.0

        # RSI 14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 0.001)
        rsi   = round(float(100 - (100 / (1 + rs.iloc[-1]))), 1)

        # Distance from 52-week high
        high_52w = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
        pct_from_high = round((price - high_52w) / high_52w * 100, 1)

        # SPY IV rank as fear proxy
        iv_rank = None
        try:
            iv_rank = get_iv_rank(ticker)
        except Exception:
            pass

        # Overall health status
        if trend_5d == "bullish" and trend_20d == "bullish" and vol_ratio > 1.1:
            status = "healthy"
        elif trend_5d == "bearish" and trend_20d == "bearish":
            status = "weak"
        elif trend_5d != trend_20d:
            status = "transitioning"
        else:
            status = "neutral"

        return {
            "status":         status,
            "trend_5d":       trend_5d,
            "trend_20d":      trend_20d,
            "vol_ratio":      vol_ratio,
            "rsi":            rsi,
            "pct_from_high":  pct_from_high,
            "above_20ema":    price > ema20,
            "iv_rank":        iv_rank,
            "price":          round(price, 2),
        }
    except Exception as e:
        return {"status": "unknown", "trend_5d": "neutral", "trend_20d": "neutral",
                "vol_ratio": 1.0, "rsi": 50, "above_20ema": None}


def check_rally_authenticity(ticker="SPY"):
    """
    Layer 3 — Rally Authenticity Score
    Detects false rallies (bull traps) by comparing:
    - Current move volume vs prior selloff volume
    - Fibonacci resistance proximity
    - RSI divergence on the bounce
    Returns: AUTHENTIC, SUSPECT, or FALSE
    """
    try:
        df = _fmp_download(ticker, "14d", "1d")
        if df is None or len(df) < 5:
            return "unknown", {}

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)
        price  = float(close.iloc[-1])

        # Find the recent selloff — biggest down day in last 5 sessions
        recent = df.iloc[-5:].copy()
        recent["pct_chg"] = recent["close"].pct_change() * 100
        worst_day_idx = recent["pct_chg"].idxmin()
        selloff_vol   = float(recent.loc[worst_day_idx, "volume"]) if worst_day_idx is not None else 0
        selloff_pct   = float(recent.loc[worst_day_idx, "pct_chg"]) if worst_day_idx is not None else 0

        # Current bounce volume vs selloff volume
        bounce_vol = float(volume.iloc[-1])
        vol_ratio  = round(bounce_vol / selloff_vol, 2) if selloff_vol > 0 else 1.0

        # Is price going up today?
        today_chg  = float(close.pct_change().iloc[-1] * 100)
        is_bouncing = today_chg > 0.3

        # Fibonacci check — is price at resistance?
        fib_result = detect_fibonacci_confluence(df, "bearish", price)
        at_fib_resistance = fib_result.get("confirmed", False) and fib_result.get("level") in ["38.2%", "50.0%", "61.8%"]

        # RSI divergence — price higher but RSI lower
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 0.001)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_now    = float(rsi_series.iloc[-1])
        rsi_prev   = float(rsi_series.iloc[-3])
        price_prev = float(close.iloc[-3])
        rsi_diverging = (price > price_prev) and (rsi_now < rsi_prev)

        # Score the authenticity
        fake_signals = 0
        if is_bouncing and vol_ratio < 0.7:  fake_signals += 2  # low volume bounce
        if at_fib_resistance:                 fake_signals += 2  # hitting resistance
        if rsi_diverging:                     fake_signals += 1  # RSI not confirming
        if selloff_pct < -2.0 and today_chg > 1.0:  fake_signals += 1  # gap up after big drop

        if fake_signals >= 4:
            authenticity = "FALSE"
        elif fake_signals >= 2:
            authenticity = "SUSPECT"
        else:
            authenticity = "AUTHENTIC"

        return authenticity, {
            "vol_ratio":       vol_ratio,
            "at_fib_res":      at_fib_resistance,
            "fib_level":       fib_result.get("level"),
            "rsi_diverging":   rsi_diverging,
            "rsi":             round(rsi_now, 1),
            "selloff_pct":     round(selloff_pct, 1),
            "bounce_pct":      round(today_chg, 1),
            "fake_signals":    fake_signals,
        }
    except Exception:
        return "unknown", {}


def classify_market_regime(breadth_score, index_health, rally_auth, go_now, watching):
    """
    Layer 4 — Regime Classifier
    Combines breadth, index health, and rally authenticity
    into a single regime label.
    """
    trend_20d    = index_health.get("trend_20d", "neutral")
    trend_5d     = index_health.get("trend_5d", "neutral")
    vol_ratio    = index_health.get("vol_ratio", 1.0)
    rsi          = index_health.get("rsi", 50)
    iv_rank      = index_health.get("iv_rank")
    index_status = index_health.get("status", "neutral")

    # Determine regime
    if rally_auth == "FALSE" and breadth_score < 0:
        regime = "BULL TRAP"
        desc   = "Rally is suspect. Volume weak, breadth bearish. Watch for reversal."
        color  = "#C1121F"
        bias   = "bearish"

    elif trend_20d == "bearish" and breadth_score < -30:
        regime = "BEAR CONFIRMED"
        desc   = "Sustained downtrend with broad participation. PUT signals elevated."
        color  = "#C1121F"
        bias   = "bearish"

    elif trend_20d == "bullish" and breadth_score > 30 and vol_ratio > 1.0:
        regime = "BULL CONFIRMED"
        desc   = "Broad participation, healthy volume. CALL signals elevated."
        color  = "#22C55E"
        bias   = "bullish"

    elif trend_5d == "bearish" and trend_20d == "bullish" and breadth_score < -10:
        regime = "DISTRIBUTION"
        desc   = "Index healthy long-term but short-term weakness. Smart money may be selling."
        color  = "#F6E27A"
        bias   = "bearish"

    elif trend_5d == "bullish" and trend_20d == "bearish" and breadth_score > 10:
        regime = "BEAR TRAP"
        desc   = "Short-term bounce in downtrend. Oversold relief rally likely."
        color  = "#F6E27A"
        bias   = "neutral"

    elif iv_rank is not None and iv_rank > 60 and rsi < 35:
        regime = "CAPITULATION"
        desc   = "Extreme fear. Oversold conditions. Potential reversal zone."
        color  = "#D4AF37"
        bias   = "neutral"

    elif rally_auth == "SUSPECT":
        regime = "SUSPECT RALLY"
        desc   = "Rally showing weakness signals. Proceed with extra caution."
        color  = "#F6E27A"
        bias   = "neutral"

    elif abs(breadth_score) < 20:
        regime = "CHOPPY"
        desc   = "No clear directional edge. Reduce size. Wait for clarity."
        color  = "#A1A1A6"
        bias   = "neutral"

    else:
        regime = "NEUTRAL"
        desc   = "Mixed signals. Standard signal criteria apply."
        color  = "#A1A1A6"
        bias   = "neutral"

    return {
        "regime": regime,
        "desc":   desc,
        "color":  color,
        "bias":   bias,
        "breadth_score": breadth_score,
    }


def apply_regime_adjustments(signals, regime_data):
    """
    Layer 5 — Signal Adjuster
    Applies regime context to each signal.
    Boosts aligned signals, flags counter-regime signals.
    """
    regime = regime_data.get("regime", "NEUTRAL")
    bias   = regime_data.get("bias", "neutral")
    adjusted = []

    for r in signals:
        r = dict(r)  # don't mutate original
        direction = r.get("direction", "bullish")
        conf      = r.get("confidence", 50)

        # Determine alignment
        if bias == "neutral":
            alignment = "NEUTRAL"
            conf_adj  = 0
        elif (bias == "bullish" and direction == "bullish") or              (bias == "bearish" and direction == "bearish"):
            alignment = "CONFIRMED"
            conf_adj  = +5  # boost aligned signals
        else:
            alignment = "COUNTER"
            conf_adj  = -10  # penalize counter-regime signals

        # Special rules per regime
        if regime == "BULL TRAP" and direction == "bullish":
            alignment = "BLOCKED"
            conf_adj  = -20  # heavily penalize calls in bull trap

        if regime == "CAPITULATION" and direction == "bullish":
            conf_adj  = +8  # bounce plays in capitulation

        if regime == "CHOPPY":
            conf_adj  = -5  # reduce confidence in choppy market

        r["regime_alignment"] = alignment
        r["confidence"]       = min(97, max(30, conf + conf_adj))
        adjusted.append(r)

    return adjusted


def detect_fibonacci_confluence(df, direction, current_price=None):
    """
    Detects if current price is at a key Fibonacci retracement level.
    
    Uses the dominant swing high and low from the dataframe.
    Returns confluence data including which level, confidence boost, and details.
    
    Levels checked: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Tolerance: 0.5% of price range to be considered "at" a level
    """
    if df is None or len(df) < 20:
        return {"confirmed": False, "level": None, "boost": 0, "detail": "Insufficient data"}

    try:
        close = df["close"] if "close" in df.columns else df.iloc[:, 4]
        high  = df["high"]  if "high"  in df.columns else df.iloc[:, 2]
        low   = df["low"]   if "low"   in df.columns else df.iloc[:, 3]

        price = current_price if current_price else float(close.iloc[-1])

        # Find dominant swing high and low
        # Use last 50 candles for intraday, all data for daily
        lookback = min(len(df), 50)
        _high_series = high.iloc[-lookback:]
        _low_series  = low.iloc[-lookback:]

        swing_high = float(_high_series.max())
        swing_low  = float(_low_series.min())
        price_range = swing_high - swing_low

        if price_range < 0.01:
            return {"confirmed": False, "level": None, "boost": 0, "detail": "Range too tight"}

        # Calculate Fib levels
        # For bullish (price coming from low, retracing up): levels measured from low
        # For bearish (price coming from high, retracing down): levels measured from high
        fib_levels = {
            "23.6%": swing_high - (price_range * 0.236),
            "38.2%": swing_high - (price_range * 0.382),
            "50.0%": swing_high - (price_range * 0.500),
            "61.8%": swing_high - (price_range * 0.618),
            "78.6%": swing_high - (price_range * 0.786),
        }

        # Confidence boosts per level
        boosts = {
            "23.6%": 3,
            "38.2%": 5,
            "50.0%": 8,
            "61.8%": 15,  # golden ratio — highest boost
            "78.6%": 6,
        }

        # Tolerance — 0.5% of current price
        tolerance = price * 0.005

        best_level = None
        best_boost = 0
        best_distance = float("inf")

        for level_name, level_price in fib_levels.items():
            distance = abs(price - level_price)
            if distance <= tolerance and distance < best_distance:
                best_level    = level_name
                best_boost    = boosts[level_name]
                best_distance = distance
                best_price    = level_price

        if best_level:
            # Check if this level aligns with prior S/R (tested before)
            # Count how many times price has been near this level in history
            near_count = sum(
                1 for p in close.iloc[:-5]  # exclude last 5 candles
                if abs(float(p) - best_price) <= tolerance * 2
            )
            touches = min(near_count, 3)  # cap at 3

            # Extra boost for multiple touches — proven level
            touch_boost = touches * 2

            total_boost = best_boost + touch_boost

            detail = "🔶 %s Fib retracement ($%.2f)" % (best_level, best_price)
            if touches >= 2:
                detail += " — %sx tested level" % touches

            return {
                "confirmed":  True,
                "level":      best_level,
                "level_price": round(best_price, 2),
                "boost":      total_boost,
                "touches":    touches,
                "detail":     detail,
                "swing_high": round(swing_high, 2),
                "swing_low":  round(swing_low, 2),
            }

        # Not at a key level — check if between levels (no man's land)
        # Find closest level for informational purposes
        closest = min(fib_levels.items(), key=lambda x: abs(price - x[1]))
        pct_away = abs(price - closest[1]) / price * 100

        return {
            "confirmed":  False,
            "level":      None,
            "boost":      0,
            "detail":     "Price %.1f%% from nearest Fib (%s at $%.2f)" % (pct_away, closest[0], closest[1]),
            "swing_high": round(swing_high, 2),
            "swing_low":  round(swing_low, 2),
        }

    except Exception as e:
        return {"confirmed": False, "level": None, "boost": 0, "detail": "Fib error: %s" % str(e)[:40]}


def detect_exhaustion(df, direction):
    """
    Elite exhaustion detection - requires 2 of 4 signals minimum.
    Volume MUST be expanding (>1.2x avg) to count as exhaustion candle.
    Confirmed = 2+ signals present (raised from 1).
    """
    if len(df) < 20:
        return False, 0, ["Insufficient data"]

    close   = df["close"].astype(float)
    high    = df["high"].astype(float)
    low     = df["low"].astype(float)
    open_   = df["open"].astype(float)
    volume  = df["volume"].astype(float)
    avg_vol = float(volume.iloc[-20:].mean())
    is_bull = direction == "bullish"
    reasons = []
    score   = 0

    # 1. Exhaustion candle - big body + MUST have above-average volume
    exh_found = False
    for j in range(-6, 0):
        body      = float(open_.iloc[j]) - float(close.iloc[j]) if is_bull else float(close.iloc[j]) - float(open_.iloc[j])
        rng       = float(high.iloc[j]) - float(low.iloc[j])
        is_big    = rng > 0 and body / rng > 0.55
        vol_ratio = float(volume.iloc[j]) / avg_vol if avg_vol > 0 else 0
        is_vol    = vol_ratio >= 1.2
        if body > 0 and is_big and is_vol:
            score += 1
            reasons.append("%s candle confirmed (%.1fx vol)" % ("Capitulation" if is_bull else "Climax", vol_ratio))
            exh_found = True
            break
    if not exh_found:
        reasons.append("No exhaustion candle with volume confirmation")

    # 2. Reversal candle
    last_body  = abs(float(close.iloc[-1]) - float(open_.iloc[-1]))
    last_range = float(high.iloc[-1]) - float(low.iloc[-1])
    is_doji    = last_range > 0 and last_body / last_range < 0.3
    if is_bull:
        lower_wick = min(float(open_.iloc[-1]), float(close.iloc[-1])) - float(low.iloc[-1])
        is_hammer  = last_range > 0 and lower_wick / last_range > 0.45
        if is_hammer or is_doji:
            score += 1
            reasons.append("Hammer/doji reversal candle")
        else:
            reasons.append("No reversal candle yet")
    else:
        upper_wick = float(high.iloc[-1]) - max(float(open_.iloc[-1]), float(close.iloc[-1]))
        is_star    = last_range > 0 and upper_wick / last_range > 0.45
        if is_star or is_doji:
            score += 1
            reasons.append("Shooting star/doji reversal candle")
        else:
            reasons.append("No reversal candle yet")

    # 3. RSI divergence
    div = detect_rsi_divergence(df)
    if div and ((is_bull and div.get("type") == "bullish") or
                (not is_bull and div.get("type") == "bearish")):
        score += 1
        reasons.append("RSI divergence confirmed")
    else:
        reasons.append("No RSI divergence")

    # 4. Structure
    if is_bull:
        lows = [float(low.iloc[i]) for i in [-15, -8, -1]]
        if lows[-1] > lows[-2]:
            score += 1
            reasons.append("Higher low structure forming")
        else:
            reasons.append("Lower low - structure not confirmed")
    else:
        highs = [float(high.iloc[i]) for i in [-15, -8, -1]]
        if highs[-1] < highs[-2]:
            score += 1
            reasons.append("Lower high structure forming")
        else:
            reasons.append("Higher high - structure not confirmed")

    confirmed = score >= 2
    return confirmed, score, reasons


def precision_score(ticker, direction, df_primary, df_confirm,
                    iv_rank, earnings_days, market_bias,
                    sector_bias, atr, dte, account_size, risk_pct,
                    trade_style, current_price=None):
    """
    Elite scoring framework v6.1
    TIER 1: Hard stops. TIER 2: 4/5 quality signals. TIER 3: scoring.
    """
    import pytz
    from datetime import datetime as _dt

    # ── TIER 1: Hard stops ────────────────────────────────────────────────────
    if earnings_days is not None and earnings_days <= 5:
        return None, "Earnings within 5 days"

    if iv_rank is not None and iv_rank > 70:
        return None, "IV too high (%s%%)" % iv_rank

    # Block zero volume — market closed or no liquidity
    if df_primary is not None and len(df_primary) > 0:
        try:
            _cur_vol = float(df_primary["volume"].iloc[-3:].mean()) if "volume" in df_primary.columns else 1
            if _cur_vol == 0:
                return None, "Zero volume — market closed or no liquidity"
        except Exception:
            pass
    # Signals going against market bias get flagged on the card instead
    _against_bias = (
        (market_bias == "bullish" and direction == "bearish") or
        (market_bias == "bearish" and direction == "bullish")
    )
    # _against_bias is used below to apply confidence penalty instead of hard block

    try:
        import math as _m
        _cp = float(current_price) if current_price is not None else 999
        if not _m.isnan(_cp) and _cp < 15:
            return None, "Stock under $15 - options liquidity too thin (%.2f)" % _cp
    except Exception:
        pass

    # Gap check - stock already moved hard against signal
    try:
        _gap_df = _yf_download(ticker, period="2d", interval="1d")
        if _gap_df is not None and len(_gap_df) >= 2:
            _gap_df  = _clean_df(_gap_df.reset_index())
            _closes  = _col(_gap_df, "close")
            prev_close = float(_closes.iloc[-2])
            curr_price = float(current_price) if current_price else float(_closes.iloc[-1])
            if prev_close > 0 and curr_price > 0:
                gap_pct = (curr_price - prev_close) / prev_close * 100
                if direction == "bearish" and gap_pct > 3.0:
                    return None, "Gapped up %.1f%% against PUT signal - invalidated" % gap_pct
                if direction == "bullish" and gap_pct < -3.0:
                    return None, "Gapped down %.1f%% against CALL signal - invalidated" % gap_pct
    except Exception:
        pass

    # Sector momentum hard stop
    try:
        sector_etf = SECTOR_ETF.get(ticker, "SPY")
        if sector_etf and sector_etf != ticker:
            _sec_df = _yf_download(sector_etf, period="2d", interval="1d")
            if _sec_df is not None and len(_sec_df) >= 2:
                _sec_df  = _clean_df(_sec_df.reset_index())
                _sc      = _col(_sec_df, "close")
                sec_prev = float(_sc.iloc[-2])
                sec_curr = float(_sc.iloc[-1])
                if sec_prev > 0 and sec_curr > 0:
                    sec_move = (sec_curr - sec_prev) / sec_prev * 100
                    if direction == "bearish" and sec_move > 2.0:
                        return None, "Sector %s up %.1f%% - fighting PUT signal" % (sector_etf, sec_move)
                    if direction == "bullish" and sec_move < -2.0:
                        return None, "Sector %s down %.1f%% - fighting CALL signal" % (sector_etf, sec_move)
    except Exception:
        pass

    # Signal invalidation - price already moved 1.5x ATR against signal
    try:
        if atr and atr > 0 and current_price and df_primary is not None and len(df_primary) >= 10:
            entry_price = float(df_primary["close"].iloc[-1])
            if direction == "bullish":
                recent_high = float(df_primary["high"].iloc[-10:].max())
                drop = recent_high - entry_price
                if drop > atr * 1.5:
                    return None, "Price dropped %.2f (%.1fx ATR) from recent high - CALL invalidated" % (drop, drop/atr)
            else:
                recent_low = float(df_primary["low"].iloc[-10:].min())
                rip = entry_price - recent_low
                if rip > atr * 1.5:
                    return None, "Price ripped %.2f (%.1fx ATR) from recent low - PUT invalidated" % (rip, rip/atr)
    except Exception:
        pass

    # ── TIER 2: Quality signals - need 4 of 5 ────────────────────────────────
    signals_hit   = 0
    signal_detail = []

    # Signal 1: Trend aligned
    try:
        trend_dir, trend_score, _, _, _, _ = get_trend(df_primary)
        if trend_dir == direction:
            signals_hit += 1
            signal_detail.append("✅ Trend Confirmed")
        else:
            signal_detail.append("❌ Trend Opposing")
    except Exception:
        signal_detail.append("❌ Trend Unavailable")

    # Signal 2: Volume confirming - 1.2x minimum
    try:
        avg_vol   = float(df_primary["volume"].iloc[-20:].mean())
        cur_vol   = float(df_primary["volume"].iloc[-3:].mean())
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio >= 1.2:
            signals_hit += 1
            signal_detail.append("✅ Volume Confirmed")
        else:
            signal_detail.append("❌ Volume Insufficient")
    except Exception:
        signal_detail.append("❌ Volume Unavailable")

    # Signal 3: Exhaustion - 2/4 minimum
    exh_confirmed, exh_score, exh_reasons = detect_exhaustion(df_primary, direction)
    if exh_confirmed:
        signals_hit += 1
        signal_detail.append("✅ Exhaustion Confirmed")
    else:
        signal_detail.append("❌ Exhaustion Not Confirmed")

    # Signal 4: Squeeze
    try:
        sq_state, sq_compression = detect_squeeze(df_primary, direction)
        if sq_state == "firing" and sq_compression >= 40:
            signals_hit += 1
            signal_detail.append("✅ Momentum Breakout Confirmed")
        elif sq_state == "squeeze" and sq_compression >= 40:
            signals_hit += 1
            signal_detail.append("✅ Momentum Building")
        elif sq_state == "firing" and sq_compression < 40:
            signal_detail.append("❌ Momentum Insufficient")
        else:
            signal_detail.append("❌ No Momentum Signal")
    except Exception:
        signal_detail.append("❌ Momentum Data Unavailable")

    # Signal 5: RSI divergence
    try:
        div = detect_rsi_divergence(df_primary)
        if div and ((direction == "bullish" and div.get("type") == "bullish") or
                    (direction == "bearish" and div.get("type") == "bearish")):
            signals_hit += 1
            signal_detail.append("✅ Price Divergence Confirmed")
        else:
            signal_detail.append("❌ No Price Divergence")
    except Exception:
        signal_detail.append("❌ Divergence Data Unavailable")

    # Signal 6: Fibonacci confluence
    try:
        current_px = float(df_primary["close"].iloc[-1]) if "close" in df_primary.columns else None
        fib_result = detect_fibonacci_confluence(df_primary, direction, current_px)
        if fib_result.get("confirmed"):
            signals_hit += 1
            signal_detail.append("✅ Fibonacci Confluence Confirmed")
        else:
            signal_detail.append("❌ No Fibonacci Confluence")
    except Exception:
        signal_detail.append("❌ Fibonacci Data Unavailable")
        fib_result = {"confirmed": False, "boost": 0}

    # 4/5 required for GO NOW - but still score 3/5 for WATCHING/ON DECK
    # Bucket assignment in full_scan uses signals_hit to separate tiers
    if signals_hit < 2:
        return None, "Only %s/5 quality signals aligned (need 2+ minimum)" % signals_hit
    # 2/5 signals: low confidence, will land in ON DECK only

    # ── TIER 3: Execution quality scoring ────────────────────────────────────
    score = 50
    score += signals_hit * 5
    score += min(exh_score * 3, 10)

    # Fibonacci confluence boost — up to +8 max (reduced to prevent score inflation)
    fib_boost = fib_result.get("boost", 0) if isinstance(fib_result, dict) else 0
    score += min(fib_boost, 8)

    tf_agree = 0
    tf_total = 0
    if df_confirm is not None:
        tf_total = 1
        try:
            if isinstance(df_confirm.columns, pd.MultiIndex):
                c = df_confirm["close"].iloc[:, 0].astype(float)
            else:
                c = df_confirm["close"].astype(float)
            em     = c.ewm(span=20).mean()
            em_val = float(em.iloc[-1].item() if hasattr(em.iloc[-1], 'item') else em.iloc[-1])
            pr_val = float(c.iloc[-1].item()  if hasattr(c.iloc[-1],  'item') else c.iloc[-1])
            if (pr_val > em_val and direction == "bullish") or                (pr_val < em_val and direction == "bearish"):
                tf_agree = 1
                score   += 10
        except Exception:
            pass

    if market_bias == direction:    score += 6
    elif market_bias == "neutral":  score += 3
    elif _against_bias:             score -= 8   # penalty not block - still tradeable
    if sector_bias  == direction:   score += 4
    elif sector_bias == "neutral":  score += 2

    if iv_rank is not None:
        if 20 <= iv_rank <= 50:   score += 5
        elif 15 <= iv_rank <= 65: score += 2

    liq_ok, liq_vol, liq_oi, _ = check_liquidity(ticker)
    if liq_ok:
        score += 3 if liq_vol >= 500 else 2 if liq_vol >= 100 else 1

    final = min(97, max(50, score))  # cap at 97 — reserve 100% for exceptional setups only

    return final, {
        "exhaustion_confirmed": exh_confirmed,
        "exhaustion_score":     exh_score,
        "exhaustion_reasons":   exh_reasons,
        "signals_hit":          signals_hit,
        "signal_detail":        signal_detail,
        "tf_agree":             tf_agree,
        "tf_total":             tf_total,
        "market_bias":          market_bias,
        "sector_bias":          sector_bias,
        "against_market_bias":  _against_bias,
        "liq_ok":               liq_ok,
        "liq_vol":              liq_vol,
        "fib_confirmed":        fib_result.get("confirmed", False) if isinstance(fib_result, dict) else False,
        "fib_level":            fib_result.get("level") if isinstance(fib_result, dict) else None,
        "fib_level_price":      fib_result.get("level_price") if isinstance(fib_result, dict) else None,
        "fib_detail":           fib_result.get("detail", "") if isinstance(fib_result, dict) else "",
        "fib_swing_high":       fib_result.get("swing_high") if isinstance(fib_result, dict) else None,
        "fib_swing_low":        fib_result.get("swing_low") if isinstance(fib_result, dict) else None,
    }


def scan_single_ticker(ticker, toggles, account_size, risk_pct,
                        dte_quick, dte_swing, max_premium,
                        trade_style_filter, market_bias):
    """
    Processes one ticker through the full precision stack.
    Designed to run in a thread pool.
    Returns list of result records (may be empty).
    """
    results = []
    _reject_reason = "unknown"
    try:
        tfs_q = fetch_multi_tf(ticker, "quick")
        tfs_s = fetch_multi_tf(ticker, "swing")

        _15m = tfs_q.get("15min"); _5m = tfs_q.get("5min")
        _1h  = tfs_s.get("1hr");  _4h = tfs_s.get("4hr"); _1d = tfs_s.get("daily")

        primary_q = _15m if _15m is not None else _5m
        primary_s = _1h

        iv_rank, _ = fetch_iv_rank(ticker)
        earn_days  = check_earnings(ticker)
        price      = fetch_current_price(ticker)
        sector_etf = SECTOR_ETF.get(ticker, "SPY")
        sec_bias   = get_sector_bias(sector_etf)
        atr        = calc_atr(_1d) if _1d is not None else None

        styles = []
        if trade_style_filter in ("quick","both") and primary_q is not None:
            styles.append(("quick", primary_q, _5m, dte_quick))
        if trade_style_filter in ("swing","both") and primary_s is not None:
            styles.append(("swing", primary_s, _4h if _4h is not None else _1d, dte_swing))

        if not styles:
            results.append({"ticker": ticker, "_rejected": True,
                "_reason": "no styles available (primary_q=%s primary_s=%s)" % (primary_q is not None, primary_s is not None)})

        for style, df_pri, df_con, dte in styles:
            if df_pri is None or len(df_pri) < 20:
                results.append({"ticker": ticker, "_rejected": True,
                    "_reason": "[%s] df_pri too short or None (len=%s)" % (style, len(df_pri) if df_pri is not None else 0)})
                continue
            _rp = df_pri["close"].iloc[-1]
            cur_price = price if price is not None else float(_rp.iloc[0] if hasattr(_rp,"iloc") else _rp)

            cands = build_candidates(df_pri, ticker, toggles,
                                     account_size, risk_pct, dte,
                                     trade_style=style, atr=atr)
            if not cands:
                results.append({"ticker": ticker, "_rejected": True,
                    "_reason": "[%s] no pattern candidates found" % style})
                continue

            best      = cands[0]
            direction = best["direction"]

            opt = calc_trade(best["entry"], best["stop"], best["target"],
                              direction, dte, account_size, risk_pct,
                              cur_price, atr=atr, trade_style=style, ticker=ticker)
            if opt["premium"] > max_premium:
                results.append({"ticker": ticker, "_rejected": True,
                    "_reason": "[%s %s] premium $%.2f > max $%.2f" % (
                        style, "CALL" if direction=="bullish" else "PUT",
                        opt["premium"], max_premium)})
                continue

            # Low R:R - tag it but don't hard reject, let it fall to ON DECK
            _low_rr = opt.get("rr_option", 0) < 2.0

            conf, detail = precision_score(
                ticker, direction, df_pri, df_con,
                iv_rank, earn_days, market_bias,
                sec_bias, atr, dte, account_size, risk_pct, style,
                current_price=cur_price
            )
            if conf is None:
                results.append({"ticker": ticker, "_rejected": True, "_reason": "[%s %s] precision_score: %s" % (style.upper(), "CALL" if direction=="bullish" else "PUT", str(detail)[:80])})
                continue
            if conf < 45:
                results.append({"ticker": ticker, "_rejected": True, "_reason": "[%s %s] conf too low: %s" % (style.upper(), "CALL" if direction=="bullish" else "PUT", conf)})
                continue

            gates, gates_passed, elevate = run_seven_point_gate(
                df_pri, best, opt, iv_rank, earn_days, opt["actual_dte"]
            )
            conf_result  = check_entry_confirmation(df_pri, direction)
            entry_status = conf_result["status"]

            # Relative volume spike (institutional signal proxy)
            avg_vol  = float(df_pri["volume"].iloc[-20:].mean()) if len(df_pri) >= 20 else 1
            cur_vol  = float(df_pri["volume"].iloc[-1])
            rel_vol  = round(cur_vol / avg_vol, 1) if avg_vol > 0 else 1.0
            vol_spike = rel_vol >= 1.5  # 1.5x+ average = notable

            # Block trade proxy: large single candles with >2x volume
            block_detected = rel_vol >= 2.5 and abs(
                float(df_pri["close"].iloc[-1]) - float(df_pri["open"].iloc[-1])
            ) > float(df_pri["close"].iloc[-1]) * 0.003

            # Squeeze state
            try:
                sq_state, sq_compression = detect_squeeze(df_pri, direction)
            except Exception:
                sq_state, sq_compression = "none", 0

            results.append({
                "ticker":        ticker,
                "style":         style,
                "direction":     direction,
                "action":        "CALL" if direction=="bullish" else "PUT",
                "pattern":       best["pattern_label"],
                "confidence":    conf,
                "gates_passed":  gates_passed,
                "low_rr":        _low_rr,
                "elevate":       elevate,
                "entry_status":  entry_status,
                "opt":           opt,
                "sig":           best,
                "price":         round(cur_price, 2),
                "iv_rank":       iv_rank,
                "earn_days":     earn_days,
                "detail":        detail,
                "market_bias":   market_bias,
                "sector_bias":   sec_bias,
                "exh_confirmed": detail.get("exhaustion_confirmed", False),
                "exh_reasons":   detail.get("exhaustion_reasons", []),
                "signal_detail": detail.get("signal_detail", []),
                "signals_hit":   detail.get("signals_hit", 0),
                "rel_vol":        rel_vol,
                "vol_spike":      vol_spike,
                "block_detected": block_detected,
                "sq_state":       sq_state,
                "sq_compression": sq_compression,
            })
    except Exception as _e:
        results.append({"ticker": ticker, "_rejected": True, "_reason": "Exception: " + str(_e)[:80]})
    return results

def full_scan(scan_list, toggles, account_size, risk_pct,
              dte_quick, dte_swing, max_premium, trade_style_filter,
              progress_cb=None):
    """
    Parallel scanner using ThreadPoolExecutor.
    Runs 10 tickers simultaneously - ~10x faster than sequential.
    """
    market_bias, _ = get_market_internals()
    go_now   = []
    watching = []
    on_deck  = []

    completed = 0
    total     = len(scan_list)

    def _process_records(records, ticker):
        for r in records:
            if r.get("_rejected"):
                on_deck.append(r)
                continue
            conf         = r.get("confidence", 0)
            gates_passed = r.get("gates_passed", 0)
            entry_status = r.get("entry_status", "")
            exh_ok       = r.get("exh_confirmed", False)
            signals_hit  = r.get("signals_hit", r.get("detail", {}).get("signals_hit", 0))

            low_rr = r.get("low_rr", False)

            # ── RR floor: quick=1.5x (30%/20%), swing=2.5x (50%/20%)
            _rr_val = r.get("opt", {}).get("rr_option", 0) or 0
            _min_rr = 1.5 if r.get("style") == "quick" else 2.5
            if _rr_val < _min_rr:
                r["_on_deck_reason"] = "Low RR (%.1fx) - need %.1f minimum" % (_rr_val, _min_rr)
                on_deck.append(r)
                continue

            # ── GO NOW - compensating controls
            # Tier 1: conf>=85%, gates>=4, signals>=2
            # Tier 2: conf>=75%, gates>=5, signals>=3
            # Both: CONFIRMED + exhaustion
            # Block GO NOW if volume is zero — market closed or no liquidity
            _vol_detail = r.get("detail", {}) or {}
            _has_volume = "No Activity" not in str(_vol_detail.get("signal_detail", []))
            _high_conf = conf >= 85 and gates_passed >= 4 and signals_hit >= 2
            _med_conf  = conf >= 75 and gates_passed >= 5 and signals_hit >= 3
            _go_now_ok = (_high_conf or _med_conf) and entry_status == "CONFIRMED" and exh_ok

            if _go_now_ok:
                go_now.append(r)

            # ── WATCHING
            elif conf >= 65 and gates_passed >= 4 and entry_status == "CONFIRMED" and signals_hit >= 2:
                watching.append(r)

            elif conf >= 60 and gates_passed >= 3 and signals_hit >= 2:
                watching.append(r)

            # ── ON DECK
            elif conf >= 45 or signals_hit >= 2:
                r["_on_deck_reason"] = (
                    "%s/5 signals" % signals_hit if signals_hit < 3
                    else "Building (%s%%)" % conf
                )
                on_deck.append(r)
    # Submit all futures first
    with ThreadPoolExecutor(max_workers=4) as executor:  # 4 workers — FMP Premium handles 750 calls/min
        futures = {
            executor.submit(
                scan_single_ticker,
                ticker, toggles, account_size, risk_pct,
                dte_quick, dte_swing, max_premium,
                trade_style_filter, market_bias
            ): ticker
            for ticker in scan_list
        }

        # Collect results - 12s per ticker max, 3 min total hard cap
        # as_completed(timeout=X) raises TimeoutError if ANY future takes too long
        # We catch it per-future so one hung ticker never freezes the whole scan
        done_tickers = set()
        _per_ticker_timeout = 12
        _global_deadline = datetime.now().timestamp() + 180  # 3 min hard cap

        # Fire progress - wrapped in try/except so TimeoutError never crashes the app
        try:
            for future in as_completed(futures, timeout=180):
                if datetime.now().timestamp() > _global_deadline:
                    break
                ticker = futures[future]
                completed += 1
                done_tickers.add(ticker)
                if progress_cb:
                    progress_cb(completed, total, ticker)
                try:
                    records = future.result(timeout=_per_ticker_timeout)
                    _process_records(records, ticker)
                except Exception as _fe:
                    on_deck.append({"ticker": ticker, "_rejected": True,
                        "_reason": "Error: " + str(_fe)[:80]})
                # Small pause between tickers — keeps Finnhub under 60 calls/min
                pass  # No delay needed — FMP Premium rate limit is 750/min
        except Exception:
            pass  # TimeoutError or other - return whatever completed so far

        # Cancel any futures still running (hung yfinance calls)
        for future in futures:
            future.cancel()

        # Log any that never completed
        for ticker in scan_list:
            if ticker not in done_tickers:
                on_deck.append({"ticker": ticker, "_rejected": True,
                    "_reason": "Timed out - yfinance hung, skipped"})

    go_now.sort(  key=lambda x: (x.get("vol_spike", False), x.get("confidence", 0)), reverse=True)
    watching.sort(key=lambda x: (x.get("vol_spike", False), x.get("confidence", 0)), reverse=True)
    on_deck.sort( key=lambda x: x.get("confidence", 0), reverse=True)

    return go_now, watching, on_deck, market_bias


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    _user_email = st.session_state.get("user_email", "")
    if _user_email:
        st.markdown("<div style='font-size:0.65rem;color:#A1A1A6;margin-bottom:4px'>Signed in as<br><b style='color:#F5F5F5'>%s</b></div>" % _user_email, unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True, key="logout_btn"):
            # Clear everything — full session wipe
            keys_to_clear = [
                "authenticated", "tos_agreed", "user_email", "user_id", "is_admin",
                "watchlist_loaded", "wq_loaded", "watch_queue", "user_watchlist",
                "onboarding_complete", "onboarding_step", "_paper_trades_loaded",
                "_access_token", "_refresh_token", "_last_token_refresh",
                "auto_scan_go_now", "auto_scan_watching", "auto_scan_on_deck",
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.rerun()
    st.markdown("---")
    selected_ticker = st.selectbox("TICKER", WATCHLIST)
    custom = st.text_input("Or type ticker symbol", "", placeholder="e.g. NVDA").upper().strip()
    if custom:
        # Validate - tickers are 1-5 uppercase letters only, no spaces or full names
        import re as _re
        if _re.match(r'^[A-Z]{1,5}$', custom):
            selected_ticker = custom
        else:
            st.error("Enter a ticker symbol only (e.g. NVDA, AAPL) - not a company name")
    selected_tf = st.selectbox("CHART TIMEFRAME", list(TIMEFRAMES.keys()), index=2)
    st.caption("Signals use automatic timeframes per mode.")
    st.markdown("---")
    st.markdown("**PATTERNS TO SCAN**")
    tog_db    = st.toggle("Double Bottom (calls)", value=True)
    tog_br_up = st.toggle("Break & Retest Up (calls)", value=True)
    tog_dt    = st.toggle("Double Top (puts)",    value=True)
    tog_br_dn = st.toggle("Break & Retest Down (puts)", value=True)
    toggles   = {"db":tog_db, "dt":tog_dt, "br":tog_br_up or tog_br_dn}
    st.markdown("---")
    st.markdown("**ACCOUNT SETTINGS**")
    account_size = st.number_input("Account Size ($)", value=10000, step=1000)
    risk_pct     = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    st.markdown("**⚡ Quick DTE** (weekly/0DTE)")
    dte_quick = st.selectbox("Quick expiry", [0,1,2,3,5,7], index=2,
                             help="0 = 0DTE, 1-7 = this week", label_visibility="collapsed")
    st.markdown("**📅 Swing DTE** (multi-week)")
    dte_swing = st.selectbox("Swing expiry", [14,21,30,45,60], index=2,
                             label_visibility="collapsed")
    trade_style = "both"  # always show both
    st.markdown("---")
    st.markdown("**AUTO REFRESH**")
    refresh_on       = st.toggle("Live refresh (manual)", value=False)
    refresh_interval = st.selectbox("Interval",["1 min","5 min","15 min"],index=1) if refresh_on else None
    st.markdown("---")
    if FMP_API_KEY:       st.success("LIVE DATA - FMP Premium")
    elif FINNHUB_API_KEY: st.success("LIVE DATA - Finnhub")
    elif POLYGON_API_KEY: st.success("LIVE DATA - Polygon")
    else:               st.warning("DEMO MODE")
    if ANTHROPIC_API_KEY: st.success("AI BRIEF READY")
    else:                 st.info("AI Brief: add ANTHROPIC_API_KEY to enable")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.success("📲 TELEGRAM CONNECTED")
        if st.button("Send Test Alert", key="tg_test"):
            import requests as _req
            _msg = (
                "📡 *PaidButPressured Test Alert*\n"
                "✅ Telegram is connected and working!\n"
                "GO NOW signals will fire here automatically."
            )
            _req.post(
                "https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": _msg, "parse_mode": "Markdown"},
                timeout=5
            )
            st.success("Test sent! Check Telegram.")
    else:
        st.info("📲 Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in Railway to enable alerts")

# Auto-refresh - NEVER fire while a scan is running.
# _BG_RESULTS["running"] is the authoritative flag - session_state.scan_running is not used.
init_watch_queue()
_queue_active = any(item["status"] != "CONFIRMED" for item in st.session_state.watch_queue.values())
try:
    _bg_running_now = _BG_RESULTS.get("running", False)
except Exception:
    _bg_running_now = False
# Auto-refresh completely disabled - background thread handles scanning,
# watch queue is in its own tab. Manual refresh button on SCAN tab.
# Only the manual_autorefresh fires IF user explicitly enables it in sidebar.
if AUTOREFRESH_AVAILABLE and not _bg_running_now:
    if refresh_on and refresh_interval and not _queue_active:
        ms = {"1 min":60000,"5 min":300000,"15 min":900000}.get(refresh_interval,300000)
        st_autorefresh(interval=ms, key="manual_autorefresh")

tf_mult,tf_span,tf_days = TIMEFRAMES[selected_tf]
df            = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)
current_price = fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
prev_close    = float(df["close"].iloc[-2]) if len(df)>1 else current_price
pct_change    = ((current_price-prev_close)/prev_close)*100
iv_rank, hv   = fetch_iv_rank(selected_ticker)
earnings_days = check_earnings(selected_ticker)

# ── ATR calculation (14-period) ───────────────────────────────────────────────
def calc_atr(df, period=14):
    if len(df) < period + 1: return None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 2)

atr = calc_atr(df)

# ── Higher timeframe confluence ───────────────────────────────────────────────
# Pull daily bars to check if the higher timeframe trend agrees with the signal
@st.cache_data(ttl=300)
def fetch_htf_trend(ticker):
    """Fetch daily data and return trend + RSI for confluence check."""
    try:
        raw = _yf_download(ticker, period="60d", interval="1d")
        if raw is None or raw.empty or len(raw) < 20: return None, None, None
        raw = _clean_df(raw)
        close = _col(raw, "close")
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - (100/(1+(gain/loss)))).iloc[-1])
        trend = "bullish" if price > ema20 else "bearish"
        return trend, round(rsi, 1), round(ema20, 2)
    except:
        return None, None, None

htf_trend, htf_rsi, htf_ema = fetch_htf_trend(selected_ticker)

# Liquidity check (cached, runs silently in background)
liq_ok, liq_vol, liq_oi, liq_msg = check_liquidity(selected_ticker)

# ── Background watch loop ────────────────────────────────────────────────────
# Watch queue rendering moved to WATCH QUEUE tab - no more top-of-page reruns
any_new_confirm = run_background_watch_checks(tf_mult, tf_span, tf_days)

# Sound alert only - no banner rendered here
if any_new_confirm:
    st.markdown("""
    <audio autoplay>
      <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YWoGAACBhYqFjpGTlZaXl5eWlZORjomEfnhyb" type="audio/wav">
    </audio>
    """, unsafe_allow_html=True)


mstatus, mtext = get_market_status()
css_class = {"open":"market-open","pre":"market-pre","after":"market-pre","closed":"market-closed"}.get(mstatus,"market-closed")
st.markdown(f"<div class='{css_class}'>{mtext}</div>", unsafe_allow_html=True)

# ── AUTO-SCAN ENGINE ──────────────────────────────────────────────────────────
SCAN_INTERVAL = 300  # 5 minutes

def should_run_auto_scan():
    if not st.session_state.auto_scan_enabled: return False
    last = st.session_state.auto_scan_last_run
    if last is None: return True
    return (datetime.now() - last).total_seconds() >= SCAN_INTERVAL

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND SCAN ENGINE
# Runs in a daemon thread completely separate from Streamlit's render cycle.
# Streamlit never runs the scan itself - it only reads results from shared state.
# This means reruns, watch queue updates, and auto-refresh NEVER interrupt a scan.
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level shared state - persists across Streamlit reruns in the same process
_BG_LOCK    = _threading.Lock()
_BG_TRIGGER = _threading.Event()   # set this to kick off an immediate scan
_BG_RESULTS = {
    "go_now":    [],
    "watching":  [],
    "on_deck":   [],
    "mkt_bias":  "neutral",
    "last_run":  None,
    "running":   False,
    "progress":  "",
    "new_go":    [],
}
_BG_THREAD_STARTED = False

def _bg_scan_loop():
    """
    Daemon thread - runs forever, sleeps between scans.
    Wakes up either on _BG_TRIGGER.set() (manual trigger)
    or every 5 minutes automatically when auto-scan is enabled.
    Never touches Streamlit state directly.
    """
    import time as _time

    while True:
        # Wait for trigger or 5-minute auto interval
        triggered = _BG_TRIGGER.wait(timeout=300)
        _BG_TRIGGER.clear()

        # Read settings from shared results dict (written by Streamlit on settings change)
        with _BG_LOCK:
            scan_list    = _BG_RESULTS.get("scan_list",    ["SPY", "QQQ", "IWM"])
            toggles      = _BG_RESULTS.get("toggles",      {"db": True, "dt": True, "br": True})
            account_size = _BG_RESULTS.get("account_size", 10000)
            risk_pct     = _BG_RESULTS.get("risk_pct",     0.01)
            dte_quick    = _BG_RESULTS.get("dte_quick",    3)
            dte_swing    = _BG_RESULTS.get("dte_swing",    30)
            max_premium  = _BG_RESULTS.get("max_premium",  15.0)
            style        = _BG_RESULTS.get("style",        "both")
            auto_enabled = _BG_RESULTS.get("auto_enabled", False)
            prev_go      = _BG_RESULTS.get("go_now",       [])

        # Only auto-scan if enabled; always scan on manual trigger
        if not triggered and not auto_enabled:
            continue

        with _BG_LOCK:
            _BG_RESULTS["running"]  = True
            _BG_RESULTS["progress"] = "Starting scan..."
            _BG_RESULTS["new_go"]   = []

        try:
            import signal as _signal

            def _progress_cb(idx, total, ticker):
                with _BG_LOCK:
                    _BG_RESULTS["progress"]       = "Scanning %s..." % ticker
                    _BG_RESULTS["progress_idx"]   = idx + 1
                    _BG_RESULTS["progress_total"] = total

            go, watching, deck, mkt = full_scan(
                scan_list, toggles, account_size, risk_pct,
                dte_quick, dte_swing, max_premium, style,
                progress_cb=_progress_cb
            )

            prev_tickers = {(r["ticker"], r.get("style","")) for r in prev_go}
            new_go = [r for r in go if (r["ticker"], r.get("style","")) not in prev_tickers]

            with _BG_LOCK:
                _BG_RESULTS["go_now"]   = go
                _BG_RESULTS["watching"] = watching
                _BG_RESULTS["on_deck"]  = deck
                _BG_RESULTS["mkt_bias"] = mkt
                _BG_RESULTS["last_run"] = datetime.now()
                _BG_RESULTS["running"]  = False
                _BG_RESULTS["progress"] = "Complete - %s GO NOW, %s WATCHING" % (len(go), len(watching))
                _BG_RESULTS["new_go"]   = new_go

            # Telegram handled by inline scan with proper gate checks
            # Background thread just saves signal history
            for r in new_go:
                try:
                    save_signal_history(r)
                except Exception:
                    pass

            # Save to Supabase
            try:
                save_scan_state(go, watching, deck)
            except Exception:
                pass

        except Exception as _e:
            with _BG_LOCK:
                _BG_RESULTS["running"]   = False
                _BG_RESULTS["last_run"]  = datetime.now()
                _BG_RESULTS["progress"]  = "❌ Scan error: %s" % str(_e)[:120]
                _BG_RESULTS["go_now"]    = []
                _BG_RESULTS["watching"]  = []
                _BG_RESULTS["on_deck"]   = []

        _time.sleep(2)  # brief pause before accepting next trigger


def start_bg_scan_thread():
    """Start the background scan thread once per process lifetime."""
    global _BG_THREAD_STARTED
    if not _BG_THREAD_STARTED:
        t = _threading.Thread(target=_bg_scan_loop, daemon=True, name="bg_scanner")
        t.start()
        _BG_THREAD_STARTED = True

def trigger_scan(scan_list, toggles, account_size, risk_pct,
                 dte_quick, dte_swing, max_premium, style, auto_enabled=False):
    """
    Called by Streamlit to kick off a scan.
    Writes settings to shared state, fires the trigger event.
    Returns immediately - scan runs in background.
    """
    with _BG_LOCK:
        _BG_RESULTS["scan_list"]    = scan_list
        _BG_RESULTS["toggles"]      = toggles
        _BG_RESULTS["account_size"] = account_size
        _BG_RESULTS["risk_pct"]     = risk_pct
        _BG_RESULTS["dte_quick"]    = dte_quick
        _BG_RESULTS["dte_swing"]    = dte_swing
        _BG_RESULTS["max_premium"]  = max_premium
        _BG_RESULTS["style"]        = style
        _BG_RESULTS["auto_enabled"] = auto_enabled
    _BG_TRIGGER.set()

def get_bg_results():
    """Thread-safe read of latest scan results."""
    with _BG_LOCK:
        return dict(_BG_RESULTS)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_supabase():
    """Returns a Supabase client if configured, else None."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def get_user_id():
    """
    Gets or creates a persistent user ID stored in the URL query params.
    This is how we identify users without full auth.
    On first visit a UUID is generated and added to the URL.
    On return visits the same ID is read from the URL.
    """
    import uuid
    params = st.query_params
    uid = params.get("uid", None)
    if not uid:
        uid = str(uuid.uuid4())[:8]  # short 8-char ID, easy to share
        st.query_params["uid"] = uid
    return uid

def load_user_data(user_id):
    """Load all user data from Supabase user_data table."""
    sb = get_supabase()
    if not sb or not user_id: return {}
    try:
        import json as _j
        res = sb.table("user_data").select("*").eq("user_id", user_id).execute()
        if res.data:
            row = res.data[0]
            return {
                "watchlist":   _j.loads(row.get("watchlist",  "[]")),
                "watch_queue": _j.loads(row.get("watch_queue", "{}")),
                "preferences": _j.loads(row.get("preferences", "{}")),
            }
    except Exception:
        pass
    return {}

def save_user_data(user_id, watchlist=None, watch_queue=None, preferences=None):
    """Save user data to Supabase."""
    sb = get_supabase()
    if not sb or not user_id: return False
    try:
        import json as _j
        row = {"user_id": str(user_id), "updated_at": datetime.now(tz=pytz.UTC).isoformat()}
        if watchlist   is not None: row["watchlist"]   = _j.dumps(watchlist)
        if watch_queue is not None: row["watch_queue"] = _j.dumps(watch_queue)
        if preferences is not None: row["preferences"] = _j.dumps(preferences)
        sb.table("user_data").upsert(row).execute()
        return True
    except Exception as e:
        print("save_user_data error:", str(e))
        return False

# Keep old names as aliases for compatibility
def load_watchlist_db(user_id):
    return load_user_data(user_id).get("watchlist", None)

def save_watchlist_db(user_id, tickers):
    # Always use the authenticated session user_id, not the passed-in one
    _uid = st.session_state.get("user_id") or user_id
    save_user_data(_uid, watchlist=tickers)

def save_scan_state(go_now, watching, on_deck):
    """Persist latest scan results to Supabase so refresh doesn't wipe them."""
    sb = get_supabase()
    if not sb:
        return
    try:
        import json as _j
        def _safe(lst):
            out = []
            for r in lst[:20]:  # cap at 20 per bucket
                try:
                    _j.dumps(r)  # test serializable
                    out.append(r)
                except Exception:
                    pass
            return out
        sb.table("scan_state").upsert({
            "id":        "latest",
            "go_now":    _j.dumps(_safe(go_now)),
            "watching":  _j.dumps(_safe(watching)),
            "on_deck":   _j.dumps(_safe(on_deck)),
            "updated_at": datetime.now(tz=pytz.UTC).isoformat(),
        }).execute()
    except Exception:
        pass

def load_scan_state():
    """Load last scan results from Supabase on app start."""
    sb = get_supabase()
    if not sb:
        return [], [], []
    try:
        import json as _j
        res = sb.table("scan_state").select("*").eq("id","latest").execute()
        if res.data:
            d = res.data[0]
            return (
                _j.loads(d.get("go_now",  "[]")),
                _j.loads(d.get("watching","[]")),
                _j.loads(d.get("on_deck", "[]")),
            )
    except Exception:
        pass
    return [], [], []

def save_paper_trades(trades):
    """Persist paper trades to Supabase so they survive redeploys."""
    sb = get_supabase()
    if not sb: return
    try:
        import json as _j
        user_id = st.session_state.get("user_id") or get_user_id()
        if not user_id: return
        serializable = []
        for t in trades:
            try:
                _j.dumps(t)
                serializable.append(t)
            except Exception:
                pass
        sb.table("paper_trades_state").upsert({
            "user_id":    str(user_id),
            "trades":     _j.dumps(serializable),
            "updated_at": datetime.now(tz=pytz.UTC).isoformat(),
        }).execute()
    except Exception:
        pass

def load_paper_trades():
    """Load paper trades from Supabase on app start."""
    sb = get_supabase()
    if not sb: return []
    try:
        import json as _j
        user_id = st.session_state.get("user_id") or get_user_id()
        if not user_id: return []
        res = sb.table("paper_trades_state").select("trades").eq("user_id", str(user_id)).execute()
        if res.data:
            return _j.loads(res.data[0].get("trades", "[]"))
    except Exception:
        pass
    return []


def save_signal_history(r):
    """Save a fired signal to Supabase signal_history table."""
    sb = get_supabase()
    if not sb:
        return
    try:
        opt = r.get("opt", {})
        sb.table("signal_history").insert({
            "ticker":      r.get("ticker"),
            "action":      r.get("action"),
            "pattern":     r.get("pattern"),
            "style":       r.get("style"),
            "confidence":  r.get("confidence"),
            "entry":       opt.get("entry"),
            "target":      opt.get("target"),
            "stop":        opt.get("stop"),
            "strike":      opt.get("strike"),
            "premium":     opt.get("premium"),
            "rr":          opt.get("rr_option"),
            "signals_hit": r.get("signals_hit", 0),
            "gates":       r.get("gates_passed", 0),
            "fired_at":    datetime.now(tz=pytz.UTC).isoformat(),
        }).execute()
    except Exception:
        pass  # never crash the app over a db write

def load_signal_history(limit=50):
    """Load recent signal history from Supabase."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        res = sb.table("signal_history") \
                .select("*") \
                .order("fired_at", desc=True) \
                .limit(limit) \
                .execute()
        return res.data if res.data else []
    except Exception:
        return []

def init_user_watchlist():
    """
    Called once on load. Uses Supabase Auth user_id to load all user data.
    Watch queue always reads fresh from Supabase.
    Watchlist only loads once per session to avoid overwriting user changes.
    """
    user_id = st.session_state.get("user_id") or get_user_id()
    st.session_state.user_id = user_id

    # Load all user data in one call
    user_data = load_user_data(user_id)

    # Restore watchlist
    if user_data.get("watchlist"):
        st.session_state.user_watchlist = user_data["watchlist"]

    # Restore watch queue
    # Always load watch queue fresh from Supabase — no caching
    _wq_data = user_data.get("watch_queue", {})
    if isinstance(_wq_data, dict) and _wq_data:
        for key, item in _wq_data.items():
            for ts_field in ["added_at", "last_checked"]:
                if item.get(ts_field) and isinstance(item[ts_field], str):
                    try:
                        item[ts_field] = datetime.fromisoformat(item[ts_field])
                    except Exception:
                        item[ts_field] = datetime.now()
        st.session_state.watch_queue = _wq_data
    else:
        st.session_state.watch_queue = {}
    st.session_state.wq_loaded = True

    # Restore last scan results
    if not st.session_state.get("auto_scan_go_now"):
        go, wa, od = load_scan_state()
        if go or wa or od:
            st.session_state.auto_scan_go_now   = go
            st.session_state.auto_scan_watching = wa
            st.session_state.auto_scan_on_deck  = od

    st.session_state.watchlist_loaded = True

init_user_watchlist()  # call immediately after definition

start_bg_scan_thread()  # start background scanner daemon

# Load paper trades from Supabase now that functions are defined
# Always reload paper trades when authenticated — ensures correct user's trades load
if st.session_state.get("authenticated") and not st.session_state.get("_paper_trades_loaded"):
    _pt = load_paper_trades()
    if _pt:
        st.session_state.paper_trades = _pt
    st.session_state._paper_trades_loaded = True
elif not st.session_state.paper_trades:
    _pt = load_paper_trades()
    if _pt:
        st.session_state.paper_trades = _pt


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM ALERT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def send_make_webhook(r):
    """
    Fires signal data to Make.com webhook for automated Canva + Instagram posting.
    Only fires from admin account.
    """
    if not MAKE_WEBHOOK_URL:
        return
    try:
        import urllib.request, json
        opt = r.get("opt", {})
        payload = json.dumps({
            "ticker":     r.get("ticker", ""),
            "action":     r.get("action", ""),
            "direction":  r.get("direction", ""),
            "pattern":    r.get("pattern", ""),
            "style":      r.get("style", ""),
            "confidence": r.get("confidence", 0),
            "entry":      round(float(opt.get("entry", 0)), 2),
            "target":     round(float(opt.get("target", 0)), 2),
            "stop":       round(float(opt.get("stop", 0)), 2),
            "strike":     round(float(opt.get("strike", 0)), 2),
            "premium":    round(float(opt.get("premium", 0)), 2),
            "rr":         opt.get("rr", 0),
            "expiration": opt.get("expiration", ""),
            "gates":      r.get("gates_passed", 0),
            "signals":    r.get("signals_hit", 0),
            "fired_at":   datetime.now().strftime("%m/%d/%Y %I:%M %p ET"),
        }).encode("utf-8")
        req = urllib.request.Request(
            MAKE_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Never block the scan if Make is down


def send_telegram_alert(r, alert_type="GO NOW"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import urllib.request, json
    is_bull = r["direction"] == "bullish"
    opt     = r["opt"]
    action  = "CALL" if is_bull else "PUT"
    emoji   = "\U0001f7e2" if is_bull else "\U0001f534"
    bucket_emoji = {"GO NOW": "\U0001f6a8", "WATCHING": "\U0001f440", "ON DECK": "\U0001f4cb"}.get(alert_type, "\U0001f4e1")
    detail      = r.get("detail", {}) or {}
    sig_detail  = r.get("signal_detail", []) or detail.get("signal_detail", [])
    signals_hit = r.get("signals_hit", 0) or detail.get("signals_hit", 0)
    sig_lines   = "\n".join(["  " + s for s in sig_detail]) if sig_detail else ("  %s/5 signals confirmed" % signals_hit)
    lines = [
        "%s <b>%s - %s %s</b>" % (bucket_emoji, alert_type, r["ticker"], action),
        "\u2501" * 20,
        "%s Pattern: <b>%s</b>" % (emoji, r["pattern"]),
        "\U0001f4b0 Entry:   <b>$%.2f</b>" % r["price"],
        "\U0001f3af Target:  <b>$%.2f</b>" % opt["target"],
        "\U0001f6d1 Stop:    <b>$%.2f</b>" % opt["stop"],
        "\U0001f4ca Strike:  <b>$%.2f</b>  |  Exp: <b>%s</b>" % (opt["strike"], opt["expiration"]),
        "\U0001f4b5 Premium: <b>$%.2f/sh</b>  |  RR: <b>%.1fx</b>" % (opt["premium"], opt.get("rr_option", 0)),
        "\u2501" * 20,
        "\U0001f9e0 Confidence: <b>%s%%</b>  |  Gates: <b>%s/7</b>" % (r["confidence"], r["gates_passed"]),
        "\U0001f4f6 Signals:",
        sig_lines,
        "\u2501" * 20,
        "<i>Not financial advice. Paper trade first.</i>",
    ]
    msg = "\n".join(lines)
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}).encode("utf-8")
    try:
        req = urllib.request.Request(
            "https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
            data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass



def send_telegram_exit_alert(t):
    """Sends a paper trade exit notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import urllib.request, json

    is_win   = t["status"] == "WIN"
    emoji    = "✅" if is_win else "❌"
    pnl_sign = "+" if t["pnl_pct"] >= 0 else ""

    msg = (
        "%s *PAPER TRADE CLOSED - %s %s*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Result:  `%s`\n"
        "Reason:  `%s`\n"
        "P&L:     `%s%.1f%%  ($%+.0f)`\n"
        "Peak:    `+%.1f%%`\n"
        "Held:    `%s scan cycles`\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "_Entry $%.2f → Exit $%.2f_"
    ) % (
        emoji, t["ticker"], t["action"],
        t["status"],
        t.get("exit_reason", ""),
        pnl_sign, t["pnl_pct"], t["pnl_dollar"],
        t["peak_pnl_pct"],
        t["cycles_open"],
        t["entry_price"], t.get("exit_price", 0),
    )

    payload = json.dumps({
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       msg,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def paper_enter_trade(r):
    """Auto-enter a paper trade from a GO NOW signal."""
    trades = st.session_state.paper_trades
    key = (r["ticker"], r["style"], r["direction"])
    # Don't double-enter same ticker/direction
    for t in trades:
        if t["status"] == "OPEN" and (t["ticker"], t["style"], t["direction"]) == key:
            return
    opt = r["opt"]
    trade = {
        "id":            len(trades) + 1,
        "ticker":        r["ticker"],
        "action":        r["action"],
        "direction":     r["direction"],
        "style":         r["style"],
        "pattern":       r["pattern"],
        "entry_price":   r["price"],          # stock price at entry
        "entry_premium": opt["premium"],       # option premium at entry
        "strike":        opt["strike"],
        "expiration":    opt["expiration"],
        "target":        opt["target"],        # stock price target
        "stop":          opt["stop"],          # stock price stop
        "contracts":     opt.get("contracts", 1),
        "max_loss":      opt.get("max_loss", 0),
        "profit_target": opt.get("profit_at_target", 0),
        "confidence":    r["confidence"],
        "gates_passed":  r["gates_passed"],
        "signals_hit":   r.get("signals_hit", 0),
        "entered_at":    datetime.now().strftime("%m/%d %I:%M%p"),
        "entered_ts":    datetime.now().isoformat(),
        "status":        "OPEN",
        "exit_reason":   None,
        "exit_price":    None,
        "exit_premium":  None,
        "exit_ts":       None,
        "pnl_pct":       0.0,
        "pnl_dollar":    0.0,
        "peak_pnl_pct":  0.0,
        "cycles_open":   0,
    }
    st.session_state.paper_trades.append(trade)
    save_paper_trades(st.session_state.paper_trades)

# Profit-taking thresholds by trade style
PAPER_PROFIT_TARGET = {
    "quick": 30,   # exit quick trades at +30% premium gain
    "swing": 50,   # exit swing trades at +50% premium gain
}
PAPER_STOP_LOSS_PCT = -20  # exit any trade at -20% premium loss

def paper_check_exits():
    """
    Run on every scan cycle. Checks all open paper trades for:
    - Profit target hit: +30% (quick) or +50% (swing) premium gain
    - Full price target hit on underlying stock
    - Stop hit on underlying stock
    - Premium decay -20%: hard stop loss
    - Expired: past expiration date
    Updates P&L and status in place.
    """
    trades = st.session_state.paper_trades
    for t in trades:
        if t["status"] != "OPEN":
            continue
        t["cycles_open"] += 1
        cur = fetch_current_price(t["ticker"])
        if cur is None:
            continue
        cur = float(cur)
        is_bull = t["direction"] == "bullish"

        # Use actual delta from trade record for more accurate P&L
        delta = abs(float(t.get("opt", {}).get("delta", 0.5) or 0.5))
        delta = max(0.1, min(0.9, delta))

        stock_move   = cur - t["entry_price"]
        premium_move = stock_move * delta if is_bull else -stock_move * delta
        cur_premium  = max(0.01, t["entry_premium"] + premium_move)
        pnl_pct      = (cur_premium - t["entry_premium"]) / t["entry_premium"] * 100
        pnl_dollar   = (cur_premium - t["entry_premium"]) * 100 * t["contracts"]

        t["pnl_pct"]    = round(pnl_pct, 1)
        t["pnl_dollar"] = round(pnl_dollar, 2)
        t["cur_price"]  = round(cur, 2)
        t["cur_premium"] = round(cur_premium, 2)
        if pnl_pct > t["peak_pnl_pct"]:
            t["peak_pnl_pct"] = round(pnl_pct, 1)

        # Profit target threshold for this trade style
        profit_threshold = PAPER_PROFIT_TARGET.get(t.get("style", "swing"), 50)

        # Check exit conditions - ordered by priority
        exit_reason = None
        is_win      = False

        if is_bull and cur >= t["target"]:
            exit_reason = "TARGET HIT"
            is_win = True
        elif not is_bull and cur <= t["target"]:
            exit_reason = "TARGET HIT"
            is_win = True
        elif pnl_pct >= profit_threshold:
            exit_reason = "+%s%% PROFIT TARGET" % profit_threshold
            is_win = True
        elif is_bull and cur <= t["stop"]:
            exit_reason = "STOP HIT"
            is_win = False
        elif not is_bull and cur >= t["stop"]:
            exit_reason = "STOP HIT"
            is_win = False
        elif pnl_pct <= PAPER_STOP_LOSS_PCT:
            exit_reason = "PREMIUM %s%%" % PAPER_STOP_LOSS_PCT
            is_win = False
        else:
            try:
                from datetime import date
                exp_str  = t["expiration"]
                exp_date = datetime.strptime(exp_str, "%%m/%%d/%%y").date() if "/" in exp_str else None
                if exp_date and date.today() > exp_date:
                    exit_reason = "EXPIRED"
                    is_win = pnl_pct > 0
            except Exception:
                pass

        if exit_reason:
            t["status"]       = "WIN" if is_win else "LOSS"
            t["is_win"]       = is_win  # save flag for win rate tracker
            t["exit_reason"]  = exit_reason
            t["exit_price"]   = round(cur, 2)
            t["exit_premium"] = round(cur_premium, 2)
            t["exit_ts"]      = datetime.now().strftime("%m/%d %I:%M%p")
            t["pnl_pct"]      = round(pnl_pct, 1)
            t["pnl_dollar"]   = round(pnl_dollar, 2)
            send_telegram_exit_alert(t)
            # Log to journal automatically
            journal = load_journal()
            journal.append({
                "Ticker":      t["ticker"],
                "Action":      t["action"],
                "Pattern":     t["pattern"],
                "Entry $":     t["entry_price"],
                "Strike":      t["strike"],
                "Premium":     t["entry_premium"],
                "Exit Premium":t["exit_premium"],
                "P&L %":        "%+.1f%%" % t["pnl_pct"],
                "P&L $":       "%+.0f" % t["pnl_dollar"],
                "Result":      t["status"],
                "Exit Reason": exit_reason,
                "Source":      "Paper Auto",
                "Date":        t["entered_at"],
            })
            st.session_state.trade_journal = journal[-200:]

def paper_close_trade(trade_id, reason="MANUAL CLOSE"):
    """Manually close a paper trade."""
    for t in st.session_state.paper_trades:
        if t["id"] == trade_id and t["status"] == "OPEN":
            cur = fetch_current_price(t["ticker"])
            cur = float(cur) if cur else t["entry_price"]
            is_bull = t["direction"] == "bullish"
            stock_move   = cur - t["entry_price"]
            premium_move = stock_move * 0.5 if is_bull else -stock_move * 0.5
            cur_premium  = max(0.01, t["entry_premium"] + premium_move)
            pnl_pct      = (cur_premium - t["entry_premium"]) / t["entry_premium"] * 100
            pnl_dollar   = (cur_premium - t["entry_premium"]) * 100 * t["contracts"]
            t["status"]       = "WIN" if pnl_pct >= 0 else "LOSS"
            t["exit_reason"]  = reason
            t["exit_price"]   = round(cur, 2)
            t["exit_premium"] = round(cur_premium, 2)
            t["exit_ts"]      = datetime.now().strftime("%m/%d %I:%M%p")
            t["pnl_pct"]      = round(pnl_pct, 1)
            t["pnl_dollar"]   = round(pnl_dollar, 2)
            break

# Auto-scan: keep background thread settings in sync whenever auto-scan is enabled
def sync_bg_auto_scan():
    """Push current sidebar settings into bg engine and enable auto mode."""
    cfg = st.session_state.auto_scan_settings
    sl  = st.session_state.user_watchlist if cfg.get("scan_list","watchlist")=="watchlist" else SCAN_UNIVERSE
    with _BG_LOCK:
        _BG_RESULTS["scan_list"]    = sl
        _BG_RESULTS["toggles"]      = toggles
        _BG_RESULTS["account_size"] = account_size
        _BG_RESULTS["risk_pct"]     = risk_pct
        _BG_RESULTS["dte_quick"]    = dte_quick
        _BG_RESULTS["dte_swing"]    = dte_swing
        _BG_RESULTS["max_premium"]  = cfg.get("max_premium", max_premium)
        _BG_RESULTS["style"]        = cfg.get("style", "both")
        _BG_RESULTS["auto_enabled"] = True

if st.session_state.auto_scan_enabled:
    sync_bg_auto_scan()

_bg_status = get_bg_results()

# ── GO NOW ALERT BANNER - fired from background thread new_go list ─────────────
_new_go_now = _bg_status.get("new_go", [])
# Deduplicate - only show banners for signals we haven't shown yet this session
_shown_banners = st.session_state.get("shown_banners", set())
for ng in _new_go_now:
    _bkey = "%s_%s_%s" % (ng["ticker"], ng.get("style",""), _bg_status.get("last_run",""))
    if _bkey in _shown_banners:
        continue
    _shown_banners.add(_bkey)
    st.session_state.shown_banners = _shown_banners
    is_bull_ng = ng["direction"] == "bullish"
    dc_ng = "#D4AF37" if is_bull_ng else "#C1121F"
    st.markdown("""
    <div style='background:#1A1500;border:2px solid #D4AF37;border-radius:10px;padding:14px 18px;margin:6px 0'>
        <div style='font-family:monospace;font-size:0.65rem;letter-spacing:3px;color:#22C55E;margin-bottom:4px'>🚨 NEW GO NOW SIGNAL</div>
        <div style='font-size:1.1rem;font-weight:700;color:%s'>%s - %s</div>
        <div style='font-size:0.8rem;color:#A1A1A6;margin-top:2px'>%s · %s%%%% · %s/7 gates · Strike $%.2f · Target $%.2f · Stop $%.2f</div>
    </div>
    """ % (dc_ng, "BUY CALL" if is_bull_ng else "BUY PUT", ng["ticker"],
           ng["pattern"], ng["confidence"], ng["gates_passed"],
           ng["opt"]["strike"], ng["opt"]["target"], ng["opt"]["stop"]),
    unsafe_allow_html=True)
    st.components.v1.html("""<script>
    try {
        var ctx=new(window.AudioContext||window.webkitAudioContext)();
        [440,554,659].forEach(function(f,i){
            var o=ctx.createOscillator(),g=ctx.createGain();
            o.connect(g);g.connect(ctx.destination);
            o.frequency.value=f;o.type="sine";
            g.gain.setValueAtTime(0.3,ctx.currentTime+i*0.18);
            g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+i*0.18+0.4);
            o.start(ctx.currentTime+i*0.18);o.stop(ctx.currentTime+i*0.18+0.4);
        });
    } catch(e){}
    </script>""", height=0)
    # Auto-enter paper trade for new GO NOW
    if st.session_state.get("paper_auto_enabled", True):
        paper_enter_trade(ng)

# auto_scan_poll removed - background thread runs independently,
# no page refresh needed to trigger it.

if earnings_days is not None:
    if earnings_days <= 1:   st.error(f"EARNINGS {'TODAY' if earnings_days==0 else 'TOMORROW'} on {selected_ticker} - Avoid new options positions.")
    elif earnings_days <= 7: st.error(f"EARNINGS IN {earnings_days} DAYS on {selected_ticker} - 7-point gate will block.")
    else:                    st.warning(f"Earnings in {earnings_days} days on {selected_ticker} - premiums may be inflated.")

c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1:
    color   = "#D4AF37" if pct_change>=0 else "#C1121F"
    arrow   = "UP" if pct_change>=0 else "DN"
    prepost = "" if mstatus=="open" else " <span style='color:#F6E27A;font-size:0.72rem'>(delayed)</span>"
    st.markdown(f"<div class='metric-card'><div style='color:#A1A1A6;font-size:0.8rem'>{selected_ticker} . {selected_tf}</div><div class='big-price'>${current_price:,.2f}{prepost}</div><div style='color:{color}'>{arrow} {pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
with c2:
    ema20v = float(df["close"].ewm(span=20).mean().iloc[-1])
    above  = current_price > ema20v
    st.markdown(f"<div class='metric-card'><div style='color:#A1A1A6;font-size:0.75rem'>TREND</div><div style='font-weight:700;color:{'#D4AF37' if above else '#C1121F'}'>{'BULL' if above else 'BEAR'}</div></div>", unsafe_allow_html=True)
with c3:
    vol = float(df["volume"].iloc[-1])
    st.markdown(f"<div class='metric-card'><div style='color:#A1A1A6;font-size:0.75rem'>VOLUME</div><div style='font-weight:700'>{vol/1e6:.1f}M</div></div>", unsafe_allow_html=True)
with c4:
    iv_color = "#D4AF37" if iv_rank is not None and iv_rank<50 else "#F6E27A" if iv_rank is not None and iv_rank<70 else "#C1121F"
    iv_text  = f"{iv_rank}%" if iv_rank is not None else "N/A"
    st.markdown(f"<div class='metric-card'><div style='color:#A1A1A6;font-size:0.75rem'>IV RANK</div><div style='font-weight:700;color:{iv_color}'>{iv_text}</div></div>", unsafe_allow_html=True)

div = detect_rsi_divergence(df)
if div:
    css = f"divergence-{'bull' if div['type']=='bullish' else 'bear'}"
    st.markdown(f"<div class='{css}'><b>{div['label']}</b><br>{div['detail']}</div>", unsafe_allow_html=True)

tab4,tab1,tab2,tab8,tab7 = st.tabs(["SCAN","SIGNALS","CHART","WATCH QUEUE","HOW IT WORKS"])

with tab1:
    cands_quick, tfs_quick = build_multi_tf_candidates(selected_ticker, toggles, account_size, risk_pct, dte_quick, "quick", atr=atr)
    cands_swing, tfs_swing = build_multi_tf_candidates(selected_ticker, toggles, account_size, risk_pct, dte_swing, "swing", atr=atr)

    no_quick = len(cands_quick) == 0
    no_swing = len(cands_swing) == 0

    if no_quick and no_swing:
        st.markdown("""<div style='background:#111827;border:2px solid #2A2A2D;border-radius:12px;padding:24px;text-align:center;color:#A1A1A6'>
            <div style='font-size:1rem;font-weight:700;margin:8px 0'>NO SIGNALS FOUND</div>
            <div style='font-size:0.85rem'>Try Daily or 4 Hour timeframe, enable more patterns, or check a different ticker.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Use first candidate list that has signals for shared logic below
        candidates = cands_quick if not no_quick else cands_swing
        # Side-by-side columns: Quick (purple) | Swing (blue)
        col_q, col_s = st.columns(2)
        with col_q:
            st.markdown(f"<div style='background:#1a0a3a;border-radius:6px;padding:6px 12px;text-align:center;color:#aa88ff;font-family:monospace;font-size:0.75rem;letter-spacing:1px'>⚡ QUICK &nbsp;|&nbsp; {dte_quick}DTE</div>", unsafe_allow_html=True)
        with col_s:
            st.markdown(f"<div style='background:#0a1a2a;border-radius:6px;padding:6px 12px;text-align:center;color:#A1A1A6;font-family:monospace;font-size:0.75rem;letter-spacing:1px'>📅 SWING &nbsp;|&nbsp; {dte_swing}DTE</div>", unsafe_allow_html=True)

        with col_q:
            render_signal_cards(cands_quick, selected_ticker, dte_quick, "quick", "q",
                                df, current_price, atr, iv_rank, earnings_days,
                                mstatus, mtext, account_size, risk_pct,
                                htf_trend, htf_rsi, htf_ema, liq_ok)
        with col_s:
            render_signal_cards(cands_swing, selected_ticker, dte_swing, "swing", "s",
                                df, current_price, atr, iv_rank, earnings_days,
                                mstatus, mtext, account_size, risk_pct,
                                htf_trend, htf_rsi, htf_ema, liq_ok)

with tab2:
    # ── Detect patterns for annotation ────────────────────────────────────────
    chart_db    = [s for s in detect_double_bottom(df, selected_ticker, rr_min=2.0) if s.confirmed]
    chart_dt    = [s for s in detect_double_top(df, selected_ticker, rr_min=2.0)    if s.confirmed]
    chart_br    = [s for s in detect_break_and_retest(df, selected_ticker, rr_min=2.0) if s.confirmed]
    # Sort all setups by confidence, show only the best one on chart
    chart_setups_all = chart_db + chart_dt + chart_br
    chart_setups = sorted(chart_setups_all,
        key=lambda s: getattr(s, "confidence", 0), reverse=True)[:1]

    # ── Build candle + volume data for Lightweight Charts ────────────────────
    try:
        import json as _json
        _df = df.copy()
        _df["timestamp"] = pd.to_datetime(_df["timestamp"])
        _df = _df.sort_values("timestamp").tail(120)  # last 120 candles

        def _ts(t):
            """Convert to unix timestamp (int) for lightweight-charts"""
            return int(pd.Timestamp(t).timestamp())

        candle_data = [
            {"time": _ts(row.timestamp),
             "open":  round(float(row.open),  4),
             "high":  round(float(row.high),  4),
             "low":   round(float(row.low),   4),
             "close": round(float(row.close), 4)}
            for _, row in _df.iterrows()
        ]

        vol_data = [
            {"time":  _ts(row.timestamp),
             "value": float(row.volume),
             "color": "#D4AF3744" if float(row.close) >= float(row.open) else "#C1121F44"}
            for _, row in _df.iterrows()
        ]

        # EMA 20
        ema_vals = _df["close"].ewm(span=20).mean()
        ema_data = [
            {"time": _ts(row.timestamp), "value": round(float(ema_vals.iloc[i]), 4)}
            for i, (_, row) in enumerate(_df.iterrows())
        ]

        # VWAP
        _tp   = (_df["high"] + _df["low"] + _df["close"]) / 3
        _vwap = (_tp * _df["volume"]).cumsum() / _df["volume"].cumsum()
        vwap_data = [
            {"time": _ts(row.timestamp), "value": round(float(_vwap.iloc[i]), 4)}
            for i, (_, row) in enumerate(_df.iterrows())
        ]

        # ── Pattern markers (circles on key candles) ─────────────────────────
        markers = []
        for s in chart_setups[:3]:
            is_bull  = s.direction == "bullish"
            pat_name = s.pattern.replace("Double", "Double ").replace("BreakRetest", "Break & Retest")

            # Find the candle closest to entry price to mark pattern confirmed
            closest_idx = (_df["close"] - s.entry_price).abs().idxmin()
            closest_row = _df.loc[closest_idx]
            marker_time = _ts(closest_row["timestamp"])

            # For double bottom: find the two lowest candles near stop level
            if "Bottom" in pat_name:
                lows_near_stop = _df[(_df["low"] - s.stop_loss).abs() < s.stop_loss * 0.015]
                for i, (idx, row) in enumerate(lows_near_stop.tail(2).iterrows()):
                    markers.append({
                        "time":     _ts(row["timestamp"]),
                        "position": "belowBar",
                        "color":    "#D4AF37",
                        "shape":    "circle",
                        "text":     "B%s" % (i+1)
                    })
                markers.append({
                    "time":     marker_time,
                    "position": "aboveBar",
                    "color":    "#D4AF37",
                    "shape":    "arrowUp",
                    "text":     "ENTRY ▲ %s" % pat_name
                })

            # For double top: find the two highest candles near stop level
            elif "Top" in pat_name:
                highs_near_stop = _df[(_df["high"] - s.stop_loss).abs() < s.stop_loss * 0.015]
                for i, (idx, row) in enumerate(highs_near_stop.tail(2).iterrows()):
                    markers.append({
                        "time":     _ts(row["timestamp"]),
                        "position": "aboveBar",
                        "color":    "#C1121F",
                        "shape":    "circle",
                        "text":     "T%s" % (i+1)
                    })
                markers.append({
                    "time":     marker_time,
                    "position": "belowBar",
                    "color":    "#C1121F",
                    "shape":    "arrowDown",
                    "text":     "ENTRY ▼ %s" % pat_name
                })

            # For break & retest
            else:
                arrow = "arrowUp" if is_bull else "arrowDown"
                pos   = "aboveBar" if not is_bull else "belowBar"
                markers.append({
                    "time":     marker_time,
                    "position": pos,
                    "color":    "#D4AF37" if is_bull else "#C1121F",
                    "shape":    arrow,
                    "text":     "ENTRY %s" % pat_name
                })

        # ── Price lines for entry / target / stop ─────────────────────────────
        price_lines = []
        for s in chart_setups[:1]:  # show lines for best setup only
            is_bull = s.direction == "bullish"
            price_lines += [
                {"price": round(s.entry_price, 4), "color": "#D4AF37" if is_bull else "#C1121F",
                 "lineWidth": 2, "lineStyle": 0, "axisLabelVisible": True,
                 "title": "Entry $%.2f" % s.entry_price},
                {"price": round(s.target, 4), "color": "#D4AF37",
                 "lineWidth": 1, "lineStyle": 1, "axisLabelVisible": True,
                 "title": "Target $%.2f" % s.target},
                {"price": round(s.stop_loss, 4), "color": "#C1121F",
                 "lineWidth": 1, "lineStyle": 2, "axisLabelVisible": True,
                 "title": "Stop $%.2f" % s.stop_loss},
            ]

        # ── Render via inline HTML (TradingView Lightweight Charts CDN) ───────
        chart_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body {{ margin:0; background:#0a0e17; }}
  #chart {{ width:100%%; height:480px; }}
  #legend {{ position:absolute; top:8px; left:12px; z-index:10;
            font-family:monospace; font-size:11px; color:#A1A1A6;
            background:rgba(10,14,23,0.85); padding:6px 10px;
            border-radius:6px; border:1px solid #2A2A2D; pointer-events:none; }}
</style>
</head>
<body>
<div id="legend">Loading...</div>
<div id="chart"></div>
<script>
const chartEl = document.getElementById('chart');
const chart = LightweightCharts.createChart(chartEl, {{
  width:  chartEl.offsetWidth || 800,
  height: 480,
  layout: {{ background: {{ color: '#0a0e17' }}, textColor: '#A1A1A6' }},
  grid:   {{ vertLines: {{ color: '#111827' }}, horzLines: {{ color: '#111827' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#2A2A2D' }},
  timeScale: {{ borderColor: '#2A2A2D', timeVisible: true, secondsVisible: false }},
}});

// Candlestick series
const candles = chart.addCandlestickSeries({{
  upColor: '#D4AF37', downColor: '#C1121F',
  borderUpColor: '#D4AF37', borderDownColor: '#C1121F',
  wickUpColor: '#D4AF37', wickDownColor: '#C1121F',
}});
candles.setData({candles});
candles.setMarkers({markers});

// Price lines
{pricelines}

// EMA 20
const ema = chart.addLineSeries({{ color: '#F6E27A', lineWidth: 1,
  lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false,
  lastValueVisible: false, title: 'EMA20' }});
ema.setData({ema});

// VWAP
const vwap = chart.addLineSeries({{ color: '#9966ff', lineWidth: 1,
  lineStyle: LightweightCharts.LineStyle.LargeDashed, priceLineVisible: false,
  lastValueVisible: false, title: 'VWAP' }});
vwap.setData({vwap});

// Volume (separate pane)
const volSeries = chart.addHistogramSeries({{
  priceFormat: {{ type: 'volume' }},
  priceScaleId: 'vol',
  scaleMargins: {{ top: 0.8, bottom: 0 }},
}});
volSeries.setData({vol});

chart.timeScale().fitContent();

// Crosshair legend
const legend = document.getElementById('legend');
chart.subscribeCrosshairMove(param => {{
  if (!param.time) {{ legend.textContent = ''; return; }}
  const c = param.seriesData.get(candles);
  if (c) {{
    const chg = ((c.close - c.open) / c.open * 100).toFixed(2);
    const clr = c.close >= c.open ? '#D4AF37' : '#C1121F';
    legend.innerHTML =
      '<span style="color:#F5F5F5;font-weight:700">{ticker}</span>  ' +
      'O:<span style="color:' + clr + '">' + c.open.toFixed(2) + '</span>  ' +
      'H:<span style="color:' + clr + '">' + c.high.toFixed(2) + '</span>  ' +
      'L:<span style="color:' + clr + '">' + c.low.toFixed(2)  + '</span>  ' +
      'C:<span style="color:' + clr + '">' + c.close.toFixed(2) + '</span>  ' +
      '<span style="color:' + clr + '">' + (chg > 0 ? '+' : '') + chg + '%%</span>';
  }}
}});

// Responsive resize
window.addEventListener('resize', () => chart.resize(chartEl.offsetWidth, 480));
</script>
</body>
</html>
""".format(
    candles   = _json.dumps(candle_data),
    markers   = _json.dumps(sorted(markers, key=lambda x: x["time"])),
    pricelines= "\n".join([
        "candles.createPriceLine({{price:{p},color:'{c}',lineWidth:{w},lineStyle:{ls},axisLabelVisible:true,title:'{t}'}});".format(
            p=pl["price"], c=pl["color"], w=pl["lineWidth"],
            ls=pl["lineStyle"], t=pl["title"]
        ) for pl in price_lines
    ]),
    ema       = _json.dumps(ema_data),
    vwap      = _json.dumps(vwap_data),
    vol       = _json.dumps(vol_data),
    ticker    = selected_ticker,
)

        # ── Pattern signal cards ───────────────────────────────────────────────
        if chart_setups:
            st.markdown(
                "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>",
                unsafe_allow_html=True
            )
            for s in chart_setups[:1]:  # best setup only
                is_bull  = s.direction == "bullish"
                pat_name = s.pattern.replace("Double","Double ").replace("BreakRetest","Break & Retest")
                border   = "#D4AF37" if is_bull else "#C1121F"
                action   = "CALL ▲" if is_bull else "PUT ▼"
                st.markdown(
                    "<div style='background:#0B0B0C;border:1px solid %s;border-radius:8px;"
                    "padding:8px 14px;font-size:0.75rem;min-width:180px'>"
                    "<span style='color:%s;font-weight:700'>%s %s</span><br>"
                    "<span style='color:#A1A1A6'>%s</span><br>"
                    "<span style='color:#F5F5F5'>Entry $%.2f · Target $%.2f · Stop $%.2f</span>"
                    "</div>" % (
                        border, border, action, selected_ticker,
                        pat_name,
                        s.entry_price, s.target, s.stop_loss
                    ),
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("No confirmed patterns detected on current timeframe.")

        st.components.v1.html(chart_html, height=490, scrolling=False)

        # Legend
        st.markdown(
            "<div style='font-size:0.68rem;color:#556677;margin-top:4px'>"
            "<span style='color:#F6E27A'>- EMA 20</span> &nbsp;"
            "<span style='color:#9966ff'>-- VWAP</span> &nbsp;"
            "<span style='color:#D4AF37'>● Pattern markers on chart</span>"
            "</div>",
            unsafe_allow_html=True
        )

    except Exception as _chart_err:
        st.error("Chart error: %s" % str(_chart_err))
        st.caption("Falling back - check Railway logs for details.")

with tab4:
    st.markdown("<div class='section-title'>MARKET SCANNER</div>", unsafe_allow_html=True)

    # ── Watchlist Manager ─────────────────────────────────────────────────────
    _uid = st.session_state.get("user_id", "local")
    _db_active = bool(SUPABASE_URL and SUPABASE_KEY)
    _wl_label  = "📋 My Watchlist (%s tickers)%s" % (
        len(st.session_state.user_watchlist),
        " · ☁️ saved" if _db_active else " · 💾 session only"
    )

    with st.expander(_wl_label, expanded=False):
        if not _db_active:
            st.caption("⚠️ Watchlist resets on browser close. Add SUPABASE_URL + SUPABASE_KEY to Railway to save permanently.")

        # Current watchlist as removable chips
        wl   = st.session_state.user_watchlist
        cols = st.columns(min(len(wl), 6)) if wl else []
        for i, tkr in enumerate(wl):
            with cols[i % len(cols)]:
                if st.button("✕ %s" % tkr, key="wl_remove_%s" % tkr, use_container_width=True):
                    if tkr in st.session_state.user_watchlist:
                        st.session_state.user_watchlist.remove(tkr)
                        save_watchlist_db(_uid, st.session_state.user_watchlist)
                    st.rerun()

        # Add ticker
        add_col1, add_col2 = st.columns([3,1])
        with add_col1:
            new_ticker = st.text_input("Add ticker", placeholder="e.g. NVDA AAPL TSLA",
                                        label_visibility="collapsed", key="wl_add_input").upper().strip()
        with add_col2:
            if st.button("Add", key="wl_add_btn", use_container_width=True, type="primary"):
                tickers_to_add = [t.strip() for t in new_ticker.replace(",", " ").split() if t.strip()]
                added = 0
                for t in tickers_to_add:
                    if t and t not in st.session_state.user_watchlist:
                        st.session_state.user_watchlist.append(t)
                        added += 1
                if added:
                    save_watchlist_db(_uid, st.session_state.user_watchlist)
                st.rerun()

        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("Reset to Default (SPY, QQQ, IWM)", key="wl_reset", use_container_width=True):
                st.session_state.user_watchlist = list(DEFAULT_WATCHLIST)
                save_watchlist_db(_uid, st.session_state.user_watchlist)
                st.rerun()
        with rc2:
            if _db_active:
                st.caption("☁️ Your ID: `%s` - watchlist saves automatically" % _uid)
            else:
                st.caption("Tip: Add multiple at once - NVDA AAPL TSLA")

    # Auto-scan settings
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scan_style = st.radio("Scan Mode", ["⚡ Quick","📅 Swing","Both"], index=2, horizontal=True)
        scan_style_key = "quick" if "Quick" in scan_style else "swing" if "Swing" in scan_style else "both"
        st.session_state.auto_scan_settings["style"] = scan_style_key
    with sc2:
        max_premium = st.number_input("Max Premium ($/sh)", value=25.00, step=0.50, min_value=0.50)
        st.session_state.auto_scan_settings["max_premium"] = max_premium
    with sc3:
        _sector_options = list(SECTOR_LISTS.keys())
        scan_universe_choice = st.selectbox("Scan Universe", _sector_options, index=0)

        # Build scan list from selection
        if scan_universe_choice == "My Watchlist":
            scan_list = st.session_state.user_watchlist or ["SPY","QQQ","IWM"]
        elif scan_universe_choice == "Full Universe":
            scan_list = SCAN_UNIVERSE
        else:
            scan_list = SECTOR_LISTS.get(scan_universe_choice, SCAN_UNIVERSE)

        scan_list = list(dict.fromkeys(scan_list))  # deduplicate
        st.caption("%s tickers" % len(scan_list))
        st.session_state.auto_scan_settings["scan_list"] = scan_universe_choice



    # Show auto-scan results if available, else prompt manual scan
    has_auto_results = len(st.session_state.auto_scan_go_now + st.session_state.auto_scan_watching + st.session_state.auto_scan_on_deck) > 0
    if has_auto_results:
        last_t = st.session_state.auto_scan_last_run
        last_str = last_t.strftime("%I:%M:%S %p") if last_t else "unknown"
        st.caption(f"Showing auto-scan results from {last_str} · {len(scan_list)} tickers scanned")
        go_now   = st.session_state.auto_scan_go_now
        watching = st.session_state.auto_scan_watching
        on_deck  = st.session_state.auto_scan_on_deck
        mkt_bias = st.session_state.auto_scan_mkt
        if st.button("🔄 Scan Now", use_container_width=True):
            with st.spinner("Scanning..."):
                go_now, watching, on_deck, mkt_bias = full_scan(
                    scan_list, toggles, account_size, risk_pct,
                    dte_quick, dte_swing, max_premium, scan_style_key
                )
                st.session_state.auto_scan_go_now   = go_now
                st.session_state.auto_scan_watching = watching
                st.session_state.auto_scan_on_deck  = on_deck
                st.session_state.auto_scan_mkt      = mkt_bias
                st.session_state.auto_scan_last_run = datetime.now()
                st.rerun()
    else:
        st.caption(f"Scanning {len(scan_list)} tickers through full precision stack")
        # ── Live scan status from background thread ──────────────────────
        # ── Inline scan - runs directly in Streamlit, results stored in session state
        # Background thread doesn't work on Railway multi-worker deployments
        # (each worker has its own memory space, results never reach the page)

        go_now   = st.session_state.get("scan_go_now",   [])
        watching = st.session_state.get("scan_watching", [])
        on_deck  = st.session_state.get("scan_on_deck",  [])
        mkt_bias = st.session_state.get("scan_mkt",      "neutral")

        _run_btn = st.button("🔍 RUN SCAN", type="primary", use_container_width=True)
        _demo_mode = False

        if _run_btn and _demo_mode:
            # Inject realistic fake signals for testing
            import random
            _fake_tickers = [
                ("NVDA","bullish","Double Bottom","quick",82,134.50,138.00,132.00,135.00,2.45,2.3,5,4),
                ("XLF","bearish","Double Top","quick",86,49.03,47.50,50.20,49.00,1.85,2.1,5,3),
                ("AAPL","bullish","Break & Retest","swing",74,221.50,228.00,218.00,222.50,4.20,2.0,4,3),
                ("SPY","bearish","Double Top","swing",65,568.00,560.00,572.00,568.00,12.50,2.2,4,3),
                ("AMD","bullish","Double Bottom","quick",58,102.00,106.00,99.50,102.00,1.90,2.4,3,2),
            ]
            _go_now_demo = []
            _watching_demo = []
            _on_deck_demo = []
            for tk,dr,pat,sty,conf,entr,tgt,stp,strk,prem,rr,gates,sigs in _fake_tickers:
                _r = {
                    "ticker": tk, "direction": dr, "action": "CALL" if dr=="bullish" else "PUT",
                    "pattern": pat, "style": sty, "confidence": conf,
                    "gates_passed": gates, "signals_hit": sigs,
                    "entry_status": "CONFIRMED", "exh_confirmed": conf >= 80,
                    "price": entr, "low_rr": False, "vol_spike": sigs >= 4,
                    "rel_vol": round(random.uniform(1.2, 3.0), 1),
                    "block_detected": False, "sq_state": "none", "sq_compression": 0,
                    "market_bias": "bearish", "sector_bias": "neutral",
                    "iv_rank": random.randint(20, 55),
                    "earn_days": None, "elevate": conf >= 85,
                    "detail": {"signals_hit": sigs, "exhaustion_confirmed": conf>=80,
                               "exhaustion_score": 2, "exhaustion_reasons": [],
                               "signal_detail": [], "against_market_bias": dr=="bullish"},
                    "opt": {"strike": strk, "premium": prem, "entry": entr,
                            "target": tgt, "stop": stp, "rr": rr,
                            "rr_option": rr, "delta": 0.52, "delta_ok": True,
                            "contracts": 2, "max_loss": round(prem*200,0),
                            "profit_at_target": round(prem*100*2*0.5,0),
                            "position_dollars": round(prem*200,2),
                            "pct_of_account": round(prem*200/account_size*100,1),
                            "expiration": "2026-03-21", "actual_dte": 5,
                            "exit_take_half": round(prem*1.5,2),
                            "exit_stop_stock": stp,
                            "rr_stock": rr},
                    "sig": {"direction": dr, "pattern_label": pat, "confidence": conf,
                            "entry_price": entr, "target": tgt, "stop_loss": stp,
                            "factors": {}, "trade_style": sty, "regime": "trending",
                            "conflict": False},
                }
                if conf >= 75 and gates >= 5 and _r["exh_confirmed"] and sigs >= 3:
                    _go_now_demo.append(_r)
                elif conf >= 65 and gates >= 4 and sigs >= 3:
                    _watching_demo.append(_r)
                else:
                    _on_deck_demo.append(_r)

            go_now, watching, on_deck, mkt_bias = _go_now_demo, _watching_demo, _on_deck_demo, "bearish"
            st.session_state.scan_go_now   = go_now
            st.session_state.scan_watching = watching
            st.session_state.scan_on_deck  = on_deck
            st.session_state.scan_mkt      = mkt_bias
            st.session_state.scan_last_run = datetime.now()
            st.success("🧪 Demo mode - %s GO NOW · %s WATCHING · %s ON DECK" % (len(go_now), len(watching), len(on_deck)))

        elif _run_btn:
            # Kill all autorefresh before scan starts
            prog_bar  = st.progress(0)
            prog_text = st.empty()

            def _cb(idx, total, ticker):
                prog_bar.progress(idx / total)
                prog_text.markdown(
                    "<div style='font-size:0.78rem;color:#A1A1A6'>"
                    "⏳ <b>Scanning %s...</b> &nbsp;·&nbsp; %s / %s tickers</div>" % (ticker, idx, total),
                    unsafe_allow_html=True
                )

            go_now, watching, on_deck, mkt_bias = full_scan(
                scan_list, toggles, account_size, risk_pct,
                dte_quick, dte_swing, max_premium, scan_style_key,
                progress_cb=_cb
            )
            prog_bar.empty()
            prog_text.empty()

            st.session_state.scan_go_now   = go_now
            st.session_state.scan_watching = watching
            st.session_state.scan_on_deck  = on_deck
            st.session_state.scan_mkt      = mkt_bias
            st.session_state.scan_last_run = datetime.now()

            paper_check_exits()

            # Fire Telegram + paper trades for GO NOW signals
            # ── Telegram fires ONLY for highest conviction signals
            # Telegram is now manual — admin hits "Send to Telegram" button on each card
            # Auto-firing removed to give full control over what gets alerted
            for r in go_now:
                try: save_signal_history(r)
                except: pass

            # ── REGIME DETECTION ENGINE ───────────────────────────────────────
            try:
                # Layer 1: Breadth
                _breadth_score, _bull_pct, _bear_pct = calculate_breadth_score(go_now, watching, on_deck)

                # Layer 2: Index health
                _index_health = check_index_health("SPY")

                # Layer 3: Rally authenticity
                _rally_auth, _rally_detail = check_rally_authenticity("SPY")

                # Layer 4: Classify regime
                _regime_data = classify_market_regime(
                    _breadth_score, _index_health, _rally_auth, go_now, watching
                )

                # Layer 5: Adjust signals
                go_now   = apply_regime_adjustments(go_now, _regime_data)
                watching = apply_regime_adjustments(watching, _regime_data)

                # Store regime in session state for display
                st.session_state.market_regime = _regime_data
                st.session_state.breadth_score  = _breadth_score
                st.session_state.bull_pct        = _bull_pct
                st.session_state.bear_pct        = _bear_pct
                st.session_state.rally_auth      = _rally_auth
                st.session_state.rally_detail    = _rally_detail
                st.session_state.index_health    = _index_health

            except Exception as _re:
                st.session_state.market_regime = {"regime": "UNKNOWN", "color": "#A1A1A6",
                                                   "desc": "Regime analysis unavailable", "bias": "neutral"}

            save_scan_state(go_now, watching, on_deck)

    # ── show completion banner
    _last_run = st.session_state.get("scan_last_run")
    if _last_run:
        elapsed = int((datetime.now() - _last_run).total_seconds())
        if elapsed < 15:
            st.markdown(
                "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:8px;"
                "padding:10px 14px;font-size:0.82rem;color:#D4AF37;margin-bottom:8px'>"
                "✅ Scan complete &nbsp;·&nbsp; <b>%s GO NOW</b> &nbsp;·&nbsp; "
                "%s WATCHING &nbsp;·&nbsp; %s ON DECK</div>" % (
                    len(go_now), len(watching), len(on_deck)),
                unsafe_allow_html=True
            )

    # ── REGIME DISPLAY BANNER ──────────────────────────────────────────────────
    _regime = st.session_state.get("market_regime")
    if _regime and _regime.get("regime") not in ["UNKNOWN", None]:
        _rc     = _regime.get("color", "#A1A1A6")
        _rname  = _regime.get("regime", "NEUTRAL")
        _rdesc  = _regime.get("desc", "")
        _bs     = st.session_state.get("breadth_score", 0)
        _bpct   = st.session_state.get("bull_pct", 50)
        _bepct  = st.session_state.get("bear_pct", 50)
        _rauth  = st.session_state.get("rally_auth", "")
        _ih     = st.session_state.get("index_health", {})
        _rsi    = _ih.get("rsi", "N/A")
        _t5     = _ih.get("trend_5d", "neutral").upper()
        _t20    = _ih.get("trend_20d", "neutral").upper()

        _rally_line = ""
        if _rauth in ["FALSE", "SUSPECT"]:
            _rally_line = " &nbsp;·&nbsp; <span style='color:#C1121F'>⚠️ Rally: %s</span>" % _rauth

        st.markdown(
            "<div style='background:%s18;border:1px solid %s44;border-radius:10px;"
            "padding:12px 16px;margin-bottom:10px'>"
            "<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div>"
            "<span style='color:%s;font-weight:700;font-size:0.9rem'>📡 %s</span>"
            "<span style='color:#A1A1A6;font-size:0.75rem;margin-left:10px'>%s</span>"
            "%s"
            "</div>"
            "<div style='text-align:right;font-size:0.72rem;color:#A1A1A6'>"
            "🟢 %s%% CALLS &nbsp; 🔴 %s%% PUTS<br>"
            "5D: <b style='color:%s'>%s</b> &nbsp; 20D: <b style='color:%s'>%s</b> &nbsp; RSI: %s"
            "</div>"
            "</div>"
            "</div>" % (
                _rc, _rc, _rc, _rname, _rdesc, _rally_line,
                _bpct, _bepct,
                "#22C55E" if _t5 == "BULLISH" else "#C1121F", _t5,
                "#22C55E" if _t20 == "BULLISH" else "#C1121F", _t20,
                _rsi
            ),
            unsafe_allow_html=True
        )

    # ── Debug: show why tickers were rejected ──────────────────────────────────
    rejected = [r for r in on_deck if r.get("_rejected")]
    real_on_deck = [r for r in on_deck if not r.get("_rejected")]
    on_deck = real_on_deck

    last_run = st.session_state.get("scan_last_run") or st.session_state.get("auto_scan_last_run")

    # ── Temp diagnostic - shows why signals landed in ON DECK vs GO NOW ────────
    # Signal breakdown removed — internal data stays internal
    # ──────────────────────────────────────────────────────────────────────────

    # ── Win Rate Badge ─────────────────────────────────────────────────────────
    _all_trades   = st.session_state.get("paper_trades", [])
    _closed       = [t for t in _all_trades if t.get("status") not in ["OPEN", None]]
    _wins         = [t for t in _closed if t.get("is_win") or t.get("status") == "WIN"]
    _losses       = [t for t in _closed if not (t.get("is_win") or t.get("status") == "WIN")]
    _total_closed = len(_closed)
    _win_rate     = round(len(_wins) / _total_closed * 100) if _total_closed > 0 else None
    _open_count   = len([t for t in _all_trades if t.get("status") == "OPEN"])

    if _total_closed > 0:
        _wr_color = "#D4AF37" if _win_rate >= 60 else "#F6E27A" if _win_rate >= 45 else "#C1121F"
        st.markdown(
            "<div style='background:#1A1A1D;border:1px solid %s44;border-radius:10px;"
            "padding:12px 16px;margin-top:8px;display:flex;align-items:center;gap:16px'>"
            "<div style='text-align:center'>"
            "<div style='font-size:1.6rem;font-weight:700;color:%s'>%s%%</div>"
            "<div style='font-size:0.65rem;color:#A1A1A6;letter-spacing:1px'>WIN RATE</div>"
            "</div>"
            "<div style='width:1px;height:36px;background:#2A2A2D'></div>"
            "<div style='display:flex;gap:20px;font-size:0.78rem'>"
            "<div><div style='color:#D4AF37;font-weight:700'>%s</div><div style='color:#A1A1A6;font-size:0.68rem'>WINS</div></div>"
            "<div><div style='color:#C1121F;font-weight:700'>%s</div><div style='color:#A1A1A6;font-size:0.68rem'>LOSSES</div></div>"
            "<div><div style='color:#F6E27A;font-weight:700'>%s</div><div style='color:#A1A1A6;font-size:0.68rem'>OPEN</div></div>"
            "<div><div style='color:#F5F5F5;font-weight:700'>%s</div><div style='color:#A1A1A6;font-size:0.68rem'>TOTAL</div></div>"
            "</div>"
            "<div style='margin-left:auto;font-size:0.68rem;color:#4a5568'>Paper trading · not financial advice</div>"
            "</div>" % (
                _wr_color, _wr_color, _win_rate,
                len(_wins), len(_losses),
                _open_count, _total_closed
            ),
            unsafe_allow_html=True
        )
    elif _open_count > 0:
        st.markdown(
            "<div style='background:#1A1A1D;border:1px solid #2A2A2D;border-radius:10px;"
            "padding:12px 16px;margin-top:8px;font-size:0.78rem;color:#A1A1A6'>"
            "📊 <b style='color:#F6E27A'>%s open trade%s</b> - win rate will appear when first trade closes"
            "</div>" % (_open_count, "s" if _open_count != 1 else ""),
            unsafe_allow_html=True
        )

    # ── Paper Trade Log ────────────────────────────────────────────────────────
    if _all_trades:
        with st.expander("📊 Paper Trade Log (%s trades)" % len(_all_trades), expanded=False):
            # Open trades first
            _open_trades  = [t for t in _all_trades if t.get("status") == "OPEN"]
            _closed_trades = sorted(
                [t for t in _all_trades if t.get("status") != "OPEN"],
                key=lambda x: x.get("exit_ts", ""), reverse=True
            )

            if _open_trades:
                st.markdown("<div style='font-size:0.65rem;color:#A1A1A6;letter-spacing:2px;margin-bottom:6px'>OPEN POSITIONS</div>", unsafe_allow_html=True)
                for t in _open_trades:
                    _is_bull = t.get("direction") == "bullish"
                    _cur_pnl = t.get("pnl_pct", 0)
                    _pnl_col = "#22C55E" if _cur_pnl > 0 else "#C1121F" if _cur_pnl < 0 else "#A1A1A6"
                    st.markdown(
                        "<div style='background:#1A1A1D;border:1px solid #2A2A2D;border-radius:8px;"
                        "padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center'>"
                        "<div>"
                        "<span style='color:#F5F5F5;font-weight:700'>%s %s</span>"
                        "<span style='color:#A1A1A6;font-size:0.75rem;margin-left:8px'>%s · $%s entry</span>"
                        "</div>"
                        "<div style='text-align:right'>"
                        "<div style='color:%s;font-weight:700'>%+.1f%%</div>"
                        "<div style='color:#A1A1A6;font-size:0.68rem'>OPEN</div>"
                        "</div>"
                        "</div>" % (
                            t.get("ticker","?"), "CALL" if _is_bull else "PUT",
                            t.get("pattern","?"), t.get("entry_price","?"),
                            _pnl_col, _cur_pnl
                        ),
                        unsafe_allow_html=True
                    )

            if _closed_trades:
                st.markdown("<div style='font-size:0.65rem;color:#A1A1A6;letter-spacing:2px;margin:10px 0 6px'>CLOSED TRADES</div>", unsafe_allow_html=True)
                for t in _closed_trades[:20]:
                    _is_win = t.get("is_win") or t.get("status") == "WIN"
                    _result_col = "#22C55E" if _is_win else "#C1121F"
                    _result_emoji = "✅" if _is_win else "❌"
                    st.markdown(
                        "<div style='background:#1A1A1D;border:1px solid %s33;border-radius:8px;"
                        "padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center'>"
                        "<div>"
                        "<span style='color:#F5F5F5;font-weight:700'>%s %s %s</span>"
                        "<span style='color:#A1A1A6;font-size:0.75rem;margin-left:8px'>%s</span>"
                        "</div>"
                        "<div style='text-align:right'>"
                        "<div style='color:%s;font-weight:700'>%+.1f%%</div>"
                        "<div style='color:#A1A1A6;font-size:0.68rem'>%s</div>"
                        "</div>"
                        "</div>" % (
                            _result_col,
                            _result_emoji, t.get("ticker","?"), "CALL" if t.get("direction")=="bullish" else "PUT",
                            t.get("exit_reason","?"),
                            _result_col, t.get("pnl_pct", 0),
                            t.get("exit_ts","?")
                        ),
                        unsafe_allow_html=True
                    )

    if go_now or watching or on_deck or rejected:
        bias_color = "#D4AF37" if mkt_bias=="bullish" else "#C1121F" if mkt_bias=="bearish" else "#F6E27A"
        bias_icon  = "📈" if mkt_bias=="bullish" else "📉" if mkt_bias=="bearish" else "↔️"
        total_found = len(go_now)+len(watching)+len(on_deck)

        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             background:#1A1A1D;border:1px solid {bias_color}33;border-radius:10px;
             padding:10px 16px;margin-bottom:4px;font-size:0.72rem'>
            <div style='color:{bias_color}'>{bias_icon} MARKET: <b>{mkt_bias.upper()}</b></div>
            <div style='display:flex;gap:20px'>
                <span style='color:#22C55E'>● {len(go_now)} GO NOW</span>
                <span style='color:#D4AF37'>● {len(watching)} WATCHING</span>
                <span style='color:#C1121F'>● {len(on_deck)} ON DECK</span>
            </div>
            <div style='color:#A1A1A6'>{total_found} total signals</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mobile-first card renderer ───────────────────────────────────────
        def conf_color(c):
            return "#D4AF37" if c>=90 else "#40d080" if c>=80 else "#F6E27A" if c>=70 else "#6699aa"
        def conf_label(c):
            return "HIGH CONVICTION" if c>=90 else "STRONG" if c>=80 else "WATCH IT" if c>=70 else "WAIT"
        def regime_badge(r):
            alignment = r.get("regime_alignment", "")
            if alignment == "CONFIRMED":
                return "<span style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E33;padding:1px 6px;border-radius:4px;font-size:0.62rem;margin-left:4px'>✅ REGIME</span>"
            elif alignment == "COUNTER":
                return "<span style='background:#C1121F22;color:#C1121F;border:1px solid #C1121F33;padding:1px 6px;border-radius:4px;font-size:0.62rem;margin-left:4px'>⚠️ COUNTER</span>"
            elif alignment == "BLOCKED":
                return "<span style='background:#C1121F44;color:#C1121F;border:1px solid #C1121F;padding:1px 6px;border-radius:4px;font-size:0.62rem;margin-left:4px'>🚫 BLOCKED</span>"
            return ""

        def mobile_card(r, bucket, idx):
            is_bull = r.get("direction", "bullish") == "bullish"
            dc  = "#D4AF37" if is_bull else "#C1121F"
            cc  = conf_color(r.get("confidence", 50))
            cl  = conf_label(r.get("confidence", 50))
            rb  = regime_badge(r)  # regime alignment badge
            opt = r.get("opt", {})

            # ON DECK records may not have full opt - show simplified card
            if not opt:
                action = "CALL" if is_bull else "PUT"
                reason = r.get("_on_deck_reason", "Developing setup")
                st.markdown(
                    "<div style='background:#1A1A1D;border:1px solid #2A2A2D;border-radius:10px;"
                    "padding:12px 14px;margin-bottom:8px'>"
                    "<div style='display:flex;justify-content:space-between;align-items:center'>"
                    "<span style='font-size:1rem;font-weight:700;color:%s'>%s</span>"
                    "<span style='font-size:0.65rem;background:#ffffff11;color:%s;"
                    "padding:2px 6px;border-radius:4px;margin-left:6px'>%s</span>"
                    "<span style='font-size:0.65rem;color:#C1121F;margin-left:auto'>📋 ON DECK</span>"
                    "</div>"
                    "<div style='font-size:0.72rem;color:#A1A1A6;margin-top:4px'>%s</div>"
                    "<div style='font-size:0.68rem;color:#4a5568;margin-top:2px'>%s · conf %s%%</div>"
                    "</div>" % (
                        dc, r.get("ticker","?"), dc, action,
                        r.get("pattern", "Pattern detected"),
                        reason, r.get("confidence", 0)
                    ),
                    unsafe_allow_html=True
                )
                return

            gc  = "#D4AF37" if r.get("gates_passed",0)>=6 else "#F6E27A" if r.get("gates_passed",0)>=5 else "#C1121F"
            exh_ok = r.get("exh_confirmed", False)
            rv     = round(r.get("rel_vol", 1.0), 1)
            block  = r.get("block_detected", False)
            si     = "⚡" if r.get("style","swing")=="quick" else "📅"
            border = "#22C55E44" if bucket=="go_now" else "#D4AF3744" if bucket=="watching" else "#C1121F44"

            # Build card using % string formatting to avoid all quote conflicts
            R = 28
            circ = round(2 * 3.14159 * R, 1)
            dash = round((r["confidence"] / 100) * circ, 1)
            act_bg  = "#D4AF3722" if is_bull else "#C1121F22"
            sty_bg  = "#1a0a3a"  if r["style"] == "quick" else "#0a1a2a"
            sty_fg  = "#aa88ff"  if r["style"] == "quick" else "#A1A1A6"
            blk_tag      = "<span style='font-size:0.58rem;color:#F6E27A'>⚡ BLOCK</span>" if block else ""
            against_bias = r.get("detail", {}).get("against_market_bias", False)
            bias_warn    = (
                " &nbsp;<span style='font-size:0.58rem;color:#f0a030;background:#2a1800;"
                "padding:1px 5px;border-radius:3px'>⚠️ vs market</span>"
                if against_bias else ""
            )
            exh_txt = "✅ confirmed" if exh_ok else "⏳ watching"
            sq_state = r.get("sq_state", "none")
            sq_pct   = r.get("sq_compression", 0)
            sq_tag   = (
                " &nbsp;·&nbsp; <span style='color:#aa88ff'>⚡ SQUEEZE FIRING</span>" if sq_state == "firing" and sq_pct >= 40
                else " &nbsp;·&nbsp; <span style='color:#A1A1A6'>◈ squeeze</span>" if sq_state == "squeeze" and sq_pct >= 40
                else ""
            )
            action  = "CALL" if is_bull else "PUT"
            parts = [
                "<div style='background:#1A1A1D;border:1px solid %s;border-radius:12px;padding:14px 16px;margin-bottom:8px'>" % border,
                "<div style='display:flex;align-items:center;gap:12px'>",
                "<div style='position:relative;width:68px;height:68px;flex-shrink:0'>",
                "<svg width='68' height='68' style='transform:rotate(-90deg);display:block'>",
                "<circle cx='34' cy='34' r='%s' fill='none' stroke='#2A2A2D' stroke-width='5'/>" % R,
                "<circle cx='34' cy='34' r='%s' fill='none' stroke='%s' stroke-width='5' stroke-dasharray='%s %s' stroke-linecap='round'/>" % (R, cc, dash, circ),
                "</svg>",
                "<div style='position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center'>",
                "<div style='font-size:0.95rem;font-weight:700;color:%s;line-height:1'>%s</div>" % (cc, r["confidence"]),
                "<div style='font-size:0.42rem;color:%s;letter-spacing:1px;margin-top:1px'>%%</div>" % cc,
                "</div></div>",
                "<div style='flex:1;min-width:0'>",
                "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px'>",
                "<span style='font-size:1.05rem;font-weight:700;color:%s'>%s</span>" % (dc, r["ticker"]),
                "<span style='font-size:0.6rem;background:%s;color:%s;padding:2px 6px;border-radius:4px;font-weight:700'>%s</span>" % (act_bg, dc, action),
                "<span style='font-size:0.58rem;background:%s;color:%s;padding:2px 6px;border-radius:4px'>%s %s</span>" % (sty_bg, sty_fg, si, r["style"].upper()),
                blk_tag,
                "</div>",
                "<div style='font-size:0.69rem;color:#A1A1A6'>%s%s</div>" % (r["pattern"], bias_warn),
                "<div style='font-size:0.65rem;color:#A1A1A6;margin-top:2px'>%sx vol &nbsp;·&nbsp; %s%s</div>" % (rv, exh_txt, sq_tag),
                "</div>",
                "<div style='text-align:right;flex-shrink:0'>",
                "<div style='font-size:0.56rem;font-weight:700;color:%s;background:%s22;padding:2px 7px;border-radius:6px;letter-spacing:1px;margin-bottom:5px;display:inline-block'>%s</div>%s" % (cc, cc, cl, rb),
                "<div style='font-size:0.65rem;color:#A1A1A6'>Gate <span style='color:%s;font-weight:700'>%s/7</span></div>" % (gc, r["gates_passed"]),
                "<div style='font-size:0.65rem;color:#A1A1A6;margin-top:3px'>Entry <span style='color:#F5F5F5;font-weight:700'>$%.2f</span></div>" % r["price"],
                "<div style='font-size:0.65rem;color:#A1A1A6;margin-top:2px'>Strike <span style='color:#F5F5F5;font-weight:700'>$%.2f</span></div>" % opt["strike"],
                "</div></div></div>",
            ]
            st.markdown("".join(parts), unsafe_allow_html=True)

            with st.expander(f"📊 {r['ticker']} full details"):
                c1, c2 = st.columns(2)
                items_l = [("TARGET", f"${opt['target']:.2f}", "#D4AF37"),
                           ("PREMIUM", f"${opt['premium']:.2f}/sh", "#F5F5F5"),
                           ("MAX LOSS", f"${opt['max_loss']:,.0f}", "#C1121F"),
                           ("IV RANK", f"{r['iv_rank']}%" if r["iv_rank"] else "N/A", "#F6E27A")]
                items_r = [("STOP OUT", f"${opt['stop']:.2f}", "#C1121F"),
                           ("EST PROFIT", f"${opt['profit_at_target']:,.0f}", "#D4AF37"),
                           ("R:R", f"{opt['rr_option']:.1f}x", "#D4AF37" if opt["rr_option"]>=2 else "#F6E27A"),
                           ("EXPIRES", opt["expiration"], "#A1A1A6")]
                with c1:
                    for lbl, val, col in items_l:
                        st.markdown(
                            "<div style='background:#1A1A1D;border-radius:8px;padding:10px;margin-bottom:6px'>"
                            "<div style='font-size:0.58rem;color:#A1A1A6'>%s</div>"
                            "<div style='font-size:0.95rem;font-weight:700;color:%s'>%s</div></div>" % (lbl, col, val),
                            unsafe_allow_html=True)
                with c2:
                    for lbl, val, col in items_r:
                        st.markdown(
                            "<div style='background:#1A1A1D;border-radius:8px;padding:10px;margin-bottom:6px'>"
                            "<div style='font-size:0.58rem;color:#A1A1A6'>%s</div>"
                            "<div style='font-size:0.95rem;font-weight:700;color:%s'>%s</div></div>" % (lbl, col, val),
                            unsafe_allow_html=True)

                side = "below" if is_bull else "above"
                _c  = opt.get("contracts", 1)
                _exit_line = (
                    "Sell %s of %s contracts at $%.2f/sh" % (_c // 2, _c, opt["exit_take_half"])
                    if _c >= 2 else
                    "Close full position at $%.2f/sh (1 contract)" % opt["exit_take_half"]
                )
                st.markdown(
                    "<div style='background:#0B0B0C;border-radius:8px;padding:10px 12px;font-size:0.72rem;color:#A1A1A6;margin:2px 0 8px;line-height:1.6'>"
                    "<span style='color:#D4AF37;font-weight:700'>%s</span> &nbsp;·&nbsp;"
                    "<span style='color:#C1121F;font-weight:700'>Close all</span> if %s $%.2f</div>" % (_exit_line, side, opt['stop']),
                    unsafe_allow_html=True)

                sig_detail = r.get("signal_detail", [])
                exh        = r.get("exh_reasons", [])
                signals_hit = r.get("signals_hit", 0)
                if sig_detail:
                    st.markdown("<div style='font-size:0.58rem;color:#A1A1A6;letter-spacing:2px;margin-bottom:4px'>SIGNAL CHECK (%s/6)</div>" % signals_hit, unsafe_allow_html=True)
                    for item in sig_detail:
                        good = item.startswith("✅")
                        tcol = "#F5F5F5" if good else "#A1A1A6"
                        st.markdown("<div style='font-size:0.73rem;color:%s;padding:2px 0'>%s</div>" % (tcol, item), unsafe_allow_html=True)

                # Fibonacci confluence display
                _fib_detail = r.get("detail", {})
                if isinstance(_fib_detail, dict) and _fib_detail.get("fib_confirmed"):
                    _fib_level = _fib_detail.get("fib_level", "")
                    _fib_price = _fib_detail.get("fib_level_price", 0)
                    _fib_high  = _fib_detail.get("fib_swing_high", 0)
                    _fib_low   = _fib_detail.get("fib_swing_low", 0)
                    _fib_color = "#F6E27A" if _fib_level != "61.8%" else "#D4AF37"
                    st.markdown(
                        "<div style='background:#1A1A1D;border:1px solid %s;border-radius:6px;"
                        "padding:8px 12px;margin:6px 0'>"
                        "<div style='font-size:0.58rem;color:#A1A1A6;letter-spacing:2px;margin-bottom:4px'>FIBONACCI CONFLUENCE</div>"
                        "<div style='font-size:0.82rem;font-weight:700;color:%s'>🔶 %s Retracement</div>"
                        "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:2px'>"
                        "Level: $%.2f &nbsp;·&nbsp; Range: $%.2f — $%.2f</div>"
                        "</div>" % (_fib_color, _fib_color, _fib_level, _fib_price, _fib_low, _fib_high),
                        unsafe_allow_html=True
                    )

                if exh:
                    st.markdown("<div style='font-size:0.58rem;color:#A1A1A6;letter-spacing:2px;margin:6px 0 4px'>EXHAUSTION DETAIL</div>", unsafe_allow_html=True)
                    for reason in exh:
                        good = any(x in reason for x in ["confirmed","forming","Higher low","Lower high","Climax","Capitulation","Hammer","doji","star","reclaim","holding","rising","falling"])
                        col  = "#D4AF37" if good else "#C1121F"
                        tcol = "#F5F5F5" if good else "#A1A1A6"
                        dot  = "●" if good else "○"
                        st.markdown("<div style='font-size:0.71rem;color:%s;padding:1px 0'><span style='color:%s'>%s</span> %s</div>" % (tcol, col, dot, reason), unsafe_allow_html=True)
                # Watch button — adds to Watch Queue directly from scan card
                _wkey_scan = "%s_%s" % (r["ticker"], r.get("direction","bullish"))
                _in_queue  = _wkey_scan in st.session_state.get("watch_queue", {})
                if _in_queue:
                    st.markdown(
                        "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:6px;"
                        "padding:8px;text-align:center;font-size:0.75rem;color:#D4AF37'>✅ In Watch Queue</div>",
                        unsafe_allow_html=True
                    )
                else:
                    if st.button("👁 Add to Watch Queue", key="scan_watch_%s_%s_%s" % (bucket, r["ticker"], idx), use_container_width=True):
                        add_to_watch_queue(r["ticker"], r.get("direction","bullish"), r.get("sig", r), r.get("opt", {}))
                        st.success("Added to Watch Queue!")

                # Admin only — Send to Telegram button
                _is_admin = (
                    st.session_state.get("is_admin", False) or
                    st.session_state.get("user_email", "") == ADMIN_EMAIL
                )
                if _is_admin and bucket == "go_now":
                    _tg_key = "tg_sent_%s_%s_%s" % (r["ticker"], r.get("style",""), idx)
                    if st.session_state.get(_tg_key):
                        st.markdown(
                            "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:6px;"
                            "padding:6px;text-align:center;font-size:0.72rem;color:#D4AF37'>✅ Sent to Telegram</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("📣 Send to Telegram", key="tg_%s_%s_%s" % (bucket, r["ticker"], idx), use_container_width=True):
                            try:
                                send_telegram_alert(r, alert_type="GO NOW")
                                st.session_state[_tg_key] = True
                                st.success("✅ Alert sent!")
                            except Exception as _te:
                                st.error("Telegram error: %s" % str(_te)[:60])

        def section_hdr(label, color, count):
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:20px 0 8px'>
                <div style='width:3px;height:16px;background:{color};border-radius:2px;flex-shrink:0'></div>
                <span style='font-size:0.65rem;letter-spacing:3px;color:{color};font-weight:700'>{label}</span>
                <div style='flex:1;height:1px;background:#2A2A2D'></div>
                <span style='font-size:0.62rem;color:#A1A1A6'>{count} signal{"s" if count!=1 else ""}</span>
            </div>""", unsafe_allow_html=True)

        def empty_bkt(msg):
            st.markdown(f"<div style='padding:14px;color:#A1A1A6;font-size:0.78rem;background:#1A1A1D;border-radius:10px;text-align:center'>{msg}</div>", unsafe_allow_html=True)

        section_hdr("GO NOW", "#22C55E", len(go_now))
        if go_now:
            for i, r in enumerate(go_now[:15]):  mobile_card(r, "go_now",   i)
        else:
            empty_bkt("No GO NOW signals - exhaustion not confirmed or gates not cleared.")

        section_hdr("WATCHING", "#D4AF37", len(watching))
        if watching:
            for i, r in enumerate(watching[:15]): mobile_card(r, "watching", i)
        else:
            empty_bkt("No setups in confirmation phase right now.")

        section_hdr("ON DECK", "#C1121F", len(on_deck))
        if on_deck:
            for i, r in enumerate(on_deck[:10]): mobile_card(r, "on_deck",  i)
        else:
            empty_bkt("No developing setups found.")

with tab8:
    st.markdown("<div class='section-title'>WATCH QUEUE</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#A1A1A6;font-size:0.82rem;margin-bottom:12px'>Signals waiting for entry confirmation. Auto-checks on every refresh.</div>", unsafe_allow_html=True)

    init_watch_queue()
    _wq = st.session_state.watch_queue

    if not _wq:
        st.markdown("<div style='color:#4a5568;text-align:center;padding:40px;font-size:0.9rem'>No signals in queue. Add from the SIGNALS tab.</div>", unsafe_allow_html=True)
    else:
        for _wkey, item in list(_wq.items()):
            elapsed_total = (datetime.now() - item["added_at"]).total_seconds()
            elapsed_mins  = int(elapsed_total / 60)
            elapsed_secs  = int(elapsed_total % 60)

            # Timeout based on trade style
            style        = item.get("style", "swing")
            timeout_mins = 30 if style == "quick" else 240  # 30min quick, 4hr swing
            remaining    = max(0, timeout_mins * 60 - elapsed_total)
            remain_mins  = int(remaining / 60)
            remain_secs  = int(remaining % 60)
            is_expired   = remaining <= 0

            # Auto-remove expired signals
            if is_expired:
                remove_from_watch_queue(_wkey)
                continue

            last_chk = ""
            if item["last_checked"]:
                secs_ago = int((datetime.now() - item["last_checked"]).total_seconds())
                last_chk = " | checked %ds ago" % secs_ago

            candle_html = ""
            for c in item.get("candles", []):
                if c == "green":   candle_html += "<span style='color:#D4AF37;font-size:1rem'>&#9650;</span> "
                elif c == "red":   candle_html += "<span style='color:#C1121F;font-size:1rem'>&#9660;</span> "
                else:              candle_html += "<span style='color:#A1A1A6;font-size:0.8rem'>&#9644;</span> "

            status = item["status"]
            is_bull_w  = item["direction"] == "bullish"
            dir_color_w = "#D4AF37" if is_bull_w else "#C1121F"
            action_w    = "CALL" if is_bull_w else "PUT"

            wq_col, dismiss_col = st.columns([6,1])
            with wq_col:
                if status == "CONFIRMED":
                    st.markdown("""
                    <div style='background:#1A1500;border:2px solid #D4AF37;border-radius:10px;padding:14px 16px;margin:4px 0'>
                        <div style='color:#D4AF37;font-family:monospace;font-size:0.72rem;letter-spacing:2px'>✅ ENTRY CONFIRMED - GET IN NOW</div>
                        <div style='font-size:1.1rem;font-weight:700;color:{dc}'>BUY {act} - {tk}</div>
                        <div style='color:#A1A1A6;font-size:0.82rem'>{pat}</div>
                        <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-size:0.85rem;margin-top:10px'>
                            <div><div style='color:#A1A1A6;font-size:0.7rem'>STRIKE</div><div style='font-weight:700;color:{dc}'>${stk:.2f}</div></div>
                            <div><div style='color:#A1A1A6;font-size:0.7rem'>ENTRY</div><div style='font-weight:700'>${ent:.2f}</div></div>
                            <div><div style='color:#A1A1A6;font-size:0.7rem'>TARGET</div><div style='font-weight:700;color:#D4AF37'>${tgt:.2f}</div></div>
                            <div><div style='color:#A1A1A6;font-size:0.7rem'>STOP</div><div style='font-weight:700;color:#C1121F'>${stp:.2f}</div></div>
                        </div>
                        <div style='margin-top:8px;color:#F5F5F5;font-size:0.78rem'>Candles: {cnd} &nbsp; <span style='color:#F6E27A'>{remain_mins}m {remain_secs}s remaining{last_chk}</span></div>
                    </div>
                    """.format(
                        dc=dir_color_w, act=action_w, tk=item["ticker"],
                        pat=item["pattern"], stk=item["strike"],
                        ent=item["entry"], tgt=item["target"], stp=item["stop"],
                        cnd=candle_html, elapsed_mins=elapsed_mins,
                        elapsed_secs=elapsed_secs, last_chk=last_chk
                    ), unsafe_allow_html=True)
                else:
                    border_clr = "#F6E27A" if status == "WAITING" else "#C1121F"
                    icon = "👁" if status == "WAITING" else "⏳"
                    st.markdown("""
                    <div style='background:#1A1A1D;border:2px solid {bc};border-radius:8px;padding:12px 16px;margin:4px 0'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <div>
                                <span style='font-size:1.1rem'>{ic}</span>
                                <b style='margin-left:6px;color:{dc}'>{tk} {act}</b>
                                <span style='color:#A1A1A6;font-size:0.82rem;margin-left:8px'>{pat} | Strike ${stk:.2f}</span>
                            </div>
                            <div style='color:#F6E27A;font-size:0.75rem;font-family:monospace'>{rm}m {rs}s left{lc}</div>
                        </div>
                        <div style='margin-top:6px'>{cnd}</div>
                        <div style='color:#F5F5F5;font-size:0.82rem;margin-top:4px'>{msg}</div>
                    </div>
                    """.format(
                        bc=border_clr, ic=icon, dc=dir_color_w,
                        tk=item["ticker"], act=action_w, pat=item["pattern"],
                        stk=item["strike"], em=elapsed_mins, es=elapsed_secs, rm=remain_mins, rs=remain_secs,
                        lc=last_chk, cnd=candle_html, msg=item["message"]
                    ), unsafe_allow_html=True)

            with dismiss_col:
                if st.button("✕", key="wq_dismiss_%s" % _wkey, help="Remove from queue"):
                    remove_from_watch_queue(_wkey)
                    st.rerun()

    # Manual refresh button
    _wq_col1, _wq_col2 = st.columns(2)
    with _wq_col1:
        if st.button("🔄 Refresh Queue", key="wq_refresh", use_container_width=True):
            st.rerun()
    with _wq_col2:
        if st.button("🗑 Clear All", key="wq_clear_all", use_container_width=True):
            st.session_state.watch_queue = {}
            st.session_state.wq_loaded = True  # prevent reload from Supabase
            user_id = st.session_state.get("user_id")
            if user_id:
                try:
                    sb = get_supabase()
                    if sb:
                        sb.table("user_data").update(
                            {"watch_queue": "{}", "updated_at": datetime.now(tz=pytz.UTC).isoformat()}
                        ).eq("user_id", str(user_id)).execute()
                        st.success("Queue cleared!")
                    else:
                        st.error("Supabase unavailable")
                except Exception as e:
                    st.error("Error: %s" % str(e)[:80])
            st.rerun()

with tab7:
    st.markdown("""
<style>
.hiw-section { background:#0B0B0C; border-radius:12px; padding:20px 24px;
               margin-bottom:16px; border-left:3px solid #2A2A2D; }
.hiw-title   { font-size:1.1rem; font-weight:700; color:#F5F5F5;
               margin-bottom:8px; letter-spacing:0.5px; }
.hiw-body    { font-size:0.8rem; color:#A1A1A6; line-height:1.8; }
.hiw-badge   { display:inline-block; padding:2px 10px; border-radius:20px;
               font-size:0.7rem; font-weight:700; margin:2px; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style='text-align:center;padding:16px 0 24px'>
  <div style='font-size:1.5rem;font-weight:700;color:#F5F5F5;letter-spacing:2px'>
    📡 HOW IT WORKS
  </div>
  <div style='font-size:0.75rem;color:#A1A1A6;margin-top:6px'>
    PaidButPressured Options Screener
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='hiw-section'>
  <div class='hiw-title'>🎯 What Is This?</div>
  <div class='hiw-body'>
    PaidButPressured is a real-time options screener built for active traders who want 
    high-conviction setups delivered fast — without the noise.<br><br>
    Our proprietary engine scans the market continuously, filters out weak signals, 
    and surfaces only the setups that meet our strict multi-layer confirmation process. 
    No guesswork. No opinions. Just data-driven setups with clear entry, target, and stop levels.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#D4AF37'>
  <div class='hiw-title'>🚦 Signal Tiers</div>
  <div class='hiw-body'>
    Every signal is scored and placed into one of three tiers:<br><br>
    <span class='hiw-badge' style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E'>🟢 GO NOW</span>
    The highest conviction setups. All confirmation criteria met. Entry is valid right now.<br><br>
    <span class='hiw-badge' style='background:#D4AF3722;color:#D4AF37;border:1px solid #D4AF37'>🟡 WATCHING</span>
    Strong setup, waiting on final confirmation. Worth tracking — entry is close.<br><br>
    <span class='hiw-badge' style='background:#1A1A1D;color:#A1A1A6;border:1px solid #A1A1A6'>📋 ON DECK</span>
    Setup is developing. Not ready yet but worth knowing about.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#9966ff'>
  <div class='hiw-title'>⚙️ The Engine</div>
  <div class='hiw-body'>
    Every signal passes through our multi-layer confirmation system before it reaches you.<br><br>
    We evaluate each setup across multiple independent dimensions — momentum, structure, 
    timing, market context, and more. A signal must satisfy a minimum number of these 
    dimensions to qualify for each tier.<br><br>
    The result is a <b style='color:#F5F5F5'>Confidence Score</b> from 0–100% and a 
    <b style='color:#F5F5F5'>Gate Count</b> showing how many checks the setup passed. 
    Higher confidence and more gates = stronger signal.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#F6E27A'>
  <div class='hiw-title'>📊 Reading a Signal Card</div>
  <div class='hiw-body'>
    Each signal card gives you everything you need to place the trade:<br><br>
    <b style='color:#F5F5F5'>Entry</b> — the stock price where the setup is valid<br>
    <b style='color:#F5F5F5'>Target</b> — our measured price objective<br>
    <b style='color:#F5F5F5'>Stop</b> — where the thesis is invalidated, exit if breached<br>
    <b style='color:#F5F5F5'>Strike</b> — the recommended options strike price<br>
    <b style='color:#F5F5F5'>Premium</b> — estimated cost per share to enter<br>
    <b style='color:#F5F5F5'>R:R</b> — risk to reward ratio on the trade<br>
    <b style='color:#F5F5F5'>Confidence %</b> — our internal conviction score<br>
    <b style='color:#F5F5F5'>Gates</b> — how many confirmation layers this signal passed
  </div>
</div>

<div class='hiw-section' style='border-left-color:#D4AF37'>
  <div class='hiw-title'>🎯 How to Trade From the Cards</div>
  <div class='hiw-body'>
    Not all signals are equal. Here's how to prioritize:<br><br>
    <b style='color:#22C55E'>Tier 1 — Highest Conviction:</b><br>
    GO NOW + Entry CONFIRMED + Daily trend agrees + Fibonacci Confluence + Gates 5/7+<br>
    This is your best trade of the day. Act on it.<br><br>
    <b style='color:#D4AF37'>Tier 2 — Solid Setup:</b><br>
    GO NOW or WATCHING + Entry CONFIRMED + 4/7 gates + Exhaustion confirmed<br>
    Good trade. Standard size. Don't chase if price has moved past entry.<br><br>
    <b style='color:#C1121F'>Tier 3 — Wait:</b><br>
    Entry still WAITING, counter-trend, gates 3/7 or lower<br>
    Add to Watch Queue. Never enter on WAITING.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#22C55E'>
  <div class='hiw-title'>📋 The Rules</div>
  <div class='hiw-body'>
    <b style='color:#F5F5F5'>1. Never enter on WAITING</b> — the entry timing check exists for a reason. Wait for CONFIRMED.<br><br>
    <b style='color:#F5F5F5'>2. Never fight the daily trend</b> — if it says "Daily trend conflicts" cut your size in half minimum.<br><br>
    <b style='color:#F5F5F5'>3. Check the move required</b> — LIKELY is your sweet spot. AMBITIOUS is a lottery ticket.<br><br>
    <b style='color:#F5F5F5'>4. Respect the stop</b> — when price hits your stop level, exit. No hoping, no holding.<br><br>
    <b style='color:#F5F5F5'>5. Quick trades close same day</b> — never hold a quick trade overnight.<br><br>
    <b style='color:#F5F5F5'>6. Fibonacci + pattern = priority</b> — when both show confirmed, that's your trade of the day.<br><br>
    <b style='color:#D4AF37'>The one sentence rule:</b><br>
    GO NOW + CONFIRMED entry + daily trend agrees + Fibonacci showing = enter.<br>
    Everything else = wait or skip.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#D4AF37'>
  <div class='hiw-title'>📡 Market Regime Engine</div>
  <div class='hiw-body'>
    Every scan runs a 5-layer market analysis that classifies current conditions and adjusts signal confidence accordingly.<br><br>
    <b style='color:#F5F5F5'>The Regime Banner</b> appears after every scan showing:<br>
    Regime label · Breadth score (% CALLS vs % PUTS) · 5-day and 20-day trend · RSI reading · Rally authenticity<br><br>
    <b style='color:#F5F5F5'>The 8 Regime Types:</b><br><br>
    <span style='color:#22C55E'>🟢 BULL CONFIRMED</span> — Broad participation, healthy volume. CALL signals elevated.<br><br>
    <span style='color:#C1121F'>🔴 BEAR CONFIRMED</span> — Sustained downtrend with broad participation. PUT signals elevated.<br><br>
    <span style='color:#C1121F'>⚠️ BULL TRAP</span> — Rally is suspect. Volume weak, breadth bearish. CALL signals blocked or penalized. Watch for reversal.<br><br>
    <span style='color:#F6E27A'>⚡ BEAR TRAP</span> — Short-term bounce in downtrend. Oversold relief rally. Treat with caution.<br><br>
    <span style='color:#F6E27A'>📉 DISTRIBUTION</span> — Index healthy long-term but short-term weakness. Smart money may be selling into strength.<br><br>
    <span style='color:#D4AF37'>💀 CAPITULATION</span> — Extreme fear. Oversold conditions. Potential reversal zone — watch for CALL setups at key support.<br><br>
    <span style='color:#F6E27A'>🔍 SUSPECT RALLY</span> — Rally showing weakness signals. Proceed with extra caution on CALL signals.<br><br>
    <span style='color:#A1A1A6'>↔️ CHOPPY</span> — No clear directional edge. Reduce size. Wait for clarity before entering.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#22C55E'>
  <div class='hiw-title'>🏷️ Signal Alignment Badges</div>
  <div class='hiw-body'>
    Every signal card shows a regime alignment badge telling you whether the signal works with or against current market conditions:<br><br>
    <span style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E44;padding:2px 8px;border-radius:4px;font-size:0.8rem'>✅ REGIME</span> &nbsp; Signal direction matches the current regime. Higher conviction — this is what you want.<br><br>
    <span style='background:#C1121F22;color:#C1121F;border:1px solid #C1121F44;padding:2px 8px;border-radius:4px;font-size:0.8rem'>⚠️ COUNTER</span> &nbsp; Signal is fighting the regime. Confidence is reduced by 10%. Extra caution required — smaller size if you take it.<br><br>
    <span style='background:#C1121F44;color:#C1121F;border:1px solid #C1121F;padding:2px 8px;border-radius:4px;font-size:0.8rem'>🚫 BLOCKED</span> &nbsp; Regime actively conflicts with this signal. Confidence reduced by 20%. BULL TRAP + CALL signal = blocked. Avoid these.<br><br>
    <b style='color:#D4AF37'>The rule:</b> Always prioritize ✅ REGIME signals. Only take ⚠️ COUNTER signals if everything else is perfect — Fibonacci confirmed, 6/7+ gates, daily trend agrees. Never take 🚫 BLOCKED signals.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#C1121F'>
  <div class='hiw-title'>⚠️ Risk Disclosure</div>
  <div class='hiw-body'>
    Options trading involves substantial risk of loss and is not appropriate for all investors. 
    Past performance of signals does not guarantee future results.<br><br>
    PaidButPressured is an educational and informational tool only. 
    Nothing on this platform constitutes financial advice, investment advice, 
    or a recommendation to buy or sell any security.<br><br>
    Always paper trade new setups before using real capital. 
    Never risk more than you can afford to lose. 
    You are solely responsible for your own trading decisions.
  </div>
</div>

<div class='hiw-section' style='border-left-color:#D4AF37'>
  <div class='hiw-title'>📱 Best Practices</div>
  <div class='hiw-body'>
    <b style='color:#F5F5F5'>Scan by sector</b> — run sector scans throughout the day instead of full universe every time. Focus on what's in play.<br><br>
    <b style='color:#F5F5F5'>Prioritize GO NOW</b> — these are your highest conviction setups. WATCHING signals need one more confirmation before entry.<br><br>
    <b style='color:#F5F5F5'>Use the Watch Queue</b> — add WATCHING signals to your queue and let the screener track entry confirmation for you.<br><br>
    <b style='color:#F5F5F5'>Respect your stop</b> — the stop level is calculated for a reason. Honor it every time.<br><br>
    <b style='color:#F5F5F5'>Check Telegram</b> — admin-curated GO NOW alerts are posted manually after review. These are the setups worth acting on.
  </div>
</div>
""", unsafe_allow_html=True)
