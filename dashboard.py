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

from pattern_detection import (
    detect_double_bottom, detect_double_top, detect_break_and_retest,
    detect_vwap_reclaim, detect_bull_bear_flag,
    detect_opening_range_breakout, detect_momentum_continuation,
    detect_ascending_descending_triangle, detect_head_and_shoulders,
)
from backtester import run_backtest

st.set_page_config(page_title="PaidButPressured", page_icon="📡", layout="centered", initial_sidebar_state="expanded")

# Supabase puts access_token in the URL hash which Streamlit can't read.
# This JS detects it and converts it to a query param so Python can handle it.
st.components.v1.html("<div></div>", height=0)

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

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_UID          = os.environ.get("ADMIN_UID", "158a9910")
ADMIN_EMAIL        = os.environ.get("ADMIN_EMAIL", "")
MAKE_WEBHOOK_URL   = os.environ.get("MAKE_WEBHOOK_URL", "https://hook.us2.make.com/k4yp47rg33vdinypxzb3tl7ch6j4u229")
POLYGON_API_KEY    = os.environ.get("POLYGON_API_KEY", "")
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")
FMP_API_KEY        = os.environ.get("FMP_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
APP_PASSWORD       = os.environ.get("APP_PASSWORD", "")
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY         = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

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

def check_auth():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.session_state.tos_agreed    = True
        st.session_state.authenticated = True
        st.session_state.user_email    = "dev@local"
        st.session_state.user_id       = "dev"
        return

    # Supabase sends access_token + type=recovery or type=invite in the URL hash
    # Streamlit doesn't expose hash params directly — check query params for token
    _qp = st.query_params
    _token_type  = _qp.get("type", "")
    _access_tok  = _qp.get("access_token", "")
    _token_hash  = _qp.get("token_hash", "")

    if _token_type in ("recovery", "invite", "signup") and (_access_tok or _token_hash):
        st.markdown("""
<div style='max-width:400px;margin:60px auto;padding:32px 36px;
background:#1A1A1D;border:1px solid #2A2A2D;border-radius:16px;text-align:center'>
<div style='font-size:1.4rem;font-weight:700;color:#F5F5F5;letter-spacing:2px;margin-bottom:4px'>
📡 PAIDBUTPRESSURED</div>
<div style='font-size:0.75rem;color:#A1A1A6;margin-bottom:24px'>Set Your Password</div>
</div>""", unsafe_allow_html=True)
        col = st.columns([1,4,1])[1]
        with col:
            new_pw  = st.text_input("New Password", type="password", placeholder="Choose a password (min 6 chars)")
            new_pw2 = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            if st.button("SET PASSWORD & LOG IN", use_container_width=True):
                if len(new_pw) < 6:
                    st.error("Password must be at least 6 characters")
                elif new_pw != new_pw2:
                    st.error("Passwords don't match")
                else:
                    try:
                        import requests as _req
                        from supabase import create_client
                        sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
                        _at = _access_tok

                        if not _at:
                            st.error("Reset link missing token. Please request a new one.")
                        else:
                            _pw_resp = _req.put(
                                "%s/auth/v1/user" % SUPABASE_URL,
                                json={"password": new_pw},
                                headers={
                                    "apikey":        SUPABASE_KEY,
                                    "Authorization": "Bearer %s" % _at,
                                    "Content-Type":  "application/json",
                                },
                                timeout=10,
                            )
                            if _pw_resp.status_code == 200:
                                _udata = _pw_resp.json()
                                _email = _udata.get("email", "")
                                _uid   = _udata.get("id", "")
                                _resp2 = sb.auth.sign_in_with_password(
                                    {"email": _email, "password": new_pw}
                                )
                                _user2 = getattr(_resp2, "user", None)
                                _sess2 = getattr(_resp2, "session", None)
                                if _user2:
                                    st.session_state.authenticated       = True
                                    st.session_state.tos_agreed          = True
                                    st.session_state.user_email          = _email
                                    st.session_state.user_id             = _uid
                                    st.session_state.is_admin            = (_email == ADMIN_EMAIL)
                                    st.session_state.watchlist_loaded    = False
                                    st.session_state._access_token       = getattr(_sess2, "access_token", "") if _sess2 else ""
                                    st.session_state._refresh_token      = getattr(_sess2, "refresh_token", "") if _sess2 else ""
                                    st.session_state._last_token_refresh = datetime.now()
                                    st.query_params.clear()
                                    st.success("Password set! Logging you in...")
                                    st.rerun()
                                else:
                                    st.warning("Password updated. Please log in manually.")
                            elif _pw_resp.status_code == 401:
                                st.error("Reset link expired. Please request a new one.")
                            else:
                                st.error("Error updating password (%s). Request a new reset link." % _pw_resp.status_code)
                    except Exception as e:
                        st.error("Error: %s" % str(e)[:150])
        st.stop()
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
        _mode = st.radio("Account Mode", ["Sign In", "Create Account"],
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

            with st.expander("Forgot your password?"):
                fp_email = st.text_input("Enter your email", key="fp_email",
                                          placeholder="your@email.com")
                if st.button("Send Reset Link", key="fp_btn", use_container_width=True):
                    if not fp_email:
                        st.error("Enter your email address")
                    else:
                        try:
                            from supabase import create_client
                            sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
                            _redirect = SUPABASE_URL.replace("https://", "https://options-screener-production.up.railway.app?type=recovery&")
                            sb.auth.reset_password_for_email(
                                fp_email,
                                {"redirect_to": "https://options-screener-production.up.railway.app"}
                            )
                            st.success("Reset link sent! Check your inbox.")
                        except Exception as e:
                            st.error("Error: %s" % str(e)[:100])

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
    An options screener built around ONE strategy: the
    <b style='color:#D4AF37'>Opening Range Breakout</b>.<br><br>
    Every morning, the first 15 minutes sets a high and a low. When price breaks
    that range and retests it, that's the trade — and this tool tracks all of it
    for you across your whole list.<br><br>
    <b style='color:#D4AF37'>Let's walk through it.</b>
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
    Open the <b style='color:#C1121F'>🔴 ORB tab</b> and pick your scope —
    <b style='color:#D4AF37'>My Watchlist</b>, a <b style='color:#D4AF37'>Sector</b>,
    or the <b style='color:#D4AF37'>Full Universe</b>.<br><br>
    Hit <b style='color:#F5F5F5'>RUN ORB SCAN</b> after 9:45 ET once the range is set.
    Want just one name? Type it in the search box for the full read — even before
    it breaks.<br><br>
    Scan for the setup, then confirm it on your own chart.
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
  <div class='ob-title'>Reading the Buckets 🚦</div>
  <div class='ob-body'>
    Every setup lands in one of three buckets:<br><br>
    <span class='ob-badge' style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E'>
      🟢 GO NOW
    </span>
    The retest held. The entry is live right now.<br><br>
    <span class='ob-badge' style='background:#D4AF3722;color:#D4AF37;border:1px solid #D4AF37'>
      🟡 WATCHING
    </span>
    Broke, but hasn't retested yet — or volume was weak. Wait.<br><br>
    <span class='ob-badge' style='background:#7AA2F722;color:#7AA2F7;border:1px solid #7AA2F7'>
      🔵 ON DECK
    </span>
    Range is set, price is near a boundary. Nothing triggered yet.
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
  <div class='ob-title'>The Retest is the Entry 🎯</div>
  <div class='ob-body'>
    The #1 rule: <b style='color:#F5F5F5'>don't chase the break — wait for the
    retest.</b> Price breaks out, pulls back to test the level, and if it holds,
    THAT's your entry with a tight stop.<br><br>
    Each GO NOW card lays out the whole trade — <b style='color:#22C55E'>entry</b>,
    <b style='color:#C1121F'>stop</b>, <b style='color:#D4AF37'>target</b>, plus a
    suggested strike and expiration.<br><br>
    The <b style='color:#C1121F'>⚡ Momentum tab</b> catches early moves before the
    range sets — powerful, but UNCONFIRMED and can reverse. Size small there.<br><br>
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
.dot-green  { width: 8px; height: 8px; background: #00C853; border-radius: 50%; display: inline-block; flex-shrink: 0; box-shadow: 0 0 6px rgba(0,200,83,0.7); }
.dot-red    { width: 8px; height: 8px; background: #FF1744; border-radius: 50%; display: inline-block; flex-shrink: 0; box-shadow: 0 0 6px rgba(255,23,68,0.6); }
.dot-yellow { width: 8px; height: 8px; background: #FFD600; border-radius: 50%; display: inline-block; flex-shrink: 0; box-shadow: 0 0 6px rgba(255,214,0,0.6); }
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
    "APP","AXON","MSTR","NBIS","ZETA","AAOI","CBRS","SPCX","SNDK","NTRA","BE",
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
    # Small/Mid Cap AI & Tech
    "BBAI","SOUN","SERV","JOBY","ACHR","AMBA","CRSR","TOST","BRZE","GTLB",
    "FROG","NCNO","JAMF","ALKT","RELY","CWAN","LSPD","TASK",
    # Small/Mid Fintech
    "NU","DAVE","MQ","UPST","LC","OPFI","NRDS","COOP","STEP",
    # Small/Mid Biotech
    "RXRX","ALNY","IONS","ACAD","HALO","ITCI","IMVT","VKTX","NUVL","ROIV",
    "KROS","ADMA","SAGE",
    # Crypto Mining
    "MARA","CLSK","RIOT","HUT","CORZ","BTDR",
    # Small/Mid Energy
    "CIVI","SM","CHRD","NOG","ESTE",
    # Consumer Small/Mid
    "RH","BOOT","BURL","FIVE","OLLI","CVNA","KMX","AN","PAG",
    # Defense & Space
    "RKLB","LUNR","KTOS","AVAV","SPCE",
    # Media & Entertainment
    "WBD","PARA","FUBO","NWSA",
    # Growth Mid Cap
    "DUOL","ELF","SKIN","GNRC",
    # ETF sectors
    "XLK","XLF","XLE","XLV","XLY","XLI","GLD","SLV","TLT","HYG","IBIT",
    # Additional liquid movers
    "OPEN","NKLA","WKHS","FUBO","WBD","PARA","GNRC","BOOT","PAG","AN",
    "CHRD","NOG","ESTE","VKTX","ROIV","NRDS","COOP","STEP","SAGE","ADMA",
    "SM","CIVI","HII","CACI","LDOS","SAIC","NWSA","ELF","DUOL","SKIN","BTDR","CORZ",
]
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))  # deduplicate

SECTOR_LISTS = {
    "My Watchlist":     [],  # populated from session state at runtime
    "Tech & Semis":     ["NVDA","AMD","INTC","AVGO","QCOM","MU","AMAT","LRCX","KLAC","MRVL",
                         "ARM","TSM","PLTR","SNOW","DDOG","NET","CRWD","ZS","PANW","FTNT",
                         "OKTA","S","APP","AXON","WDC","SMCI","CBRS","SNDK"],
    "Mega Cap":         ["AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA","NVDA","BRK-B","SPY","QQQ","IWM","DIA"],
    "Financials":       ["JPM","BAC","GS","MS","C","WFC","BLK","V","MA","PYPL","AXP","SOFI","AFRM","HOOD","XYZ","COIN"],
    "Healthcare":       ["UNH","JNJ","PFE","MRNA","BNTX","ABBV","LLY","BMY","GILD","REGN","BIIB"],
    "Energy & Power":   ["XOM","CVX","OXY","SLB","HAL","MPC","PSX","VST","CEG","GEV"],
    "Consumer":         ["WMT","TGT","COST","HD","LOW","NKE","LULU","MCD","SBUX","CMG","DKNG","CELH","HIMS"],
    "High Momentum":    ["PLTR","TSLA","COIN","MSTR","ASTS","NVDA","AMD","HOOD","SOFI","AFRM",
                         "RIVN","RDW","IREN","NBIS","ZETA","AAOI","CRDO","LCID","GRAB","SE",
                         "BBAI","SOUN","MARA","RIOT","CLSK","UPST","RKLB","LUNR","NU","CVNA","CBRS","SPCX","SNDK","BE"],
    "Small/Mid AI & Tech": ["BBAI","SOUN","SERV","JOBY","ACHR","AMBA","CRSR","TOST","BRZE","GTLB",
                         "FROG","NCNO","JAMF","ALKT","RELY","CWAN","LSPD","TASK","NU","DAVE",
                         "MQ","UPST","LC","OPFI","NRDS","COOP","STEP","DUOL","SKIN","ELF"],
    "Biotech":          ["RXRX","ALNY","IONS","ACAD","HALO","ITCI","IMVT","VKTX","NUVL","ROIV",
                         "KROS","ADMA","SAGE","MRNA","BNTX","GILD","REGN","BIIB","PFE","NTRA"],
    "Crypto & Mining":  ["MARA","CLSK","RIOT","HUT","CORZ","BTDR","COIN","MSTR","IBIT"],
    "Defense & Space":  ["RKLB","LUNR","KTOS","AVAV","SPCE","LMT","RTX","BA","GE","AXON"],
    "Affordable Movers": ["BBAI","SOUN","SERV","JOBY","ACHR","AMBA","CRSR","TOST","MARA","CLSK",
                         "RIOT","HUT","CORZ","CIVI","SM","NOG","BOOT","OLLI","CVNA","RKLB",
                         "LUNR","KTOS","WBD","PARA","FUBO","UPST","NU","DAVE","RXRX","ACAD",
                         "SOFI","AFRM","HOOD","RIVN","LCID","ASTS","IREN","ZETA","NBIS","AAOI"],
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
    "CBRS":"XLK","SNDK":"XLK","BE":"XLE",
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

# Doing so fires "missing ScriptRunContext" warnings and can cause instability.
# Functions in the scan worker path use _thread_cache instead - pure Python,
# no Streamlit context required, thread-safe via a single lock.
import time as _time_mod
_THREAD_CACHE      = {}
_THREAD_CACHE_LOCK = _threading.Lock()

# ────────────────────────────────────────────────────────────────────────
# SHARED HTTP SESSION — bounded connection pool prevents thread/socket leaks
# under concurrent load. Single Session reused by every API call.
# ────────────────────────────────────────────────────────────────────────
_HTTP_SESSION = None
_HTTP_SESSION_LOCK = _threading.Lock()

def _get_http_session():
    """Return a process-wide HTTP session with connection pooling and retries.
    Caps concurrent socket count at 50 regardless of user load."""
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    with _HTTP_SESSION_LOCK:
        if _HTTP_SESSION is not None:
            return _HTTP_SESSION
        try:
            import requests as _r
            from requests.adapters import HTTPAdapter
            try:
                from urllib3.util.retry import Retry
                retry = Retry(
                    total=2, backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"],
                )
            except Exception:
                retry = None
            adapter = HTTPAdapter(
                pool_connections=20,    # number of connection pools
                pool_maxsize=50,        # max sockets per pool
                pool_block=False,       # don't block, recycle instead
                max_retries=retry if retry else 2,
            )
            sess = _r.Session()
            sess.mount("https://", adapter)
            sess.mount("http://",  adapter)
            sess.headers.update({"User-Agent": "PBP/1.0"})
            _HTTP_SESSION = sess
            return sess
        except Exception as _se:
            print("[http_session] init error: %s" % str(_se)[:120])
            return None

class _StubResponse:
    """Fake response returned when an HTTP call fails entirely.
    status_code=0 so existing `if r.status_code != 200` checks naturally fail."""
    def __init__(self):
        self.status_code = 0
        self.text = ""
        self.content = b""
    def json(self):
        return {}
    def raise_for_status(self):
        pass

def _http_get(url, timeout=8, **kwargs):
    """Drop-in replacement for requests.get() using a shared pooled session.
    Always returns an object (real response or stub with status_code=0).
    Hard timeout prevents hung sockets from leaking threads."""
    sess = _get_http_session()
    try:
        if sess is not None:
            return sess.get(url, timeout=timeout, **kwargs)
        import requests as _r
        return _r.get(url, timeout=timeout, **kwargs)
    except Exception as _he:
        try:
            print("[http_get] %s on %s" % (type(_he).__name__, url[:80]))
        except Exception:
            pass
        return _StubResponse()


def _thread_cache(ttl=300):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = (fn.__name__,) + args + tuple(sorted(kwargs.items()))
            now = _time_mod.time()
            with _THREAD_CACHE_LOCK:
                entry = _THREAD_CACHE.get(key)
                if entry and (now - entry[0]) < ttl:
                    return entry[1]
            result = fn(*args, **kwargs)
            with _THREAD_CACHE_LOCK:
                _THREAD_CACHE[key] = (now, result)
            return result
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

# yfinance v0.2.x returns MultiIndex columns for single-ticker downloads.
# e.g. df["close"] returns a DataFrame with shape (n,1) instead of a Series.

def _col(df, name):
    """Return column `name` from df as a guaranteed 1D float Series."""
    c = df[name]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.astype(float)

def _clean_df(df):
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
        r = _http_get(url, timeout=4)
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
            r = _http_get(url, timeout=5)
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
        r = _http_get(
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
            "60d":60,"90d":90,"1mo":30,"3mo":90,"6mo":180,"1y":365,"18mo":545,"2y":760,
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
            r = _http_get(url, timeout=8)
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
            r = _http_get(url, timeout=8)
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
        r = _http_get(url, timeout=8)
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
        r = _http_get(
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
            r = _http_get(
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

@_thread_cache(ttl=3600)
def fetch_200ma(ticker):
    try:
        df = _fmp_download(ticker, "1y", "1d")
        if df is None or len(df) < 20:
            return None, None, None, None
        close      = df["close"].astype(float)
        ma_period  = min(200, len(close))
        ma         = close.rolling(ma_period).mean().dropna()
        if ma.empty:
            return None, None, None, None
        current_ma    = float(ma.iloc[-1])
        prior_ma      = float(ma.iloc[-6]) if len(ma) >= 6 else current_ma
        current_price = float(close.iloc[-1])
        above_ma      = current_price > current_ma
        slope_rising  = current_ma > prior_ma
        pct_from_ma   = round((current_price - current_ma) / current_ma * 100, 2) if current_ma > 0 else 0
        return above_ma, round(current_ma, 2), slope_rising, pct_from_ma
    except Exception:
        return None, None, None, None

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
    if not FMP_API_KEY or not ticker:
        return None
    try:
        import requests as _req
        url = (
            "https://financialmodelingprep.com/api/v3/options/%s"
            "?apikey=%s" % (ticker.upper(), FMP_API_KEY)
        )
        r = _http_get(url, timeout=6)
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

    iv_adj  = min(iv * 1.3, 0.80) if actual_dte <= 7 else iv
    # ATM option premium approximation: price * IV * sqrt(DTE/365) * ~0.4 for ATM
    otm_discount = 0.38 if trade_style == "quick" else 0.22
    premium = round(current_price * iv_adj * (max(actual_dte, 1)/365)**0.5 * otm_discount, 2)
    premium = max(premium, 0.05)
    breakeven = (strike + premium) if is_call else (strike - premium)

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
    move_pct = round((move_needed / current_price) * 100, 1) if current_price > 0 else 0.0

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

    # This aligns the "Est Profit" display with what the exit rules actually say
    option_gain_per_share = premium * 1.0  # 100% gain = 2x premium
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
    elif cur_vol > avg_vol * 1.2:
        vol_ok    = True
        vol_label = "Volume: Confirming (%.1fx avg)" % (cur_vol / avg_vol if avg_vol > 0 else 0)
    else:
        vol_ok    = False
        vol_label = "Volume: Insufficient (%.1fx avg — need 1.2x+)" % (cur_vol / avg_vol if avg_vol > 0 else 0)

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

def check_entry_confirmation(df, direction):
    if len(df) < 4:
        return {"confirmed": False, "status": "WAITING", "candles": [], "message": "Not enough data"}

    recent  = df.tail(5)
    is_bull = direction == "bullish"

    candle_dirs = []
    for _, row in recent.iterrows():
        o, c = float(row["open"]), float(row["close"])
        body  = abs(c - o)
        rng   = float(row.get("high", c)) - float(row.get("low", o))
        # Only count as directional if body is at least 30% of the range
        if rng > 0 and body / rng >= 0.3:
            candle_dirs.append("green" if c > o else "red")
        else:
            candle_dirs.append("doji")

    c1 = recent.iloc[-2]
    c2 = recent.iloc[-1]
    c1_close, c1_open = float(c1["close"]), float(c1["open"])
    c2_close, c2_open = float(c2["close"]), float(c2["open"])

    last_dir   = candle_dirs[-1]
    second_dir = candle_dirs[-2]

    if is_bull:
        # Strong confirm: last 2 both green
        strong  = last_dir == "green" and second_dir == "green" and c2_close > c1_close
        # Soft confirm: last candle green with meaningful close above prior close
        soft    = last_dir == "green" and c2_close > c1_close
        # Against: last candle clearly red closing below prior close
        against = last_dir == "red" and c2_close < c1_close

        if strong:
            status  = "CONFIRMED"
            message = "2 bullish candles confirmed — buyers in control. Entry window open near $%.2f" % c2_close
            confirmed = True
        elif soft:
            status  = "CONFIRMED"
            message = "Bullish candle confirmed — price closing higher. Entry near $%.2f" % c2_close
            confirmed = True
        elif against:
            status  = "AGAINST"
            message  = "Price dropping. Signal valid but wait for a green close before entering."
            confirmed = False
        else:
            status  = "WAITING"
            message  = "Neutral candle. Watching for directional close in your favor."
            confirmed = False
    else:
        strong  = last_dir == "red" and second_dir == "red" and c2_close < c1_close
        soft    = last_dir == "red" and c2_close < c1_close
        against = last_dir == "green" and c2_close > c1_close

        if strong:
            status  = "CONFIRMED"
            message = "2 bearish candles confirmed — sellers in control. Entry window open near $%.2f" % c2_close
            confirmed = True
        elif soft:
            status  = "CONFIRMED"
            message = "Bearish candle confirmed — price closing lower. Entry near $%.2f" % c2_close
            confirmed = True
        elif against:
            status  = "AGAINST"
            message  = "Price climbing. Signal valid but wait for a red close before entering."
            confirmed = False
        else:
            status  = "WAITING"
            message  = "Neutral candle. Watching for directional close in your favor."
            confirmed = False

    return {"confirmed": confirmed, "status": status, "candles": candle_dirs, "message": message}


def analyze_breakout_state(ticker, direction, sr_resistance, sr_support):
    """
    Two-layer S/R-aware breakout monitor.
    Returns dict with state and message for Watch Queue display.

    States:
      WAITING_BREAKOUT    — price approaching key level
      MOMENTUM_ENTRY      — strong break, vol confirmed, enter now
      WEAK_BREAK          — thin break, optional caution entry or wait for retest
      RETEST_READY        — pullback to level holding, cleanest entry
      RETEST_FAILED       — retest failed, signal invalid (auto-remove)
      NO_LEVELS           — no S/R context, fall back to candle logic
    """
    is_bull = direction == "bullish"
    if is_bull:
        key_level = sr_resistance  # CALL needs to break above resistance
    else:
        key_level = sr_support     # PUT needs to break below support

    if not key_level or key_level <= 0:
        return {"state": "NO_LEVELS", "message": "", "key_level": None, "current_price": None}

    try:
        df_1m = _fmp_download(ticker, "1d", "1m")
        if df_1m is None or len(df_1m) < 5:
            return {"state": "NO_LEVELS", "message": "1m data unavailable", "key_level": key_level, "current_price": None}

        df_1m = df_1m.sort_values("datetime").reset_index(drop=True) if "datetime" in df_1m.columns else df_1m
        last_5  = df_1m.tail(5)
        current = float(df_1m.iloc[-1]["close"])
        last_candle  = df_1m.iloc[-1]
        prior_candle = df_1m.iloc[-2] if len(df_1m) >= 2 else last_candle

        # Volume context — last candle vs avg of last 20
        avg_vol_20 = float(df_1m.tail(20)["volume"].mean()) if len(df_1m) >= 20 else 1
        last_vol   = float(last_candle.get("volume", 0))
        vol_mult   = last_vol / max(avg_vol_20, 1)

        # Body strength — body as % of total range
        c_open  = float(last_candle["open"])
        c_close = float(last_candle["close"])
        c_high  = float(last_candle.get("high", c_close))
        c_low   = float(last_candle.get("low",  c_open))
        body    = abs(c_close - c_open)
        rng     = max(c_high - c_low, 0.001)
        body_pct = body / rng

        # Distance from key level
        dist_pct = abs(current - key_level) / key_level * 100

        if is_bull:
            broken_above = current > key_level and c_close > key_level
            still_below  = current < key_level
            broke_back   = c_close < key_level and float(prior_candle["close"]) > key_level

            if still_below and dist_pct < 0.5:
                return {
                    "state":         "WAITING_BREAKOUT",
                    "message":       "Approaching $%.2f resistance — watching 1-min for break above" % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }
            if broken_above:
                # Strong break: vol ≥ 1.5x and body ≥ 50% of range
                if vol_mult >= 1.5 and body_pct >= 0.50:
                    return {
                        "state":         "MOMENTUM_ENTRY",
                        "message":       "Strong break above $%.2f — volume %.1fx, momentum confirmed. Enter now." % (key_level, vol_mult),
                        "key_level":     key_level,
                        "current_price": current,
                        "vol_mult":      round(vol_mult, 2),
                    }
                # Weak break: low vol or thin body
                else:
                    return {
                        "state":         "WEAK_BREAK",
                        "message":       "Breakout printed above $%.2f but volume thin (%.1fx). Enter with caution at own risk, or wait for pullback to $%.2f. Loss of $%.2f on 1-min close = invalid." % (key_level, vol_mult, key_level, key_level),
                        "key_level":     key_level,
                        "current_price": current,
                        "vol_mult":      round(vol_mult, 2),
                    }
            if broke_back:
                return {
                    "state":         "RETEST_FAILED",
                    "message":       "Failed retest — closed back below $%.2f. Setup invalid." % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }
            # Already above and holding on retest
            if current >= key_level * 0.998 and current <= key_level * 1.005 and c_close > key_level:
                return {
                    "state":         "RETEST_READY",
                    "message":       "Retest of $%.2f holding — clean entry zone now." % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }
        else:
            broken_below = current < key_level and c_close < key_level
            still_above  = current > key_level
            broke_back   = c_close > key_level and float(prior_candle["close"]) < key_level

            if still_above and dist_pct < 0.5:
                return {
                    "state":         "WAITING_BREAKOUT",
                    "message":       "Approaching $%.2f support — watching 1-min for break below" % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }
            if broken_below:
                if vol_mult >= 1.5 and body_pct >= 0.50:
                    return {
                        "state":         "MOMENTUM_ENTRY",
                        "message":       "Strong break below $%.2f — volume %.1fx, momentum confirmed. Enter now." % (key_level, vol_mult),
                        "key_level":     key_level,
                        "current_price": current,
                        "vol_mult":      round(vol_mult, 2),
                    }
                else:
                    return {
                        "state":         "WEAK_BREAK",
                        "message":       "Breakdown printed below $%.2f but volume thin (%.1fx). Enter with caution at own risk, or wait for retest to $%.2f. Reclaim of $%.2f on 1-min close = invalid." % (key_level, vol_mult, key_level, key_level),
                        "key_level":     key_level,
                        "current_price": current,
                        "vol_mult":      round(vol_mult, 2),
                    }
            if broke_back:
                return {
                    "state":         "RETEST_FAILED",
                    "message":       "Failed retest — reclaimed $%.2f. Setup invalid." % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }
            if current <= key_level * 1.002 and current >= key_level * 0.995 and c_close < key_level:
                return {
                    "state":         "RETEST_READY",
                    "message":       "Retest of $%.2f holding below — clean entry zone now." % key_level,
                    "key_level":     key_level,
                    "current_price": current,
                }

        return {
            "state":         "WAITING_BREAKOUT",
            "message":       "Watching $%.2f — price not yet at decision point" % key_level,
            "key_level":     key_level,
            "current_price": current,
        }
    except Exception as _e:
        return {"state": "NO_LEVELS", "message": "Breakout check error: %s" % str(_e)[:60], "key_level": key_level, "current_price": None}

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


def run_background_watch_checks(tf_mult, tf_span, tf_days):
    """
    Two-layer S/R-aware breakout monitor.
    5-min: setup health (is signal still valid)
    1-min: actual breakout/retest execution trigger via analyze_breakout_state()
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
            ticker        = item["ticker"]
            direction     = item["direction"]
            sr_resistance = item.get("sr_resistance")
            sr_support    = item.get("sr_support")

            # ── LAYER 1: S/R-aware breakout analysis (1-min) ─────────────
            br = analyze_breakout_state(ticker, direction, sr_resistance, sr_support)

            # If we have S/R levels, use breakout-state logic
            if br["state"] != "NO_LEVELS" and br.get("key_level"):
                prev_state = item.get("breakout_state", "WAITING")
                item["breakout_state"] = br["state"]
                item["key_level"]      = br.get("key_level")
                item["current_price"]  = br.get("current_price")
                item["vol_mult"]       = br.get("vol_mult")
                item["message"]        = br["message"]
                item["last_checked"]   = datetime.now()

                # Auto-remove failed retests
                if br["state"] == "RETEST_FAILED":
                    to_remove.append(key)
                    continue

                # Map breakout state to legacy status for display
                if br["state"] in ("MOMENTUM_ENTRY", "RETEST_READY"):
                    item["status"] = "CONFIRMED"
                    if not item["alerted"]:
                        item["alerted"] = True
                        any_new_confirm = True
                elif br["state"] == "WEAK_BREAK":
                    item["status"] = "WEAK_BREAK"
                else:
                    item["status"] = "WAITING"

                # Still pull candle direction for display
                try:
                    df_1m_disp = _fmp_download(ticker, "1d", "1m")
                    if df_1m_disp is not None and len(df_1m_disp) >= 5:
                        df_1m_disp = df_1m_disp.tail(5)
                        candle_dirs = []
                        for _, row in df_1m_disp.iterrows():
                            o, c = float(row["open"]), float(row["close"])
                            body = abs(c - o)
                            rng  = float(row.get("high", c)) - float(row.get("low", o))
                            if rng > 0 and body / rng >= 0.3:
                                candle_dirs.append("green" if c > o else "red")
                            else:
                                candle_dirs.append("doji")
                        item["candles"] = candle_dirs
                except Exception:
                    pass

                continue  # done with this item

            # ── LAYER 2: Fallback — original candle-confirmation logic ───
            style = item.get("style", "swing")
            interval, period = ("5m", "2d") if style == "quick" else ("1h", "14d")

            raw = _fmp_download(ticker, period, interval)
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                raw = _yf_download(ticker, period=period, interval=interval)
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

            conf = check_entry_confirmation(fresh_df, direction)
            was_confirmed_before = item["status"] == "CONFIRMED"
            item["status"]         = conf["status"]
            item["message"]        = conf["message"]
            item["candles"]        = conf.get("candles", [])
            item["last_checked"]   = datetime.now()
            item["breakout_state"] = "NO_LEVELS"

            if conf["confirmed"] and not was_confirmed_before and not item["alerted"]:
                item["alerted"]   = True
                any_new_confirm   = True
        except Exception as e:
            import traceback
            err_msg = str(e)[:80]
            item["message"] = "Check error: %s" % err_msg
            print("[WATCH QUEUE ERROR] %s: %s" % (item.get("ticker", "?"), traceback.format_exc()))

    for key in to_remove:
        if key in queue:
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
def check_liquidity(ticker):
    if not POLYGON_API_KEY:
        return True, 0, 0, "Verify OI manually"
    try:
        import requests as _req
        # Get options contracts for this ticker - nearest expiry, calls
        url = (
            "https://api.polygon.io/v3/snapshot/options/%s"
            "?limit=25&apiKey=%s" % (ticker.upper(), POLYGON_API_KEY)
        )
        r = _http_get(url, timeout=4)
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
    vol_expanding = cur_vol > avg_vol * 1.2  # consistent 1.2x threshold
    vol_present   = cur_vol > avg_vol * 1.0  # below 1.2x = not confirmed

    factors = {
        "Pattern":{"pass":True,          "label":"Pattern confirmed"},
        "RSI Div":{"pass":rsi_div_match, "label":"Price Divergence: Confirmed" if rsi_div_match else "Price Divergence: Not Detected"},
        "Volume": {"pass":vol_expanding, "label":"Volume: Confirming (1.2x+ required)" if vol_expanding else "Volume: Insufficient (below 1.2x avg)"},
        "EMA":    {"pass":(price>ema20 if is_bull else price<ema20),"label":"Trend Filter: Aligned" if (price>ema20 if is_bull else price<ema20) else "Trend Filter: Against"},
        "VWAP":   {"pass":(price>vwap  if is_bull else price<vwap), "label":"Intraday Bias: Aligned" if (price>vwap if is_bull else price<vwap) else "Intraday Bias: Against"},
    }

    # Base score: each factor = 10pts, max 50
    raw_score  = sum(1 for f in factors.values() if f["pass"])
    base_score = raw_score * 10  # 0-50

    return factors, raw_score, base_score, rsi, vwap, ema20

def calc_quick_levels(price, direction, atr):
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

def build_candidates(df, ticker, toggles, account, risk_pct, dte, trade_style="swing", atr=None):
    trend_dir,trend_score,trend_factors,t_ema,t_vwap,t_rsi = get_trend(df)
    _raw_price = df["close"].iloc[-1]
    price = float(_raw_price.iloc[0] if hasattr(_raw_price, "iloc") else _raw_price)
    regime, regime_strength = detect_market_regime(df)
    is_quick = trade_style == "quick"
    regime_bonus = 5 if regime == "trending" else -5
    candidates = []
    raw = []
    if is_quick:
        if toggles.get("br"):   raw += [s for s in detect_break_and_retest(df, ticker, rr_min=1.5) if s.confirmed]
        if toggles.get("vwap"): raw += [s for s in detect_vwap_reclaim(df, ticker, rr_min=1.5) if s.confirmed]
        if toggles.get("flag"): raw += [s for s in detect_bull_bear_flag(df, ticker, rr_min=1.5, trade_style="quick") if s.confirmed]
        if toggles.get("orb"):  raw += [s for s in detect_opening_range_breakout(df, ticker, rr_min=1.5) if s.confirmed]
        if toggles.get("mom"):  raw += [s for s in detect_momentum_continuation(df, ticker, rr_min=1.5) if s.confirmed]
    else:
        if toggles.get("db"):   raw += [s for s in detect_double_bottom(df, ticker, rr_min=2.0) if s.confirmed]
        if toggles.get("dt"):   raw += [s for s in detect_double_top(df, ticker, rr_min=2.0) if s.confirmed]
        if toggles.get("br"):   raw += [s for s in detect_break_and_retest(df, ticker, rr_min=2.0) if s.confirmed]
        if toggles.get("flag"): raw += [s for s in detect_bull_bear_flag(df, ticker, rr_min=2.0, trade_style="swing") if s.confirmed]
        if toggles.get("tri"):  raw += [s for s in detect_ascending_descending_triangle(df, ticker, rr_min=2.0) if s.confirmed]
        if toggles.get("hs"):   raw += [s for s in detect_head_and_shoulders(df, ticker, rr_min=2.0) if s.confirmed]

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

def get_ai_brief(ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status):
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

# PRECISION SCAN ENGINE

@_thread_cache(ttl=300)
def get_market_internals():
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


# MARKET REGIME DETECTION ENGINE
# Layer 1: Breadth Calculator
# Layer 2: Index Health Check  
# Layer 3: Rally Authenticity
# Layer 4: Regime Classifier
# Layer 5: Signal Adjuster

def calculate_breadth_score(go_now, watching, on_deck):
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
            iv_rank, _ = fetch_iv_rank(ticker)
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
            conf_adj  = 0  # flag shown on card — user decides, no score penalty

        # Special rules per regime
        # BULL TRAP: warn but don't hard block — a stock at support with
        # volume confirmation is still a valid call even in a bull trap
        if regime == "BULL TRAP" and direction == "bullish":
            alignment = "COUNTER"
            conf_adj  = -8  # soft penalty, card shows warning, trader decides

        if regime == "CAPITULATION" and direction == "bullish":
            conf_adj  = +8  # bounce plays in capitulation

        if regime == "CHOPPY":
            conf_adj  = -5  # reduce confidence in choppy market

        r["regime_alignment"] = alignment
        r["confidence"]       = min(97, max(30, conf + conf_adj))
        adjusted.append(r)

    return adjusted

def detect_fibonacci_confluence(df, direction, current_price=None):
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



@_thread_cache(ttl=1800)
def detect_sr_levels(ticker, current_price, direction):
    empty = {
        "at_support": False, "at_resistance": False,
        "nearest_support": None, "nearest_resistance": None,
        "support_levels": [], "resistance_levels": [],
        "conf_boost": 0, "label": "S/R Unavailable", "detail": "",
    }
    if not current_price or current_price <= 0:
        return empty
    try:
        df = _fmp_download(ticker, "1y", "1d")
        if df is None or len(df) < 20:
            return empty

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        tol  = current_price * 0.006   # 0.6% tolerance to be "at" a level
        sups = []
        ress = []

        lb = min(len(df), 60)
        hs = high.iloc[-lb:].reset_index(drop=True)
        ls = low.iloc[-lb:].reset_index(drop=True)

        for i in range(2, len(hs) - 2):
            h = float(hs.iloc[i])
            if h > float(hs.iloc[i-1]) and h > float(hs.iloc[i-2])                and h > float(hs.iloc[i+1]) and h > float(hs.iloc[i+2]):
                ress.append(h)

        for i in range(2, len(ls) - 2):
            l = float(ls.iloc[i])
            if l < float(ls.iloc[i-1]) and l < float(ls.iloc[i-2])                and l < float(ls.iloc[i+1]) and l < float(ls.iloc[i+2]):
                sups.append(l)

        periods = min(252, len(close))
        wk52h   = float(high.iloc[-periods:].max())
        wk52l   = float(low.iloc[-periods:].min())
        ress.append(wk52h)
        sups.append(wk52l)

        if current_price < 20:     incr = 1
        elif current_price < 50:   incr = 5
        elif current_price < 200:  incr = 10
        elif current_price < 500:  incr = 25
        else:                      incr = 50

        base = round(current_price / incr) * incr
        for mult in range(-6, 7):
            level = base + mult * incr
            if level > 0:
                if level < current_price - tol:
                    sups.append(float(level))
                elif level > current_price + tol:
                    ress.append(float(level))

        def cluster(levels, thresh=0.015):
            if not levels:
                return []
            levels = sorted(set(round(l, 2) for l in levels if l > 0))
            out    = []
            grp    = [levels[0]]
            for l in levels[1:]:
                if abs(l - grp[-1]) / grp[-1] < thresh:
                    grp.append(l)
                else:
                    out.append(round(sum(grp) / len(grp), 2))
                    grp = [l]
            out.append(round(sum(grp) / len(grp), 2))
            return out

        sups = cluster([s for s in sups if s < current_price])
        ress = cluster([r for r in ress if r > current_price])

        nearest_sup = round(max(sups), 2) if sups else None
        nearest_res = round(min(ress), 2) if ress else None

        at_sup = nearest_sup is not None and abs(current_price - nearest_sup) <= tol * 3
        at_res = nearest_res is not None and abs(current_price - nearest_res) <= tol * 3

        if direction == "bullish":
            if at_sup:
                boost  = 8
                label  = "Price at Support \u2705"
                detail = "Buying into $%.2f support \u2014 favorable risk/reward" % nearest_sup
            elif at_res:
                boost  = -6
                label  = "Price at Resistance \u26a0\ufe0f"
                detail = "Buying into $%.2f resistance \u2014 CALL needs clean breakout first" % nearest_res
            elif nearest_sup is not None:
                dist = (current_price - nearest_sup) / current_price * 100
                if dist < 3.0:
                    boost  = 4
                    label  = "Near Support (%.1f%% away)" % dist
                    detail = "Support at $%.2f below \u2014 good cushion on stop" % nearest_sup
                else:
                    boost  = 0
                    label  = "Between S/R Levels"
                    detail = "Support $%.2f \u00b7 Resistance %s" % (
                        nearest_sup, "$%.2f" % nearest_res if nearest_res else "N/A"
                    )
            else:
                boost  = 0
                label  = "No Key S/R Nearby"
                detail = "No clear support/resistance within range"
        else:  # bearish
            if at_res:
                boost  = 8
                label  = "Price at Resistance \u2705"
                detail = "Selling from $%.2f resistance \u2014 favorable risk/reward" % nearest_res
            elif at_sup:
                boost  = -6
                label  = "Price at Support \u26a0\ufe0f"
                detail = "Shorting into $%.2f support \u2014 PUT needs clean breakdown first" % nearest_sup
            elif nearest_res is not None:
                dist = (nearest_res - current_price) / current_price * 100
                if dist < 3.0:
                    boost  = 4
                    label  = "Near Resistance (%.1f%% away)" % dist
                    detail = "Resistance at $%.2f above \u2014 natural ceiling for stock" % nearest_res
                else:
                    boost  = 0
                    label  = "Between S/R Levels"
                    detail = "Support %s \u00b7 Resistance $%.2f" % (
                        "$%.2f" % nearest_sup if nearest_sup else "N/A", nearest_res
                    )
            else:
                boost  = 0
                label  = "No Key S/R Nearby"
                detail = "No clear support/resistance within range"

        return {
            "at_support":         at_sup,
            "at_resistance":      at_res,
            "nearest_support":    nearest_sup,
            "nearest_resistance": nearest_res,
            "support_levels":     sups[-5:] if sups else [],
            "resistance_levels":  ress[:5]  if ress else [],
            "conf_boost":         boost,
            "label":              label,
            "detail":             detail,
        }
    except Exception as _e:
        empty["label"]  = "S/R Error"
        empty["detail"] = str(_e)[:50]
        return empty
def detect_exhaustion(df, direction):
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




# Commodity ETF map for energy/materials names
_COMMODITY_MAP = {
    "XOM": "USO", "CVX": "USO", "OXY": "USO", "SLB": "USO", "HAL": "USO",
    "MPC": "USO", "PSX": "USO", "COP": "USO", "EOG": "USO", "DVN": "USO",
    "GLD": "GLD", "SLV": "SLV", "FCX": "CPER", "NEM": "GLD",
    "VST": "XLE", "CEG": "XLE", "GEV": "XLE",
}

@_thread_cache(ttl=300)
def check_sector_etf_trend(sector_etf):
    try:
        df = _fmp_download(sector_etf, "30d", "1d")
        if df is None or len(df) < 22:
            return True, True, "Sector data unavailable"
        close = df["close"].astype(float)
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])
        above = price > ema20
        pct   = round((price - ema20) / ema20 * 100, 1)
        if above:
            detail = "%s above 20 EMA by %.1f%% — sector bullish" % (sector_etf, pct)
        else:
            detail = "%s below 20 EMA by %.1f%% — sector bearish" % (sector_etf, abs(pct))
        return above, not above, detail
    except Exception:
        return True, True, "Sector check unavailable"

@_thread_cache(ttl=300)
def check_relative_strength(ticker, sector_etf, days=10):
    try:
        tk_df  = _fmp_download(ticker,     "20d", "1d")
        sec_df = _fmp_download(sector_etf, "20d", "1d")
        if tk_df is None or sec_df is None or len(tk_df) < days + 1 or len(sec_df) < days + 1:
            return True, True, "RS data unavailable"
        tk_ret  = (float(tk_df["close"].iloc[-1])  - float(tk_df["close"].iloc[-days]))  / float(tk_df["close"].iloc[-days])  * 100
        sec_ret = (float(sec_df["close"].iloc[-1]) - float(sec_df["close"].iloc[-days])) / float(sec_df["close"].iloc[-days]) * 100
        outperforms = tk_ret > sec_ret
        diff = round(tk_ret - sec_ret, 1)
        detail = "%s %s%s%% vs %s %s%s%% (%s by %.1f%%)" % (
            ticker, "+" if tk_ret >= 0 else "", round(tk_ret, 1),
            sector_etf, "+" if sec_ret >= 0 else "", round(sec_ret, 1),
            "outperforming" if outperforms else "underperforming", abs(diff)
        )
        return outperforms, not outperforms, detail
    except Exception:
        return True, True, "RS check unavailable"

@_thread_cache(ttl=300)
def check_momentum_filter(ticker, days_short=5, days_long=10):
    try:
        df = _fmp_download(ticker, "20d", "1d")
        if df is None or len(df) < days_long + 1:
            return True, True, "Momentum data unavailable"
        close    = df["close"].astype(float)
        ret_5d   = round((float(close.iloc[-1]) - float(close.iloc[-days_short]))  / float(close.iloc[-days_short])  * 100, 1)
        ret_10d  = round((float(close.iloc[-1]) - float(close.iloc[-days_long]))   / float(close.iloc[-days_long])   * 100, 1)
        bull_ok  = ret_5d > 0 and ret_10d > 0
        bear_ok  = ret_5d < 0 and ret_10d < 0
        detail   = "5D: %s%.1f%% | 10D: %s%.1f%%" % (
            "+" if ret_5d >= 0 else "", ret_5d,
            "+" if ret_10d >= 0 else "", ret_10d
        )
        return bull_ok, bear_ok, detail
    except Exception:
        return True, True, "Momentum check unavailable"

@_thread_cache(ttl=600)
def check_commodity_trend(ticker, direction):
    commodity_etf = _COMMODITY_MAP.get(ticker.upper())
    if not commodity_etf:
        return True, "No commodity dependency"
    try:
        df = _fmp_download(commodity_etf, "20d", "1d")
        if df is None or len(df) < 22:
            return True, "Commodity data unavailable"
        close = df["close"].astype(float)
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])
        above = price > ema20
        aligned = (direction == "bullish" and above) or (direction == "bearish" and not above)
        trend   = "bullish" if above else "bearish"
        detail  = "%s (%s) is %s — %s for %s signal" % (
            commodity_etf, ticker, trend,
            "aligned" if aligned else "OPPOSING", direction
        )
        return aligned, detail
    except Exception:
        return True, "Commodity check unavailable"


# PBP NEWS SENTIMENT ENGINE v2
# Philosophy: News is directional data, not a blocker.
# Only trading halt and delisted hard-block — everything else is PUT/CALL intel.

_BEAR_KEYWORDS = {
    "fraud": 3, "sec investigation": 3, "criminal": 3, "bankruptcy": 3,
    "defaulted": 3, "fda rejection": 3, "recall": 3,
    "class action": 3, "indicted": 3, "arrested": 3,
    "accounting irregularity": 3, "restatement": 3,
    "going concern": 3, "ceo resigned": 3, "ceo fired": 3,
    "downgrade": 2, "guidance cut": 2, "layoffs": 2, "job cuts": 2,
    "revenue decline": 2, "loss widens": 2, "margin compression": 2,
    "tariff": 2, "sanction": 2, "probe": 2, "investigation": 2,
    "lawsuit": 2, "subpoena": 2, "ousted": 2,
    "lower guidance": 2, "below expectations": 2, "disappoints": 2,
    "concern": 1, "warning": 1, "headwind": 1, "pressure": 1,
    "decline": 1, "fell": 1, "dropped": 1, "slumped": 1,
    "missed": 1, "weak": 1, "disappointing": 1,
}

_BULL_KEYWORDS = {
    "fda approval": 3, "fda approved": 3, "buyout": 3, "acquisition": 3,
    "takeover bid": 3, "merger agreement": 3, "record revenue": 3,
    "record earnings": 3, "blowout quarter": 3,
    "major contract": 3, "government contract": 3,
    "upgrade": 2, "raised guidance": 2, "guidance raised": 2,
    "above expectations": 2, "strong demand": 2, "revenue growth": 2,
    "margin expansion": 2, "new product launch": 2, "partnership": 2,
    "buyback": 2, "dividend increase": 2, "special dividend": 2,
    "earnings beat": 2, "exceeds estimates": 2,
    "growth": 1, "surged": 1, "jumped": 1, "soared": 1,
    "strong": 1, "solid results": 1, "momentum": 1,
    "expansion": 1, "increased": 1,
}

_UNTRADEABLE_KEYWORDS = [
    "trading halt", "trading halted", "halt trading",
    "delisted", "delisting", "exchange delisted",
    "suspended from trading", "trading suspended",
]

_MACRO_BEAR_TRIGGERS = [
    "tariff", "trade war", "sanctions", "rate hike",
    "recession", "inflation surge", "banking crisis",
    "market crash", "circuit breaker",
]

_MACRO_BULL_TRIGGERS = [
    "rate cut", "fed pivot", "stimulus", "trade deal",
    "ceasefire", "peace deal", "deregulation", "tax cut",
]



_SOURCE_WEIGHT = {
    "reuters.com": 3.0, "bloomberg.com": 3.0, "wsj.com": 3.0,
    "ft.com": 3.0, "cnbc.com": 2.5, "apnews.com": 2.5,
    "marketwatch.com": 2.0, "barrons.com": 2.0,
    "businesswire.com": 2.0, "prnewswire.com": 2.0, "sec.gov": 3.0,
    "fool.com": 1.5, "thestreet.com": 1.5, "investors.com": 1.5,
    "benzinga.com": 1.0, "zacks.com": 1.0,
    "seekingalpha.com": 0.8,
    # Finnhub source name formats (no .com)
    "reuters": 3.0, "bloomberg": 3.0,
    "cnbc": 2.5, "marketwatch": 2.0, "barrons": 2.0,
    "benzinga": 1.0, "seekingalpha": 0.8,
    "yahoo": 1.0, "yahoo finance": 1.0,
}

def get_source_weight(site):
    site = (site or "").lower().strip()
    for domain, weight in _SOURCE_WEIGHT.items():
        if domain in site:
            return weight
    return 1.0

def calculate_news_velocity(articles, window_minutes=30):
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = 0
    for article in articles:
        pub = article.get("publishedDate", "")
        try:
            dt = datetime.strptime(pub[:19], "%Y-%m-%d %H:%M:%S")
            if dt >= cutoff:
                recent += 1
        except Exception:
            pass
    is_breaking    = recent >= 3
    velocity_score = min(100, recent * 25)
    return velocity_score, is_breaking, recent

# PBP SNIPER ENGINE
# 1-minute candle analysis → micro trigger → execution score → entry zone
# Transforms setup intelligence into execution intelligence

@_thread_cache(ttl=60)
def fetch_1min(ticker, bars=30):
    if not FMP_API_KEY:
        return None
    try:
        import requests as _req
        url = (
            "https://financialmodelingprep.com/stable/historical-chart/1min"
            "?symbol=%s&apikey=%s" % (ticker.upper(), FMP_API_KEY)
        )
        r = _http_get(url, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data[:bars])  # most recent bars first
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)  # oldest first
        required = ["datetime", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].dropna().reset_index(drop=True)
    except Exception:
        return None


def detect_micro_trigger(df_1min, direction):
    if df_1min is None or len(df_1min) < 5:
        return "NO TRIGGER", False, "not active", "Insufficient 1min data"

    try:
        close  = df_1min["close"].astype(float)
        high   = df_1min["high"].astype(float)
        low    = df_1min["low"].astype(float)
        vol    = df_1min["volume"].astype(float)
        op     = df_1min["open"].astype(float)

        # VWAP on 1min bars
        tp       = (high + low + close) / 3
        vwap_1m  = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1]) if vol.sum() > 0 else float(close.iloc[-1])

        c1 = float(close.iloc[-1])   # current
        c2 = float(close.iloc[-2])   # 1 bar ago
        c3 = float(close.iloc[-3])   # 2 bars ago
        h1 = float(high.iloc[-1])
        l1 = float(low.iloc[-1])
        o1 = float(op.iloc[-1])
        v1 = float(vol.iloc[-1])
        v_avg = float(vol.iloc[-10:].mean()) if len(vol) >= 10 else float(vol.mean())

        body_1    = abs(c1 - o1)
        range_1   = h1 - l1
        body_pct  = body_1 / range_1 if range_1 > 0 else 0
        vol_spike = v1 > v_avg * 1.3

        # Recent high/low for breakout detection
        recent_high = float(high.iloc[-6:-1].max()) if len(high) >= 6 else float(high.iloc[:-1].max())
        recent_low  = float(low.iloc[-6:-1].min())  if len(low)  >= 6 else float(low.iloc[:-1].min())

        is_bull = direction == "bullish"

        if is_bull:
            just_reclaimed = c2 < vwap_1m and c1 > vwap_1m
            if just_reclaimed:
                active = vol_spike or body_pct > 0.5
                window = "1–2 candles" if active else "closing"
                return "VWAP RECLAIM", active, window, "Price crossed above 1min VWAP — buyers taking control"
        else:
            just_rejected = c2 > vwap_1m and c1 < vwap_1m
            if just_rejected:
                active = vol_spike or body_pct > 0.5
                window = "1–2 candles" if active else "closing"
                return "VWAP RECLAIM", active, window, "Price crossed below 1min VWAP — sellers taking control"

        if is_bull:
            near_vwap  = abs(c1 - vwap_1m) / vwap_1m < 0.003  # within 0.3%
            holding_up = c1 > c2 and c1 > vwap_1m
            if near_vwap and holding_up:
                return "PULLBACK HOLD", True, "1–2 candles", "Price pulled back to VWAP and holding — buyers stepping in"
            # EMA8 hold
            ema8 = float(close.ewm(span=8).mean().iloc[-1])
            if abs(c1 - ema8) / ema8 < 0.003 and c1 > ema8 and c2 > c3:
                return "PULLBACK HOLD", True, "1–2 candles", "Price holding EMA8 on 1min — momentum intact"
        else:
            near_vwap  = abs(c1 - vwap_1m) / vwap_1m < 0.003
            holding_dn = c1 < c2 and c1 < vwap_1m
            if near_vwap and holding_dn:
                return "PULLBACK HOLD", True, "1–2 candles", "Price pulled back to VWAP and rejecting — sellers in control"

        if is_bull:
            breakout = c1 > recent_high and vol_spike and body_pct > 0.4
            if breakout:
                return "MOMENTUM BREAK", True, "1–2 candles", "Breaking recent 1min high with volume — momentum entry"
        else:
            breakdown = c1 < recent_low and vol_spike and body_pct > 0.4
            if breakdown:
                return "MOMENTUM BREAK", True, "1–2 candles", "Breaking recent 1min low with volume — momentum entry"

        upper_wick = h1 - max(c1, o1)
        lower_wick = min(c1, o1) - l1
        if is_bull and upper_wick > body_1 * 2 and upper_wick > range_1 * 0.4:
            return "REJECTION WICK", False, "not active", "Long upper wick on 1min — sellers rejecting higher prices"
        if not is_bull and lower_wick > body_1 * 2 and lower_wick > range_1 * 0.4:
            return "REJECTION WICK", False, "not active", "Long lower wick on 1min — buyers rejecting lower prices"

        # Check if trigger expired (price has moved far from VWAP)
        extension = abs(c1 - vwap_1m) / vwap_1m
        if extension > 0.008:  # more than 0.8% from VWAP
            if (is_bull and c1 > vwap_1m) or (not is_bull and c1 < vwap_1m):
                return "EXTENDED", False, "expired", "Price extended from VWAP — wait for pullback before entering"

        return "NO TRIGGER", False, "not active", "No clear 1min entry trigger — monitor for setup development"

    except Exception as e:
        return "NO TRIGGER", False, "not active", "Trigger detection error"


def calc_execution_score(trigger_type, trigger_active, vol_confirmed,
                          entry_status, direction, df_1min, current_price, atr):
    score = 0

    # Trigger present and active (+30)
    if trigger_active:
        score += 30
    elif trigger_type not in ("NO TRIGGER", "EXTENDED", "REJECTION WICK"):
        score += 10  # trigger exists but not fully active yet

    # Volume on current 1min bar (+20)
    if vol_confirmed:
        score += 20

    # Not extended — price in entry zone (+20)
    if trigger_type != "EXTENDED" and entry_status != "LATE — DO NOT CHASE":
        score += 20

    # Momentum on 1min — last 3 closes in signal direction (+15)
    try:
        if df_1min is not None and len(df_1min) >= 3:
            closes = df_1min["close"].astype(float).iloc[-3:].tolist()
            if direction == "bullish":
                if closes[-1] > closes[-2] > closes[-3]:
                    score += 15
                elif closes[-1] > closes[-2]:
                    score += 8
            else:
                if closes[-1] < closes[-2] < closes[-3]:
                    score += 15
                elif closes[-1] < closes[-2]:
                    score += 8
    except Exception:
        pass

    # Trigger type quality (+15)
    trigger_quality = {
        "VWAP RECLAIM":   15,
        "PULLBACK HOLD":  12,
        "MOMENTUM BREAK": 10,
        "REJECTION WICK":  0,
        "EXTENDED":        0,
        "NO TRIGGER":      0,
    }
    score += trigger_quality.get(trigger_type, 0)

    return min(100, max(0, score))


def build_entry_zone(entry_price, direction, pattern_label, atr, trigger_type):
    if atr is None or atr <= 0:
        atr = entry_price * 0.01  # fallback: 1% of price

    # Entry type from pattern
    breakout_patterns = ["Break & Retest", "Opening Range Breakout", "Bull Flag", "Bear Flag", "Momentum Continuation"]
    pullback_patterns = ["VWAP Reclaim", "VWAP Rejection", "Double Bottom", "Double Top"]
    retest_patterns   = ["Break & Retest", "Head & Shoulders", "Inverse H&S", "Ascending Triangle", "Descending Triangle"]

    if pattern_label in breakout_patterns or trigger_type == "MOMENTUM BREAK":
        entry_type = "BREAKOUT"
        zone_low   = round(entry_price, 2)
        zone_high  = round(entry_price + (0.15 * atr), 2)
    elif pattern_label in pullback_patterns or trigger_type == "PULLBACK HOLD":
        entry_type = "PULLBACK"
        zone_low   = round(entry_price - (0.1 * atr), 2)
        zone_high  = round(entry_price + (0.2 * atr), 2)
    elif trigger_type == "VWAP RECLAIM":
        entry_type = "RECLAIM"
        zone_low   = round(entry_price - (0.05 * atr), 2)
        zone_high  = round(entry_price + (0.15 * atr), 2)
    else:
        entry_type = "RETEST"
        zone_low   = round(entry_price - (0.1 * atr), 2)
        zone_high  = round(entry_price + (0.1 * atr), 2)

    # Execution script
    if direction == "bullish":
        script = (
            "Enter between $%.2f – $%.2f on push\n"
            "If price breaks above $%.2f → wait for pullback\n"
            "If price loses $%.2f → setup invalid"
        ) % (zone_low, zone_high, zone_high, zone_low)
        missed = "Wait for pullback to $%.2f – $%.2f" % (zone_low, round(zone_low + (zone_high - zone_low) * 0.5, 2))
    else:
        script = (
            "Enter between $%.2f – $%.2f on rejection\n"
            "If price drops below $%.2f → momentum entry allowed\n"
            "If price breaks above $%.2f → setup invalid"
        ) % (zone_low, zone_high, zone_low, zone_high)
        missed = "Wait for bounce to $%.2f – $%.2f" % (round(zone_low + (zone_high - zone_low) * 0.5, 2), zone_high)

    return zone_low, zone_high, entry_type, script, missed


def get_sniper_entry_status(trigger_type, trigger_active, current_price,
                             zone_low, zone_high, execution_score):
    if trigger_type == "EXTENDED":
        return "EXTENDED — DO NOT CHASE", "#FF1744", "🔴"
    if trigger_type == "REJECTION WICK":
        return "WAIT — COUNTER SIGNAL", "#FFD600", "⚠️"
    if trigger_active and zone_low <= current_price <= zone_high:
        return "ENTER NOW", "#00C853", "🟢"
    if trigger_active and execution_score >= 60:
        return "ENTER NOW", "#00C853", "🟢"
    if trigger_type in ("VWAP RECLAIM", "PULLBACK HOLD", "MOMENTUM BREAK") and execution_score >= 40:
        return "WAIT — PULLBACK FORMING", "#FFD600", "🟡"
    if trigger_type == "NO TRIGGER":
        return "NO ENTRY — MONITOR", "#4a5568", "⚪"
    return "WAIT — SETUP DEVELOPING", "#FFD600", "🟡"


def render_sniper_strip_html(ticker, direction, trigger_type, trigger_active,
                              entry_window, trigger_detail, entry_status, entry_status_color,
                              entry_status_emoji, execution_score, zone_low, zone_high,
                              entry_type, execution_script, missed_plan):
    """Renders the sniper strip HTML block for the top of signal cards."""

    # Execution score bar color
    if execution_score >= 75:
        exec_color = "#00C853"
        exec_label = "STRONG"
    elif execution_score >= 50:
        exec_color = "#FFD600"
        exec_label = "MODERATE"
    else:
        exec_color = "#FF1744"
        exec_label = "WEAK"

    trigger_display = {
        "VWAP RECLAIM":   "⚡ VWAP RECLAIM",
        "PULLBACK HOLD":  "⚡ PULLBACK HOLD → GO",
        "MOMENTUM BREAK": "⚡ MOMENTUM BREAK",
        "REJECTION WICK": "⚠️ REJECTION WICK",
        "EXTENDED":       "🚫 PRICE EXTENDED",
        "NO TRIGGER":     "— MONITORING",
    }.get(trigger_type, "— MONITORING")

    script_lines = execution_script.replace("\n", "<br>")

    return (
        "<div style='background:linear-gradient(135deg,#0f0f12,#1a1a1d);"
        "border:1px solid %s;border-radius:10px;padding:14px 16px;margin-bottom:10px'>"

        # Row 1: Entry Status (big, decision-first)
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
        "<div style='font-family:Barlow Condensed,Arial Black,sans-serif;"
        "font-size:1.15rem;font-weight:900;letter-spacing:0.05em;color:%s'>"
        "%s %s</div>"
        "<div style='text-align:right'>"
        "<div style='font-size:0.62rem;color:#A1A1A6;letter-spacing:0.1em'>EXECUTION</div>"
        "<div style='font-size:0.95rem;font-weight:700;color:%s'>%s%%</div>"
        "<div style='font-size:0.58rem;color:%s'>%s</div>"
        "</div></div>"

        # Row 2: Trigger + Window
        "<div style='display:flex;gap:16px;margin-bottom:10px'>"
        "<div style='flex:1'>"
        "<div style='font-size:0.65rem;color:#A1A1A6;letter-spacing:0.1em;margin-bottom:2px'>TRIGGER</div>"
        "<div style='font-size:0.82rem;font-weight:700;color:#F5F5F5'>%s</div>"
        "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:2px'>%s</div>"
        "</div>"
        "<div>"
        "<div style='font-size:0.65rem;color:#A1A1A6;letter-spacing:0.1em;margin-bottom:2px'>WINDOW</div>"
        "<div style='font-size:0.82rem;font-weight:700;color:#F5F5F5'>⏱ %s</div>"
        "</div></div>"

        # Row 3: Entry Zone
        "<div style='background:#111115;border-radius:6px;padding:8px 12px;margin-bottom:8px'>"
        "<div style='font-size:0.62rem;color:#A1A1A6;letter-spacing:0.1em;margin-bottom:4px'>%s ZONE</div>"
        "<div style='font-size:0.95rem;font-weight:700;color:#D4AF37'>"
        "$%.2f – $%.2f</div>"
        "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:4px;line-height:1.5'>%s</div>"
        "</div>"

        # Row 4: Missed entry
        "<div style='font-size:0.68rem;color:#4a5568'>"
        "Missed entry: %s</div>"

        "</div>"
    ) % (
        entry_status_color,
        entry_status_color, entry_status_emoji, entry_status,
        exec_color, execution_score,
        exec_color, exec_label,
        trigger_display, trigger_detail,
        entry_window,
        entry_type,
        zone_low, zone_high, script_lines,
        missed_plan,
    )


# WEEKLY MACRO BIAS ENGINE
# Runs once per week. Cached in Supabase. Served fresh all week.
# Reads ES (SPY proxy), NQ (QQQ proxy), BTC weekly candles from FMP.

def _wbias_fetch_weekly(ticker, limit=3):
    """Fetch weekly proxy data using FMP daily endpoint — filter to weekly manually."""
    if not FMP_API_KEY:
        return []
    try:
        import requests as _req
        # Use daily data and sample weekly — FMP weekly endpoint is unreliable
        url = (
            "https://financialmodelingprep.com/stable/historical-price-eod/full"
            "?symbol=%s&from=%s&apikey=%s"
            % (ticker, (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), FMP_API_KEY)
        )
        r = _http_get(url, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json()
        hist = data.get("historical", data) if isinstance(data, dict) else data
        if not hist:
            return []
        # Return last N daily candles sorted newest first — caller uses [0] as current
        hist_sorted = sorted(hist, key=lambda x: x.get("date",""), reverse=True)
        return hist_sorted[:limit*7]  # enough days to cover N weeks
    except Exception:
        return []

def _wbias_score_asset(ticker_key, fmp_ticker):
    """Score a single asset bullish/bearish/neutral based on recent price action."""
    candles = _wbias_fetch_weekly(fmp_ticker, limit=3)
    if len(candles) < 5:
        return {"asset": ticker_key, "bias": "NEUTRAL", "pct": None}
    try:
        # Compare most recent close vs close 5 days ago
        current_close = float(candles[0]["close"])
        week_ago_close = float(candles[4]["close"])
        pct = (current_close - week_ago_close) / week_ago_close * 100
        if pct >= 0.5:
            bias = "BULLISH"
        elif pct <= -0.5:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        return {"asset": ticker_key, "bias": bias, "pct": round(pct, 2)}
    except Exception:
        return {"asset": ticker_key, "bias": "NEUTRAL", "pct": None}

def _wbias_get_week_start():
    """Return most recent Monday as YYYY-MM-DD string."""
    today = datetime.now().date()
    days_back = today.weekday()  # Monday=0
    return str(today - timedelta(days=days_back))

def _wbias_load_supabase(week_start):
    """Load this week's bias from Supabase cache."""
    try:
        r = _supabase_request(
            "GET",
            "/rest/v1/weekly_bias?week_start=eq.%s&limit=1" % week_start,
        )
        if r and isinstance(r, list) and len(r) > 0:
            return r[0]
    except Exception:
        pass
    return None

def _supabase_request(method, path, payload=None):
    """Generic Supabase REST call reusing existing env vars."""
    import requests as _req
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    url = SUPABASE_URL.rstrip("/") + path
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": "Bearer " + SUPABASE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        if method == "GET":
            r = _http_get(url, headers=headers, timeout=8)
            return r.json() if r.status_code in (200, 201) else None
        elif method == "POST":
            r = _req.post(url, headers={**headers, "Prefer": "resolution=merge-duplicates"}, json=payload, timeout=8)
            return r.status_code in (200, 201, 204)
    except Exception:
        return None

def _wbias_save_supabase(week_start, assets, overall):
    """Save this week's bias to Supabase."""
    row = {
        "week_start":   week_start,
        "es_bias":      assets.get("SPY", {}).get("bias", "NEUTRAL"),
        "nq_bias":      assets.get("QQQ", {}).get("bias", "NEUTRAL"),
        "btc_bias":     assets.get("BTC", {}).get("bias", "NEUTRAL"),
        "es_pct":       assets.get("SPY", {}).get("pct"),
        "nq_pct":       assets.get("QQQ", {}).get("pct"),
        "btc_pct":      assets.get("BTC", {}).get("pct"),
        "overall_bias": overall,
        "created_at":   datetime.utcnow().isoformat(),
    }
    _supabase_request("POST", "/rest/v1/weekly_bias", row)

@_thread_cache(ttl=3600)
def get_weekly_macro_bias():
    week_start = _wbias_get_week_start()

    # Try cache first
    cached = _wbias_load_supabase(week_start)
    if cached:
        return {
            "week_start": cached.get("week_start"),
            "overall":    cached.get("overall_bias", "NEUTRAL"),
            "assets": {
                "SPY": {"bias": cached.get("es_bias", "NEUTRAL"), "pct": cached.get("es_pct")},
                "QQQ": {"bias": cached.get("nq_bias", "NEUTRAL"), "pct": cached.get("nq_pct")},
                "BTC": {"bias": cached.get("btc_bias", "NEUTRAL"), "pct": cached.get("btc_pct")},
            },
            "source": "cache",
        }

    # Fresh fetch — SPY and QQQ as ES/NQ proxies (FMP futures tickers unreliable)
    BIAS_ASSETS = {"SPY": "SPY", "QQQ": "QQQ", "BTC": "BTCUSD"}
    asset_results = {}
    for key, ticker in BIAS_ASSETS.items():
        result = _wbias_score_asset(key, ticker)
        asset_results[key] = result

    # Weighted scoring: SPY 2x, QQQ 2x, BTC 1x
    weights = {"SPY": 2, "QQQ": 2, "BTC": 1}
    score, total_w = 0, 0
    for key, res in asset_results.items():
        w = weights.get(key, 1)
        s = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}.get(res["bias"], 0)
        score  += s * w
        total_w += w
    ratio = score / total_w if total_w else 0
    overall = "BULLISH" if ratio >= 0.4 else "BEARISH" if ratio <= -0.4 else "NEUTRAL"

    _wbias_save_supabase(week_start, asset_results, overall)

    return {
        "week_start": week_start,
        "overall":    overall,
        "assets":     {k: {"bias": v["bias"], "pct": v.get("pct")} for k,v in asset_results.items()},
        "source":     "live",
    }

def render_weekly_bias_banner():
    """Renders the weekly macro bias banner. Call above the scan tab."""
    try:
        bias = get_weekly_macro_bias()
    except Exception:
        return

    overall = bias.get("overall", "NEUTRAL")
    assets  = bias.get("assets", {})
    week    = bias.get("week_start", "")

    colors = {"BULLISH": "#00C853", "BEARISH": "#C1121F", "NEUTRAL": "#A1A1A6"}
    icons  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
    a_icons = {"BULLISH": "✅", "BEARISH": "❌", "NEUTRAL": "➖"}
    col    = colors.get(overall, "#A1A1A6")
    icon   = icons.get(overall, "⚪")

    asset_html = ""
    for k, v in assets.items():
        b = v.get("bias", "NEUTRAL")
        p = v.get("pct")
        pct_str = " %+.1f%%" % p if p is not None else ""
        a_col = colors.get(b, "#A1A1A6")
        asset_html += (
            "<span style='font-size:0.75rem;margin-right:20px'>"
            "%s <b style='color:#F5F5F5'>%s</b> "
            "<span style='color:%s'>%s%s</span></span>"
        ) % (a_icons.get(b, "➖"), k, a_col, b, pct_str)

    st.markdown(
        "<div style='background:#0d0d0f;border:1px solid %s44;border-left:3px solid %s;"
        "border-radius:8px;padding:10px 16px;margin-bottom:10px'>"
        "<div style='font-size:0.6rem;color:#A1A1A6;letter-spacing:0.15em;margin-bottom:4px'>"
        "WEEKLY MACRO BIAS — WEEK OF %s</div>"
        "<div style='font-size:1rem;font-weight:700;color:%s;margin-bottom:6px'>%s %s</div>"
        "<div>%s</div></div>" % (col, col, week, col, icon, overall, asset_html),
        unsafe_allow_html=True
    )


@_thread_cache(ttl=900)
@_thread_cache(ttl=900)
def fetch_ticker_news(ticker, hours=4, limit=10):
    """Fetch ticker news from Finnhub. 15-min cache prevents rate-limit spam."""
    if not FINNHUB_API_KEY:
        return []
    try:
        from_d = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        to_d   = datetime.now().strftime("%Y-%m-%d")
        url = (
            "https://finnhub.io/api/v1/company-news"
            "?symbol=%s&from=%s&to=%s&token=%s"
            % (ticker.upper(), from_d, to_d, FINNHUB_API_KEY)
        )
        r = _http_get(url, timeout=8)
        if r is None:
            return []
        if r.status_code == 429:
            print("[news] Finnhub rate limit hit for %s" % ticker)
            return []
        if r.status_code != 200:
            print("[news] Finnhub returned %s for %s" % (r.status_code, ticker))
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        normalized = []
        for a in data[:limit]:
            normalized.append({
                "title":         a.get("headline", ""),
                "text":          a.get("summary", ""),
                "site":          a.get("source", ""),
                "url":           a.get("url", ""),
                "publishedDate": str(a.get("datetime", "")),
            })
        return normalized
    except Exception:
        return []


@_thread_cache(ttl=180)
def fetch_market_news(hours=2, limit=20):
    """Fetch general market news from Finnhub."""
    if not FINNHUB_API_KEY:
        return []
    try:
        import requests as _req
        url = (
            "https://finnhub.io/api/v1/news"
            "?category=general&token=%s" % FINNHUB_API_KEY
        )
        r = _http_get(url, timeout=5)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        normalized = []
        for a in data[:limit]:
            normalized.append({
                "title":         a.get("headline", ""),
                "text":          a.get("summary", ""),
                "site":          a.get("source", ""),
                "url":           a.get("url", ""),
                "publishedDate": str(a.get("datetime", "")),
            })
        return normalized
    except Exception:
        return []


def score_news_sentiment(articles, ticker=""):
    if not articles:
        return {
            "sentiment": "neutral", "score": 0,
            "bear_score": 0, "bull_score": 0,
            "suggested_action": "EITHER",
            "untradeable": False, "untradeable_reason": "",
            "flags": [], "headlines": [], "article_count": 0,
        }

    total_bear = 0
    total_bull = 0
    flags      = []
    headlines  = []
    untradeable        = False
    untradeable_reason = ""

    for article in articles[:10]:
        title    = (article.get("title",   "") or "").lower()
        text     = (article.get("text",    "") or
                    article.get("content", "") or
                    article.get("summary", "") or "").lower()
        combined = title + " " + text

        if article.get("title"):
            headlines.append({
                "title": article["title"],
                "url":   article.get("url", ""),
                "date":  article.get("publishedDate", ""),
                "site":  article.get("site", ""),
            })

        for word in _UNTRADEABLE_KEYWORDS:
            if word in combined:
                untradeable        = True
                untradeable_reason = "Trading halt or delisting — options market unavailable"
                flags.append("🚫 %s" % word.upper())

        src_wt = get_source_weight(article.get("site", ""))
        for kw, weight in _BEAR_KEYWORDS.items():
            if kw in combined:
                total_bear += weight * src_wt
                if weight >= 2:
                    flags.append("🔴 %s" % kw)

        for kw, weight in _BULL_KEYWORDS.items():
            if kw in combined:
                total_bull += weight * src_wt
                if weight >= 2:
                    flags.append("🟢 %s" % kw)

    total = total_bear + total_bull
    norm_score = 0 if total == 0 else int(((total_bull - total_bear) / total) * 100)

    if norm_score >= 25:
        sentiment        = "bullish"
        suggested_action = "CALL"
    elif norm_score <= -25:
        sentiment        = "bearish"
        suggested_action = "PUT"
    else:
        sentiment        = "neutral"
        suggested_action = "EITHER"

    seen, deduped = set(), []
    for f in flags:
        k = f.lower().strip()
        if k not in seen:
            seen.add(k)
            deduped.append(f)

    velocity_score, is_breaking, recent_count = calculate_news_velocity(articles)

    return {
        "sentiment":          sentiment,
        "score":              norm_score,
        "bear_score":         total_bear,
        "bull_score":         total_bull,
        "suggested_action":   suggested_action,
        "untradeable":        untradeable,
        "untradeable_reason": untradeable_reason,
        "flags":              deduped[:8],
        "headlines":          headlines[:5],
        "article_count":      len(articles),
        "velocity_score":     velocity_score,
        "is_breaking":        is_breaking,
        "recent_count":       recent_count,
    }

def check_macro_sentiment():
    articles   = fetch_market_news(hours=2, limit=20)
    macro_bear = False
    macro_bull = False
    triggers   = []

    for article in articles:
        title    = (article.get("title",   "") or "").lower()
        text     = (article.get("text",    "") or article.get("content", "") or "").lower()
        combined = title + " " + text
        for t in _MACRO_BEAR_TRIGGERS:
            if t in combined:
                macro_bear = True
                triggers.append("⚠️ %s" % t)
        for t in _MACRO_BULL_TRIGGERS:
            if t in combined:
                macro_bull = True
                triggers.append("✅ %s" % t)

    seen, deduped = set(), []
    for t in triggers:
        k = t.lower().strip()
        if k not in seen:
            seen.add(k)
            deduped.append(t)

    return macro_bear, macro_bull, deduped[:6]


def run_news_check(ticker, direction):
    try:
        articles = fetch_ticker_news(ticker, hours=4, limit=10)
        news     = score_news_sentiment(articles, ticker)

        if news["untradeable"]:
            return True, news["untradeable_reason"], news, 0, False, ""

        score     = news["score"]
        sentiment = news["sentiment"]

        if direction == "bullish":
            if sentiment == "bullish":   conf_adj = +8
            elif sentiment == "bearish": conf_adj = -12
            else:                        conf_adj = 0
        else:
            if sentiment == "bearish":   conf_adj = +8
            elif sentiment == "bullish": conf_adj = -12
            else:                        conf_adj = 0

        flip_signal = False
        flip_reason = ""

        if direction == "bullish" and score <= -50:
            flip_signal = True
            flip_reason = (
                "News strongly BEARISH (score %s) — "
                "consider PUT to ride the news catalyst." % score
            )
        elif direction == "bearish" and score >= 50:
            flip_signal = True
            flip_reason = (
                "News strongly BULLISH (score %s) — "
                "consider CALL to ride the news catalyst." % score
            )

        return False, "", news, conf_adj, flip_signal, flip_reason

    except Exception:
        empty = {
            "sentiment": "neutral", "score": 0, "bear_score": 0, "bull_score": 0,
            "suggested_action": "EITHER", "untradeable": False,
            "untradeable_reason": "", "flags": [], "headlines": [], "article_count": 0,
        }
        return False, "", empty, 0, False, ""


@_thread_cache(ttl=3600)
def fetch_400ma_daily(ticker):
    """Daily 400 SMA — the long-term trend line."""
    try:
        df = _fmp_download(ticker, "1y", "1d")
        if df is None or len(df) < 50:
            return None, None, None
        close   = df["close"].astype(float)
        period  = min(400, len(close))
        ma      = close.rolling(period).mean().dropna()
        if ma.empty:
            return None, None, None
        ma_now    = float(ma.iloc[-1])
        ma_prior  = float(ma.iloc[-6]) if len(ma) >= 6 else ma_now
        cur_price = float(close.iloc[-1])
        above     = cur_price > ma_now
        rising    = ma_now > ma_prior
        return above, round(ma_now, 2), rising
    except Exception:
        return None, None, None

@_thread_cache(ttl=300)
def fetch_400ma_5min(ticker):
    """5-min 400 SMA - intraday long-term reference."""
    try:
        df = _fmp_download(ticker, "5d", "5m")
        if df is None or len(df) < 50:
            return None, None
        close  = df["close"].astype(float)
        period = min(400, len(close))
        ma     = close.rolling(period).mean().dropna()
        if ma.empty:
            return None, None
        ma_now    = float(ma.iloc[-1])
        cur_price = float(close.iloc[-1])
        above     = cur_price > ma_now
        return above, round(ma_now, 2)
    except Exception:
        return None, None

def _calc_vwap_now(df):
    try:
        if df is None or len(df) < 5:
            return None
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        tp = (h + l + c) / 3
        cv = (tp * v).cumsum()
        cu = v.cumsum().replace(0, 1)
        return round(float((cv / cu).iloc[-1]), 2)
    except Exception:
        return None

def _calc_9ema_now(df):
    try:
        if df is None or len(df) < 9:
            return None
        return round(float(df["close"].astype(float).ewm(span=9).mean().iloc[-1]), 2)
    except Exception:
        return None

def detect_confluence_setup(ticker, current_price, direction, atr=None):
    """Confluence Intel: VWAP + 9EMA + 400SMA stack with plain-English game plan."""
    out = {
        "available":        False,
        "daily_bias":       None,
        "daily_400ma":      None,
        "daily_400ma_rising": None,
        "intraday_trend":   None,
        "intraday_400ma":   None,
        "vwap":             None,
        "ema9":             None,
        "zone_low":         None,
        "zone_high":        None,
        "game_plan":        "",
        "invalidation":     "",
        "timestamp":        "",
        "alignment_count":  0,
    }
    try:
        df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is not None and "datetime" in df_5m.columns:
            df_5m = df_5m.sort_values("datetime").reset_index(drop=True)

        vwap = _calc_vwap_now(df_5m)
        ema9 = _calc_9ema_now(df_5m)
        d_above, d_400, d_rising = fetch_400ma_daily(ticker)
        i_above, i_400           = fetch_400ma_5min(ticker)

        is_bull = direction == "bullish"

        out["vwap"]               = vwap
        out["ema9"]               = ema9
        out["daily_bias"]         = d_above
        out["daily_400ma"]        = d_400
        out["daily_400ma_rising"] = d_rising
        out["intraday_trend"]     = i_above
        out["intraday_400ma"]     = i_400

        align = 0
        if d_above is not None:
            if (is_bull and d_above) or (not is_bull and not d_above):
                align += 1
        if i_above is not None:
            if (is_bull and i_above) or (not is_bull and not i_above):
                align += 1
        if vwap is not None and current_price:
            above_vwap = current_price > vwap
            if (is_bull and above_vwap) or (not is_bull and not above_vwap):
                align += 1
        if ema9 is not None and current_price:
            above_9 = current_price > ema9
            if (is_bull and above_9) or (not is_bull and not above_9):
                align += 1
        out["alignment_count"] = align

        if vwap and ema9 and current_price:
            zone_lo = round(min(vwap, ema9), 2)
            zone_hi = round(max(vwap, ema9), 2)
            out["zone_low"]  = zone_lo
            out["zone_high"] = zone_hi

            if is_bull:
                if current_price >= zone_lo and current_price <= zone_hi:
                    out["game_plan"] = (
                        "Price is in the sweet spot zone. "
                        "If it pushes above $%.2f on the next 5-min candle, "
                        "that is your entry signal." % zone_hi
                    )
                elif current_price > zone_hi:
                    out["game_plan"] = (
                        "Price already above the sweet spot. Don't chase. "
                        "Wait for a pullback into $%.2f - $%.2f before entering." % (zone_lo, zone_hi)
                    )
                else:
                    out["game_plan"] = (
                        "Price below the sweet spot. Wait for it to climb back into "
                        "$%.2f - $%.2f and hold there before entering." % (zone_lo, zone_hi)
                    )
                out["invalidation"] = (
                    "If price closes below $%.2f on a 5-min candle, "
                    "the setup is broken. Stay out." % zone_lo
                )
            else:
                if current_price >= zone_lo and current_price <= zone_hi:
                    out["game_plan"] = (
                        "Price is in the sweet spot zone. "
                        "If it drops below $%.2f on the next 5-min candle, "
                        "that is your entry signal for a put." % zone_lo
                    )
                elif current_price < zone_lo:
                    out["game_plan"] = (
                        "Price already dropped past the sweet spot. Don't chase. "
                        "Wait for a bounce back to $%.2f - $%.2f before entering." % (zone_lo, zone_hi)
                    )
                else:
                    out["game_plan"] = (
                        "Price above the sweet spot. Wait for it to drop into "
                        "$%.2f - $%.2f and reject before entering." % (zone_lo, zone_hi)
                    )
                out["invalidation"] = (
                    "If price closes above $%.2f on a 5-min candle, "
                    "the setup is broken. Stay out." % zone_hi
                )

        try:
            et = pytz.timezone("America/New_York")
            out["timestamp"] = datetime.now(et).strftime("%I:%M:%S %p ET")
        except Exception:
            out["timestamp"] = datetime.now().strftime("%H:%M:%S")

        out["available"] = (vwap is not None and ema9 is not None)
        return out
    except Exception:
        return out

def get_time_of_day_context():
    """Returns trading window + plain-English meaning for current time."""
    try:
        et = pytz.timezone("America/New_York")
        now = datetime.now(et)
        wd  = now.weekday()
        if wd >= 5:
            return {"window":"WEEKEND","label":"Market Closed","color":"#A1A1A6","icon":"🌙",
                    "meaning":"Market opens Monday at 9:30 AM ET. Use this time to plan, not trade."}
        from datetime import time as dtime
        t = now.time()
        if t < dtime(9, 30):
            return {"window":"PRE","label":"Pre-Market","color":"#F6E27A","icon":"⏰",
                    "meaning":"Market hasn't opened yet. Levels can shift hard at 9:30. Wait for the open before entering."}
        if t < dtime(10, 30):
            return {"window":"OPEN","label":"Opening Hour","color":"#FF6B35","icon":"🚨",
                    "meaning":"First hour. Volatility is highest. Moves are fast and can reverse hard. Tight stops or wait until 10:30."}
        if t < dtime(12, 0):
            return {"window":"MORNING","label":"Morning Trend","color":"#22C55E","icon":"📈",
                    "meaning":"Best window of the day. Morning trend is usually most reliable. If signals fire here, take them seriously."}
        if t < dtime(14, 0):
            return {"window":"LUNCH","label":"Lunch Chop","color":"#A1A1A6","icon":"⏸",
                    "meaning":"Volume drops off. Lots of fakeouts. Save your bullets for after 2 PM unless the setup is exceptional."}
        if t < dtime(15, 30):
            return {"window":"AFTERNOON","label":"Afternoon Trend","color":"#22C55E","icon":"🎯",
                    "meaning":"Real trend of the day usually shows up here. Best window for swing entries — what closes here often follows through tomorrow."}
        if t < dtime(16, 0):
            return {"window":"POWER","label":"Power Hour","color":"#D4AF37","icon":"🔥",
                    "meaning":"Final 30 minutes. Big moves common. Stick with the established trend — don't chase reversals here."}
        return {"window":"AFTER","label":"After-Hours","color":"#A1A1A6","icon":"🌙",
                "meaning":"Market is closed. Anything you see here is for tomorrow's plan. Don't trade options after-hours."}
    except Exception:
        return {"window":"UNKNOWN","label":"-","color":"#A1A1A6","icon":"-","meaning":"Time context unavailable."}

def build_confluence_summary(r):
    """One-line synthesis verdict across the entire signal stack."""
    detail = r.get("detail", {}) or {}
    direction = r.get("direction", "bullish")
    is_bull   = direction == "bullish"

    aligned = 0
    total   = 0
    factors = []
    against = []

    sig_hit = r.get("signals_hit", detail.get("signals_hit", 0))
    if sig_hit >= 5:
        aligned += 2
        factors.append("Most setup signals confirmed (%s of 7)" % sig_hit)
    elif sig_hit >= 3:
        aligned += 1
        factors.append("Some setup signals confirmed (%s of 7)" % sig_hit)
    else:
        against.append("Few setup signals confirmed (%s of 7)" % sig_hit)
    total += 2

    ma200_above  = detail.get("ma200_above")
    ma200_rising = detail.get("ma200_rising")
    if ma200_above is not None:
        if is_bull and ma200_above and ma200_rising:
            aligned += 1; factors.append("Long-term trend up")
        elif not is_bull and not ma200_above and not ma200_rising:
            aligned += 1; factors.append("Long-term trend down")
        else:
            against.append("Long-term trend not aligned")
        total += 1

    mb_label = r.get("macro_bias_label", "")
    if "ALIGNED" in mb_label:
        aligned += 1; factors.append("Big-picture macro on your side"); total += 1
    elif "HEADWIND" in mb_label:
        against.append("Big-picture macro against you"); total += 1

    align_state = r.get("regime_alignment", "")
    if align_state == "CONFIRMED":
        aligned += 1; factors.append("Market regime backs the trade"); total += 1
    elif align_state == "COUNTER":
        against.append("Trading against current market regime"); total += 1
    elif align_state == "BLOCKED":
        against.append("Market regime blocks this direction"); total += 1

    sr = detail.get("sr_data", {}) or {}
    sr_boost = sr.get("conf_boost", 0)
    if sr_boost >= 8:
        aligned += 1; factors.append(("At support" if is_bull else "At resistance") + " - clean entry"); total += 1
    elif sr_boost > 0:
        aligned += 1; factors.append("Near key level"); total += 1
    elif sr_boost < 0:
        against.append("Buying into resistance" if is_bull else "Shorting into support"); total += 1

    cfl = r.get("confluence", {}) or {}
    if cfl.get("available"):
        cfl_align = cfl.get("alignment_count", 0)
        if cfl_align >= 3:
            aligned += 1; factors.append("Trend lines stacked correctly")
        elif cfl_align >= 2:
            aligned += 1; factors.append("Most trend lines aligned")
        else:
            against.append("Trend lines not aligned for this direction")
        total += 1

    news_sent = detail.get("news_sentiment", "neutral")
    if (is_bull and news_sent == "bullish") or (not is_bull and news_sent == "bearish"):
        aligned += 1; factors.append("News flow supports the trade"); total += 1
    elif (is_bull and news_sent == "bearish") or (not is_bull and news_sent == "bullish"):
        against.append("News flow goes the other way"); total += 1

    against_bias = detail.get("against_market_bias", False)
    if against_bias:
        against.append("Trade goes against today's market bias"); total += 1

    pct = (aligned / max(total, 1)) * 100 if total else 0

    if aligned >= 7 and pct >= 80:
        tier = "ALL SYSTEMS GO"; color = "#22C55E"; icon = "🟢"
        verdict = "Every layer points the same way. Highest-conviction setup. Trade with normal size."
    elif aligned >= 5 and pct >= 60:
        tier = "STRONG ALIGNMENT"; color = "#D4AF37"; icon = "✅"
        verdict = "Most layers agree. Solid setup. Trade with normal size."
    elif aligned >= 3 and pct >= 40:
        tier = "MIXED SIGNALS"; color = "#F6E27A"; icon = "⚠️"
        verdict = "Some layers agree, some don't. Reduce position size by half. Tight stop."
    elif aligned >= 2:
        tier = "WEAK ALIGNMENT"; color = "#FF6B35"; icon = "🟠"
        verdict = "Most layers conflict. Consider sitting this one out or paper-trading it."
    else:
        tier = "CONFLICTING"; color = "#C1121F"; icon = "🔴"
        verdict = "Stack disagrees with the trade direction. Skip — wait for a cleaner setup."

    return {"tier":tier, "color":color, "icon":icon, "verdict":verdict,
            "aligned":aligned, "total":total,
            "factors":factors[:5], "against":against[:3]}

@_thread_cache(ttl=3600)
def get_todays_macro_events():
    """Pull today's high-impact economic events from FMP.
    Returns list of dicts with event name, time, impact level."""
    if not FMP_API_KEY:
        return []
    try:
        et   = pytz.timezone("America/New_York")
        now  = datetime.now(et)
        date = now.strftime("%Y-%m-%d")
        url  = (
            "https://financialmodelingprep.com/api/v3/economic_calendar"
            "?from=%s&to=%s&apikey=%s" % (date, date, FMP_API_KEY)
        )
        r = _http_get(url, timeout=8)
        if r is None or r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []

        HIGH_IMPACT = [
            "cpi", "ppi", "inflation", "fed", "fomc", "interest rate",
            "jobs", "nonfarm", "unemployment", "gdp", "retail sales",
            "pce", "payroll", "consumer price", "producer price",
            "core inflation", "housing starts", "ism manufacturing",
            "ism services", "consumer confidence", "durable goods",
        ]
        events = []
        for item in data:
            if item.get("country", "").upper() != "US":
                continue
            impact = (item.get("impact") or "").lower()
            name   = (item.get("event") or "").lower()
            if impact not in ("high", "medium") and not any(k in name for k in HIGH_IMPACT):
                continue
            if impact == "low" and not any(k in name for k in HIGH_IMPACT):
                continue
            # Parse event time
            event_time = item.get("date", "") or item.get("time", "")
            events.append({
                "name":   item.get("event", "Economic Event"),
                "time":   event_time,
                "impact": impact,
                "actual": item.get("actual"),
                "estimate": item.get("estimate"),
                "previous": item.get("previous"),
            })
        return events
    except Exception as _e:
        print("[macro_events] error: %s" % str(_e)[:100])
        return []

def get_macro_event_warning():
    """Returns warning dict if high-impact events are today.
    Levels:
      PRE_EVENT  = event coming up in next 3 hours, treat signals as lower confidence
      POST_EVENT = event already printed, market may still be digesting
      CLEAR      = no events or events are minor
    """
    try:
        et      = pytz.timezone("America/New_York")
        now     = datetime.now(et)
        events  = get_todays_macro_events()
        if not events:
            return {"level": "CLEAR", "events": [], "message": ""}

        HIGH_KEYWORDS = [
            "cpi", "ppi", "fed", "fomc", "nonfarm", "unemployment",
            "gdp", "interest rate", "core inflation", "pce", "payroll",
        ]

        warnings = []
        for ev in events:
            name   = ev["name"].lower()
            is_key = any(k in name for k in HIGH_KEYWORDS)
            if not is_key and ev["impact"] != "high":
                continue

            # Try to parse event time
            ev_hour = None
            try:
                ev_time_str = ev["time"]
                if ev_time_str:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%H:%M"):
                        try:
                            parsed = datetime.strptime(ev_time_str[:19], fmt[:len(ev_time_str[:19])])
                            ev_hour = parsed.hour
                            break
                        except Exception:
                            continue
            except Exception:
                pass

            warnings.append({
                "name":    ev["name"],
                "time":    ev.get("time", ""),
                "hour":    ev_hour,
                "actual":  ev.get("actual"),
                "estimate":ev.get("estimate"),
            })

        if not warnings:
            return {"level": "CLEAR", "events": [], "message": ""}

        # Determine level based on timing
        current_hour = now.hour
        pre_events   = []
        post_events  = []
        unknown_time = []

        for w in warnings:
            h = w.get("hour")
            if h is None:
                unknown_time.append(w)
            elif current_hour < h and (h - current_hour) <= 3:
                pre_events.append(w)
            elif current_hour >= h:
                post_events.append(w)
            else:
                unknown_time.append(w)

        if pre_events:
            names = ", ".join(e["name"] for e in pre_events[:2])
            return {
                "level":   "PRE_EVENT",
                "events":  pre_events,
                "message": (
                    "%s coming up. Signals carry elevated risk until data clears. "
                    "Consider waiting for the print before entering." % names
                ),
            }
        elif post_events:
            # Check if actual vs estimate shows a surprise
            surprises = [e for e in post_events if e.get("actual") and e.get("estimate")]
            msg_parts = []
            for e in surprises[:2]:
                try:
                    act = float(str(e["actual"]).replace("%","").replace("K","").replace("M",""))
                    est = float(str(e["estimate"]).replace("%","").replace("K","").replace("M",""))
                    if abs(act - est) / max(abs(est), 0.001) > 0.05:
                        direction = "HIGHER" if act > est else "LOWER"
                        msg_parts.append(
                            "%s came in %s than expected (%.2f vs %.2f est)" % (
                                e["name"], direction, act, est
                            )
                        )
                except Exception:
                    msg_parts.append("%s already printed" % e["name"])

            names = ", ".join(e["name"] for e in post_events[:2])
            base_msg = "%s already printed. " % names
            if msg_parts:
                base_msg += " ".join(msg_parts) + ". "
            base_msg += "Market may still be digesting — signals in first 30-60 min post-print are noisier."
            return {
                "level":   "POST_EVENT",
                "events":  post_events,
                "message": base_msg,
            }
        else:
            names = ", ".join(e["name"] for e in (unknown_time or warnings)[:2])
            return {
                "level":   "PRE_EVENT",
                "events":  unknown_time or warnings,
                "message": (
                    "%s scheduled today. Treat signals as lower confidence "
                    "until the data clears." % names
                ),
            }
    except Exception as _e:
        print("[macro_warning] error: %s" % str(_e)[:100])
        return {"level": "CLEAR", "events": [], "message": ""}

@_thread_cache(ttl=900)
def get_market_stress_monitor():
    out = {
        "available":      False,
        "spy_price":      None,
        "spy_rsi_1d":     None,
        "spy_rsi_1w":     None,
        "spy_pct_5d":     None,
        "spy_vol_ratio":  None,
        "qqq_rsi_1d":     None,
        "iwm_rsi_1d":     None,
        "iwm_new_low":    None,
        "spy_new_low":    None,
        "at_support":     False,
        "support_level":  None,
        "dist_to_sup":    None,
        "def_holding":    0,
        "vix_est":        None,
        "note":           "",
    }
    try:
        def _rsi(close, period=14):
            d = close.diff()
            g = d.clip(lower=0).rolling(period).mean()
            l = (-d.clip(upper=0)).rolling(period).mean()
            return round(float((100 - 100 / (1 + g / l.replace(0, 0.001))).iloc[-1]), 1)

        # SPY daily
        df_spy = _fmp_download("SPY", "60d", "1d")
        if df_spy is None or len(df_spy) < 15:
            return out
        df_spy = df_spy.sort_values("datetime").reset_index(drop=True)
        spy_cl = df_spy["close"].astype(float)
        spy_vol = df_spy["volume"].astype(float)

        out["spy_price"]    = round(float(spy_cl.iloc[-1]), 2)
        out["spy_rsi_1d"]   = _rsi(spy_cl)
        out["spy_pct_5d"]   = round((float(spy_cl.iloc[-1]) - float(spy_cl.iloc[-6])) / float(spy_cl.iloc[-6]) * 100, 1) if len(spy_cl) >= 6 else None
        avg_vol             = float(spy_vol.iloc[-20:].mean())
        out["spy_vol_ratio"] = round(float(spy_vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else None

        # SPY weekly RSI
        try:
            df_wk = _fmp_download("SPY", "2y", "1wk")
            if df_wk is not None and len(df_wk) >= 14:
                out["spy_rsi_1w"] = _rsi(df_wk["close"].astype(float))
        except Exception:
            pass

        # New low check (20-day)
        spy_20d_low = float(spy_cl.iloc[-20:].min())
        out["spy_new_low"] = float(spy_cl.iloc[-1]) <= spy_20d_low * 1.005

        # QQQ and IWM
        try:
            df_qqq = _fmp_download("QQQ", "30d", "1d")
            if df_qqq is not None and len(df_qqq) >= 14:
                out["qqq_rsi_1d"] = _rsi(df_qqq["close"].astype(float).sort_values().reset_index(drop=True))
        except Exception:
            pass

        try:
            df_iwm = _fmp_download("IWM", "30d", "1d")
            if df_iwm is not None and len(df_iwm) >= 14:
                iwm_cl = df_iwm["close"].astype(float)
                out["iwm_rsi_1d"]  = _rsi(iwm_cl)
                out["iwm_new_low"] = float(iwm_cl.iloc[-1]) <= float(iwm_cl.iloc[-20:].min()) * 1.005
        except Exception:
            pass

        # Support level
        try:
            sr = detect_sr_levels("SPY", out["spy_price"], "bearish")
            sup = sr.get("nearest_support")
            if sup:
                dist = round((out["spy_price"] - sup) / out["spy_price"] * 100, 1)
                out["at_support"]    = sr.get("at_support", False) or dist < 1.0
                out["support_level"] = round(sup, 2)
                out["dist_to_sup"]   = dist
        except Exception:
            pass

        # Defensive sectors holding
        def_count = 0
        for etf in ("XLU", "XLP", "XLV"):
            try:
                df_d = _fmp_download(etf, "20d", "1d")
                if df_d is not None and len(df_d) >= 5:
                    r = _rsi(df_d["close"].astype(float))
                    if r > 42:
                        def_count += 1
            except Exception:
                pass
        out["def_holding"] = def_count

        # VIX approximation via VIXY (flagged as proxy, not spot)
        try:
            df_vx = _fmp_download("VIXY", "10d", "1d")
            if df_vx is not None and len(df_vx) >= 2:
                out["vix_est"] = round(float(df_vx["close"].astype(float).iloc[-1]), 2)
        except Exception:
            pass

        # Plain note based on what we see — no prediction, just description
        notes = []
        rsi_d = out["spy_rsi_1d"]
        rsi_w = out["spy_rsi_1w"]
        if rsi_d and rsi_d < 30:
            notes.append("SPY daily RSI deeply oversold (%.0f)" % rsi_d)
        elif rsi_d and rsi_d < 38:
            notes.append("SPY daily RSI oversold (%.0f)" % rsi_d)
        if rsi_w and rsi_w < 35:
            notes.append("weekly RSI also oversold (%.0f) — selling broad" % rsi_w)
        if out["spy_new_low"] and out["iwm_new_low"] is False:
            notes.append("IWM not making new lows with SPY — selling may be narrowing")
        if out["at_support"]:
            notes.append("sitting at key support $%.2f" % out["support_level"])
        elif out["dist_to_sup"] and out["dist_to_sup"] < 2.5:
            notes.append("%.1f%% above key support $%.2f" % (out["dist_to_sup"], out["support_level"]))
        if out["def_holding"] >= 2:
            notes.append("%d/3 defensive sectors holding (rotation signal)" % out["def_holding"])
        if out["spy_pct_5d"] and out["spy_pct_5d"] < -4:
            notes.append("down %.1f%% in 5 days — pace of selling elevated" % out["spy_pct_5d"])

        out["note"]      = " · ".join(notes) if notes else ""
        out["available"] = rsi_d is not None and rsi_d < 45
        return out
    except Exception as _e:
        print("[stress] error: %s" % str(_e)[:150])
        return out

@_thread_cache(ttl=300)
def get_pattern_win_rate(pattern, style=None, direction=None):
    """Win rate for a specific pattern."""
    min_sample = 10
    sb = get_supabase(service=True)
    if not sb:
        return None
    try:
        q = sb.table("signal_outcomes").select("*").eq("pattern", pattern).neq("result", "OPEN")
        if style:
            q = q.eq("style", style)
        if direction:
            q = q.eq("direction", direction)
        res = q.limit(500).execute()
        rows = res.data or []
        if len(rows) < min_sample:
            return {"win_rate": None, "sample_size": len(rows), "min_sample": min_sample,
                    "avg_winner": None, "avg_loser": None, "expectancy": None,
                    "ready": False}

        wins   = [float(r.get("outcome_5d") or r.get("outcome_3d") or r.get("outcome_1d") or 0)
                  for r in rows if r.get("result") == "WIN"]
        losses = [float(r.get("outcome_5d") or r.get("outcome_3d") or r.get("outcome_1d") or 0)
                  for r in rows if r.get("result") == "LOSS"]
        n_total = len(wins) + len(losses)
        if n_total == 0:
            return {"win_rate": None, "sample_size": 0, "min_sample": min_sample,
                    "ready": False}

        win_rate   = len(wins) / n_total
        avg_winner = sum(wins) / len(wins) if wins else 0
        avg_loser  = sum(losses) / len(losses) if losses else 0
        expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)

        return {
            "win_rate":    round(win_rate * 100, 1),
            "sample_size": n_total,
            "min_sample":  min_sample,
            "avg_winner":  round(avg_winner, 2),
            "avg_loser":   round(avg_loser, 2),
            "expectancy":  round(expectancy, 2),
            "ready":       True,
        }
    except Exception as _e:
        print("[analytics] pattern win rate error: %s" % str(_e)[:120])
        return None

@_thread_cache(ttl=300)
def get_regime_win_rates():
    """Win rates grouped by market regime."""
    min_sample = 5
    sb = get_supabase(service=True)
    if not sb:
        return []
    try:
        res = sb.table("signal_outcomes").select("*").neq("result", "OPEN").limit(1000).execute()
        rows = res.data or []
        buckets = {}
        for row in rows:
            regime = row.get("market_regime") or "UNKNOWN"
            buckets.setdefault(regime, {"wins": 0, "losses": 0})
            if row.get("result") == "WIN":
                buckets[regime]["wins"] += 1
            elif row.get("result") == "LOSS":
                buckets[regime]["losses"] += 1
        out = []
        for regime, d in buckets.items():
            n = d["wins"] + d["losses"]
            if n < min_sample:
                continue
            wr = d["wins"] / n if n else 0
            out.append({
                "regime":      regime,
                "sample":      n,
                "wins":        d["wins"],
                "losses":      d["losses"],
                "win_rate":    round(wr * 100, 1),
            })
        return sorted(out, key=lambda x: x["win_rate"], reverse=True)
    except Exception as _e:
        print("[analytics] regime win rates error: %s" % str(_e)[:120])
        return []

@_thread_cache(ttl=300)
def get_confluence_correlation():
    """Does confluence alignment correlate with wins?"""
    min_sample = 5
    sb = get_supabase(service=True)
    if not sb:
        return []
    try:
        res = sb.table("signal_outcomes").select("*").neq("result", "OPEN").limit(1000).execute()
        rows = res.data or []
        buckets = {0: {"w":0,"l":0}, 1: {"w":0,"l":0}, 2: {"w":0,"l":0},
                   3: {"w":0,"l":0}, 4: {"w":0,"l":0}}
        for row in rows:
            try:
                ca = int(row.get("confluence_alignment") or 0)
            except Exception:
                ca = 0
            ca = max(0, min(4, ca))
            if row.get("result") == "WIN":
                buckets[ca]["w"] += 1
            elif row.get("result") == "LOSS":
                buckets[ca]["l"] += 1
        out = []
        for ca, d in sorted(buckets.items()):
            n = d["w"] + d["l"]
            if n < min_sample:
                out.append({"alignment": ca, "sample": n, "win_rate": None, "ready": False})
            else:
                wr = d["w"] / n if n else 0
                out.append({"alignment": ca, "sample": n,
                            "win_rate": round(wr * 100, 1), "ready": True})
        return out
    except Exception as _e:
        print("[analytics] confluence correlation error: %s" % str(_e)[:120])
        return []

@_thread_cache(ttl=300)
def get_overall_stats():
    """Total stats across all signals - top of dashboard summary."""
    sb = get_supabase(service=True)
    if not sb:
        return None
    try:
        res = sb.table("signal_outcomes").select("result").limit(2000).execute()
        rows = res.data or []
        n_open = sum(1 for r in rows if r.get("result") == "OPEN")
        n_win  = sum(1 for r in rows if r.get("result") == "WIN")
        n_loss = sum(1 for r in rows if r.get("result") == "LOSS")
        n_resolved = n_win + n_loss
        wr = (n_win / n_resolved * 100) if n_resolved else None
        return {
            "total":      len(rows),
            "open":       n_open,
            "wins":       n_win,
            "losses":     n_loss,
            "resolved":   n_resolved,
            "win_rate":   round(wr, 1) if wr is not None else None,
        }
    except Exception:
        return None

def render_sniper_card_html(r, sh):
    dc = "#D4AF37" if r.get("direction") == "bullish" else "#C1121F"
    act = "BUY CALL" if r.get("direction") == "bullish" else "BUY PUT"
    conf = str(r.get("confidence", 0))
    gates = str(r.get("gates_passed", 0))
    entry = "%.2f" % (r.get("price", 0) or 0)
    stop  = "%.2f" % ((r.get("opt", {}) or {}).get("stop", 0) or 0)
    return (
        "<div style='background:linear-gradient(135deg,#0f0f12,#1a1a1d);"
        "border:2px solid " + dc + ";border-radius:10px;"
        "padding:16px 20px;margin-bottom:12px'>"
        "<div style='display:flex;justify-content:space-between;"
        "align-items:center;margin-bottom:10px'>"
        "<div>"
        "<span style='font-family:Barlow Condensed,Arial Black,sans-serif;"
        "font-size:1.3rem;font-weight:900;color:" + dc + "'>"
        + str(r["ticker"]) + "</span>"
        "<span style='color:#A1A1A6;font-size:0.82rem;margin-left:10px'>"
        + act + " - " + str(r.get("pattern", "Signal")) + "</span>"
        "</div>"
        "<div style='text-align:right'>"
        "<div style='font-size:0.65rem;color:#A1A1A6'>EXECUTION</div>"
        "<div style='font-size:1.1rem;font-weight:700;color:#00C853'>"
        + str(sh["exec"]) + "%</div>"
        "</div></div>"
        "<div style='font-size:0.8rem;color:#F5F5F5;font-weight:700'>"
        "! " + str(sh["trigger"]) + " -> " + str(sh["window"]) + "</div>"
        "<div style='font-size:0.72rem;color:#A1A1A6;margin-top:4px'>"
        + str(sh["detail"]) + "</div>"
        "<div style='display:flex;gap:20px;margin-top:10px;font-size:0.78rem'>"
        "<span style='color:#A1A1A6'>Conf: <b style='color:#F5F5F5'>" + conf + "%</b></span>"
        "<span style='color:#A1A1A6'>Gates: <b style='color:#F5F5F5'>" + gates + "/7</b></span>"
        "<span style='color:#A1A1A6'>Entry: <b style='color:#D4AF37'>$" + entry + "</b></span>"
        "<span style='color:#A1A1A6'>Stop: <b style='color:#FF1744'>$" + stop + "</b></span>"
        "</div></div>"
    )

def section_hdr(label, color, count):
    _plural = "s" if count != 1 else ""
    _html = (
        "<div style='display:flex;align-items:center;"
        "gap:10px;margin:20px 0 8px'>"
        "<div style='width:3px;height:16px;background:"
        + color +
        ";border-radius:2px;flex-shrink:0'></div>"
        "<span style='font-size:0.65rem;letter-spacing:3px;color:"
        + color +
        ";font-weight:700'>"
        + label +
        "</span>"
        "<div style='flex:1;height:1px;background:#2A2A2D'></div>"
        "<span style='font-size:0.62rem;color:#A1A1A6'>"
        + str(count) +
        " signal" + _plural +
        "</span></div>"
    )
    st.markdown(_html, unsafe_allow_html=True)

def empty_bkt(msg):
    st.markdown(
        "<div style='padding:14px;color:#A1A1A6;font-size:0.78rem;"
        "background:#1A1A1D;border-radius:10px;text-align:center'>"
        + str(msg) + "</div>",
        unsafe_allow_html=True
    )

def render_summary_line_html(r):
    """Top-of-card alignment verdict block."""
    s = build_confluence_summary(r)
    factors_html = ""
    for f in s["factors"]:
        factors_html += "<div style='font-size:0.72rem;color:#22C55E;margin:1px 0'>✅ " + f + "</div>"
    against_html = ""
    for a in s["against"]:
        against_html += "<div style='font-size:0.72rem;color:#C1121F;margin:1px 0'>⚠️ " + a + "</div>"

    return (
        "<div style='background:#0d0d0f;border:2px solid " + s["color"] + ";border-radius:10px;"
        "padding:12px 16px;margin:8px 0'>"
        "<div style='display:flex;justify-content:space-between;align-items:center'>"
        "<div>"
        "<span style='font-family:Barlow Condensed,Arial Black,sans-serif;"
        "font-size:1rem;font-weight:900;letter-spacing:0.05em;color:" + s["color"] + "'>"
        + s["icon"] + " " + s["tier"] + "</span>"
        "<span style='color:#A1A1A6;font-size:0.7rem;margin-left:8px'>"
        "(" + str(s["aligned"]) + " of " + str(s["total"]) + " factors aligned)</span>"
        "</div></div>"
        "<div style='font-size:0.78rem;color:#F5F5F5;margin-top:4px;line-height:1.5'>" + s["verdict"] + "</div>"
        "<div style='margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:6px'>"
        "<div>" + (factors_html or "<div style='color:#4a5568;font-size:0.7rem'>No strong positives</div>") + "</div>"
        "<div>" + (against_html or "<div style='color:#4a5568;font-size:0.7rem'>No major issues</div>") + "</div>"
        "</div>"
        "</div>"
    )
def render_confluence_block_html(cfl, direction):
    """Confluence Intel display block - VWAP + 9EMA + 400SMA stack."""
    if not cfl or not cfl.get("available"):
        return ""

    is_bull = direction == "bullish"
    d_above = cfl.get("daily_bias")
    i_above = cfl.get("intraday_trend")
    d_400   = cfl.get("daily_400ma")
    i_400   = cfl.get("intraday_400ma")
    vwap    = cfl.get("vwap")
    ema9    = cfl.get("ema9")
    z_lo    = cfl.get("zone_low")
    z_hi    = cfl.get("zone_high")
    plan    = cfl.get("game_plan", "")
    invalid = cfl.get("invalidation", "")
    ts      = cfl.get("timestamp", "")

    if d_above is None:
        d_chip = ""
    elif (is_bull and d_above) or (not is_bull and not d_above):
        d_chip = (
            "<div style='font-size:0.72rem'>"
            "<span style='color:#22C55E'>✅ Big-Picture Trend</span>: "
            "Price is " + ("above" if d_above else "below") + " the long-term trend line ($" + ("%.2f" % d_400) + ") - good for this trade."
            "</div>"
        )
    else:
        d_chip = (
            "<div style='font-size:0.72rem'>"
            "<span style='color:#C1121F'>⚠️ Big-Picture Trend</span>: "
            "Price is " + ("above" if d_above else "below") + " the long-term trend line ($" + ("%.2f" % d_400) + ") - working against this trade."
            "</div>"
        )

    if i_above is None:
        i_chip = ""
    elif (is_bull and i_above) or (not is_bull and not i_above):
        i_chip = (
            "<div style='font-size:0.72rem;margin-top:3px'>"
            "<span style='color:#22C55E'>✅ Today's Trend</span>: "
            "Price is " + ("above" if i_above else "below") + " the intraday trend line ($" + ("%.2f" % i_400) + ") - momentum on your side."
            "</div>"
        )
    else:
        i_chip = (
            "<div style='font-size:0.72rem;margin-top:3px'>"
            "<span style='color:#C1121F'>⚠️ Today's Trend</span>: "
            "Price is " + ("above" if i_above else "below") + " the intraday trend line ($" + ("%.2f" % i_400) + ") - momentum against you."
            "</div>"
        )

    lines_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;"
        "background:#111115;border-radius:6px;padding:8px 10px'>"
        "<div>"
        "<div style='font-size:0.6rem;color:#A1A1A6;letter-spacing:1px'>TODAY'S AVG PRICE (VWAP)</div>"
        "<div style='font-size:0.95rem;font-weight:700;color:#F5F5F5'>$" + ("%.2f" % vwap) + "</div>"
        "</div>"
        "<div>"
        "<div style='font-size:0.6rem;color:#A1A1A6;letter-spacing:1px'>RECENT TREND LINE (9 EMA)</div>"
        "<div style='font-size:0.95rem;font-weight:700;color:#F5F5F5'>$" + ("%.2f" % ema9) + "</div>"
        "</div>"
        "</div>"
    )

    zone_html = ""
    if z_lo and z_hi:
        zone_html = (
            "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:6px;"
            "padding:8px 10px;margin-top:8px'>"
            "<div style='font-size:0.6rem;color:#D4AF37;letter-spacing:1px'>SWEET SPOT ZONE</div>"
            "<div style='font-size:0.95rem;font-weight:700;color:#D4AF37'>$" + ("%.2f" % z_lo) + " - $" + ("%.2f" % z_hi) + "</div>"
            "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:2px'>"
            "Where the trend lines cluster. Best risk/reward entry area.</div>"
            "</div>"
        )

    plan_html = ""
    if plan:
        plan_html = (
            "<div style='background:#0a1a0a;border-left:3px solid #22C55E;border-radius:4px;"
            "padding:8px 12px;margin-top:8px'>"
            "<div style='font-size:0.6rem;color:#22C55E;letter-spacing:1px'>GAME PLAN</div>"
            "<div style='font-size:0.78rem;color:#F5F5F5;margin-top:3px;line-height:1.5'>" + plan + "</div>"
            "</div>"
        )

    invalid_html = ""
    if invalid:
        invalid_html = (
            "<div style='background:#1a0a0a;border-left:3px solid #C1121F;border-radius:4px;"
            "padding:8px 12px;margin-top:6px'>"
            "<div style='font-size:0.6rem;color:#C1121F;letter-spacing:1px'>WHEN TO BAIL</div>"
            "<div style='font-size:0.78rem;color:#F5F5F5;margin-top:3px;line-height:1.5'>" + invalid + "</div>"
            "</div>"
        )

    return (
        "<div style='background:#0B0B0C;border:1px solid #2A2A2D;border-radius:10px;"
        "padding:12px 14px;margin-top:10px'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;"
        "margin-bottom:6px'>"
        "<span style='color:#D4AF37;font-family:monospace;font-size:0.68rem;"
        "letter-spacing:2px;font-weight:700'>🎯 CONFLUENCE INTEL</span>"
        "<span style='color:#4a5568;font-size:0.62rem'>Levels as of " + ts + "</span>"
        "</div>"
        + d_chip + i_chip + lines_html + zone_html + plan_html + invalid_html +
        "</div>"
    )

def render_time_of_day_banner_html():
    """Top-of-page banner showing what time-of-day means for trading."""
    t = get_time_of_day_context()
    return (
        "<div style='background:#0d0d0f;border:1px solid " + t["color"] + "44;border-left:3px solid " + t["color"] + ";"
        "border-radius:8px;padding:8px 14px;margin-bottom:8px'>"
        "<div style='display:flex;justify-content:space-between;align-items:center'>"
        "<div>"
        "<span style='font-size:0.95rem'>" + t["icon"] + "</span>"
        "<span style='color:" + t["color"] + ";font-weight:700;font-size:0.78rem;margin-left:8px'>" + t["label"] + "</span>"
        "</div>"
        "</div>"
        "<div style='font-size:0.72rem;color:#A1A1A6;margin-top:4px;line-height:1.4'>" + t["meaning"] + "</div>"
        "</div>"
    )

def render_perf_chip_html(pattern, style, direction):
    """Small chip showing historical win rate for this pattern.
    Only renders if we have enough sample size."""
    try:
        stats = get_pattern_win_rate(pattern, style=style, direction=direction)
        if not stats:
            return ""
        if not stats.get("ready"):
            n = stats.get("sample_size", 0)
            need = stats.get("min_sample", 10)
            return (
                "<div style='background:#1A1A1D;border:1px dashed #4a5568;border-radius:6px;"
                "padding:6px 12px;margin:6px 0;font-size:0.7rem;color:#A1A1A6'>"
                "* Building dataset for this pattern (" + str(n) + " of " + str(need) + " trades). "
                "Win rate unlocks at " + str(need) + "+."
                "</div>"
            )

        wr   = stats["win_rate"]
        n    = stats["sample_size"]
        exp  = stats["expectancy"]
        avgW = stats["avg_winner"]
        avgL = stats["avg_loser"]

        if wr >= 60:    color = "#22C55E"; tier = "Strong edge"
        elif wr >= 50:  color = "#D4AF37"; tier = "Slight edge"
        elif wr >= 40:  color = "#F6E27A"; tier = "Coin flip"
        else:           color = "#C1121F"; tier = "Negative edge"

        return (
            "<div style='background:#0d0d0f;border:1px solid " + color + ";border-radius:8px;"
            "padding:8px 12px;margin:6px 0'>"
            "<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<span style='color:" + color + ";font-family:monospace;font-size:0.66rem;"
            "letter-spacing:1.5px;font-weight:700'>* HISTORICAL EDGE</span>"
            "<span style='color:" + color + ";font-weight:700;font-size:0.78rem'>"
            + str(wr) + "% wins (" + str(n) + " trades)</span>"
            "</div>"
            "<div style='font-size:0.7rem;color:#F5F5F5;margin-top:3px'>"
            + tier + " * Avg winner +" + str(avgW) + "% * Avg loser " + str(avgL) + "% * "
            "Expected per trade: " + ("+" if exp >= 0 else "") + str(exp) + "%"
            "</div>"
            "</div>"
        )
    except Exception:
        return ""

def render_market_stress_html(s):
    if not s or not s.get("available"):
        return ""

    rsi_d = s.get("spy_rsi_1d")
    rsi_w = s.get("spy_rsi_1w")
    pct5  = s.get("spy_pct_5d")
    sup   = s.get("support_level")
    dist  = s.get("dist_to_sup")
    vix   = s.get("vix_est")
    def_h = s.get("def_holding", 0)
    iwm_nl= s.get("iwm_new_low")
    spy_nl= s.get("spy_new_low")
    note  = s.get("note", "")
    price = s.get("spy_price")

    def rsi_color(r):
        if r is None: return "#A1A1A6"
        if r < 25:    return "#C1121F"
        if r < 35:    return "#FF6B35"
        if r < 45:    return "#F6E27A"
        return "#22C55E"

    def stat_block(label, val, color, sub=""):
        return (
            "<div style='background:#111115;border-radius:6px;padding:8px 10px'>"
            "<div style='font-size:0.58rem;color:#A1A1A6;letter-spacing:1px'>%s</div>"
            "<div style='font-size:1.0rem;font-weight:700;color:%s'>%s</div>"
            "%s"
            "</div>"
        ) % (label, color,
             val if val is not None else "-",
             ("<div style='font-size:0.62rem;color:#A1A1A6;margin-top:1px'>%s</div>" % sub) if sub else "")

    # RSI blocks
    rsi_d_str = ("%.0f" % rsi_d) if rsi_d else "-"
    rsi_w_str = ("%.0f" % rsi_w) if rsi_w else "-"
    rsi_d_sub = "Oversold" if rsi_d and rsi_d < 30 else "Approaching" if rsi_d and rsi_d < 38 else "Elevated"
    rsi_w_sub = "Oversold" if rsi_w and rsi_w < 35 else "Elevated" if rsi_w and rsi_w < 45 else "OK"

    # 5-day move
    pct5_color = "#C1121F" if pct5 and pct5 < -3 else "#FF6B35" if pct5 and pct5 < 0 else "#22C55E"
    pct5_str = ("%+.1f%%" % pct5) if pct5 is not None else "-"

    # Support
    sup_color = "#22C55E" if s.get("at_support") else "#D4AF37" if dist and dist < 2 else "#A1A1A6"
    sup_str   = ("$%.2f" % sup) if sup else "-"
    sup_sub   = "AT SUPPORT" if s.get("at_support") else ("%.1f%% above" % dist if dist else "")

    # VIX proxy
    vix_color = "#C1121F" if vix and vix > 30 else "#F6E27A" if vix and vix > 20 else "#22C55E"
    vix_str   = ("%.1f" % vix) if vix else "-"
    vix_sub   = "proxy via VIXY"

    # Breadth row
    breadth_items = []
    if iwm_nl is not None:
        iwm_color = "#22C55E" if not iwm_nl else "#C1121F"
        iwm_label = "IWM: no new low" if not iwm_nl else "IWM: new 20d low"
        breadth_items.append("<span style='color:%s;font-size:0.72rem'>%s</span>" % (iwm_color, iwm_label))
    if def_h > 0:
        def_color = "#22C55E" if def_h >= 2 else "#F6E27A"
        breadth_items.append("<span style='color:%s;font-size:0.72rem'>%d/3 defensives holding</span>" % (def_color, def_h))

    breadth_html = ""
    if breadth_items:
        breadth_html = (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;padding-top:8px;"
            "border-top:1px solid #2A2A2D'>"
            + "  ".join(breadth_items) +
            "</div>"
        )

    note_html = ""
    if note:
        note_html = (
            "<div style='font-size:0.72rem;color:#A1A1A6;margin-top:8px;"
            "padding-top:8px;border-top:1px solid #2A2A2D;line-height:1.5'>"
            + note +
            "</div>"
        )

    spy_str = ("SPY $%.2f" % price) if price else "SPY"

    return (
        "<div style='background:#0d0d0f;border:1px solid #2A2A2D;border-left:3px solid #D4AF37;"
        "border-radius:10px;padding:12px 16px;margin-bottom:10px'>"
        "<div style='color:#D4AF37;font-family:monospace;font-size:0.68rem;"
        "letter-spacing:2px;font-weight:700;margin-bottom:10px'>"
        "MARKET STRESS MONITOR  "
        "<span style='color:#4a5568;font-weight:400;font-size:0.62rem'>"
        + spy_str +
        "  |  no predictions, just data</span>"
        "</div>"
        "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px'>"
        + stat_block("DAILY RSI", rsi_d_str, rsi_color(rsi_d), rsi_d_sub)
        + stat_block("WEEKLY RSI", rsi_w_str, rsi_color(rsi_w), rsi_w_sub)
        + stat_block("5-DAY MOVE", pct5_str, pct5_color, "SPY")
        + stat_block("KEY SUPPORT", sup_str, sup_color, sup_sub)
        + stat_block("VIX PROXY", vix_str, vix_color, vix_sub)
        + "</div>"
        + breadth_html
        + note_html
        + "</div>"
    )

def render_macro_event_warning_html(w):
    """Renders a warning banner when high-impact events are today."""
    if not w or w.get("level") == "CLEAR":
        return ""
    level   = w["level"]
    message = w["message"]
    events  = w["events"]

    if level == "PRE_EVENT":
        color  = "#FF6B35"
        label  = "MACRO EVENT ALERT"
        bg     = "#1a0f00"
        border = "#FF6B35"
        icon   = "!"
    else:
        color  = "#F6E27A"
        label  = "MACRO EVENT PRINTED"
        bg     = "#1a1800"
        border = "#F6E27A"
        icon   = "*"

    event_chips = ""
    for ev in events[:3]:
        actual_str = ""
        if ev.get("actual") and ev.get("estimate"):
            try:
                act = float(str(ev["actual"]).replace("%","").replace("K","").replace("M",""))
                est = float(str(ev["estimate"]).replace("%","").replace("K","").replace("M",""))
                surprise_color = "#C1121F" if act > est else "#22C55E"
                actual_str = (
                    " <span style='color:%s;font-size:0.65rem'>actual %.2f</span>"
                    " <span style='color:#A1A1A6;font-size:0.65rem'>vs %.2f est</span>"
                ) % (surprise_color, act, est)
            except Exception:
                pass
        event_chips += (
            "<div style='font-size:0.72rem;color:#F5F5F5;margin:2px 0'>"
            "- " + ev["name"] + actual_str +
            "</div>"
        )

    return (
        "<div style='background:%s;border:1px solid %s;border-left:3px solid %s;"
        "border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
        "<div style='color:%s;font-family:monospace;font-size:0.68rem;"
        "letter-spacing:2px;font-weight:700;margin-bottom:6px'>"
        "%s  %s"
        "</div>"
        "<div style='font-size:0.75rem;color:#F5F5F5;line-height:1.5;margin-bottom:6px'>"
        + message +
        "</div>"
        + event_chips +
        "</div>"
    ) % (bg, border, border, color, icon, label)
# PROPER SWING HIGH / SWING LOW DETECTION
# Replaces the naive max/min approach with structural pivot detection.
#
# A real swing high:
#   - At least N candles on BOTH sides with lower highs
#   - Minimum size threshold (avoids noise spikes)
#   - Confirmed — not just the highest point in a window
#
# A real swing low:
#   - At least N candles on BOTH sides with higher lows
#   - Minimum size threshold
#   - Confirmed

def detect_swing_points(df, n_confirm=3, min_swing_pct=0.005):
    """
    Find confirmed structural swing highs and lows.

    Args:
        df: OHLCV dataframe (must have 'high', 'low', 'close')
        n_confirm: candles required on EACH side to confirm a swing (default 3)
        min_swing_pct: minimum swing size as % of price to filter noise (default 0.5%)

    Returns:
        dict with:
            swing_highs: list of (index, price) tuples, most recent last
            swing_lows:  list of (index, price) tuples, most recent last
            last_swing_high: (index, price) or None
            last_swing_low:  (index, price) or None
    """
    result = {
        "swing_highs":      [],
        "swing_lows":       [],
        "last_swing_high":  None,
        "last_swing_low":   None,
        "available":        False,
    }

    try:
        if df is None or len(df) < (n_confirm * 2 + 3):
            return result

        highs  = df["high"].astype(float).values
        lows   = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        n      = len(highs)
        avg_price = float(closes[-1])
        min_size  = avg_price * min_swing_pct

        swing_highs = []
        swing_lows  = []

        # Check each candle (excluding first and last n_confirm candles)
        for i in range(n_confirm, n - n_confirm):
            h = highs[i]
            l = lows[i]

            # === Swing High check ===
            # Left side: all n_confirm candles before must have lower highs
            left_ok  = all(highs[i-j] < h for j in range(1, n_confirm+1))
            # Right side: all n_confirm candles after must have lower highs
            right_ok = all(highs[i+j] < h for j in range(1, n_confirm+1))

            if left_ok and right_ok:
                # Check it's a meaningful swing, not a tiny pip
                left_low  = min(highs[max(0, i-n_confirm):i])
                swing_size = h - left_low
                if swing_size >= min_size:
                    swing_highs.append((i, round(h, 2)))

            # === Swing Low check ===
            left_ok_l  = all(lows[i-j] > l for j in range(1, n_confirm+1))
            right_ok_l = all(lows[i+j] > l for j in range(1, n_confirm+1))

            if left_ok_l and right_ok_l:
                right_high = max(lows[i:min(n, i+n_confirm)])
                swing_size_l = right_high - l
                if swing_size_l >= min_size:
                    swing_lows.append((i, round(l, 2)))

        result["swing_highs"] = swing_highs
        result["swing_lows"]  = swing_lows

        if swing_highs:
            result["last_swing_high"] = swing_highs[-1]
        if swing_lows:
            result["last_swing_low"] = swing_lows[-1]

        result["available"] = bool(swing_highs and swing_lows)
        return result

    except Exception as _e:
        print("[swing_detect] error: %s" % str(_e)[:120])
        return result


def find_most_recent_breakout_move(df, direction, n_confirm=3, min_swing_pct=0.005):
    """
    Find the most recent clean breakout move to draw the fib on.

    For bullish:
        Most recent significant swing low → most recent significant swing high
        (Price moved up from the low — we measure the retrace of that move)

    For bearish:
        Most recent significant swing high → most recent significant swing low

    Returns:
        dict with move_start (price), move_end (price), move_pct, valid (bool)
    """
    result = {
        "valid":        False,
        "move_start":   None,   # where the move came FROM
        "move_end":     None,   # where the move went TO
        "move_pct":     None,   # size of the move in %
        "move_dollars": None,   # size in dollars
        "start_idx":    None,
        "end_idx":      None,
    }

    try:
        swings = detect_swing_points(df, n_confirm=n_confirm, min_swing_pct=min_swing_pct)
        if not swings["available"]:
            return result

        highs = swings["swing_highs"]
        lows  = swings["swing_lows"]

        if direction == "bullish":
            # Find the most recent swing low that came BEFORE the most recent swing high
            # That's the move we're drawing fib on
            if not highs or not lows:
                return result

            last_high_idx, last_high_price = highs[-1]

            # Find the swing low that preceded this high (lower index)
            preceding_lows = [(i, p) for i, p in lows if i < last_high_idx]
            if not preceding_lows:
                return result

            # Use the most recent preceding low
            low_idx, low_price = preceding_lows[-1]

            move_dollars = last_high_price - low_price
            move_pct     = (move_dollars / low_price) * 100

            # Minimum move size — avoid drawing fib on tiny wiggles
            # At least 1.5% move to qualify
            if move_pct < 0.5:
                return result

            result.update({
                "valid":        True,
                "move_start":   round(low_price, 2),   # bottom of the move
                "move_end":     round(last_high_price, 2), # top of the move
                "move_pct":     round(move_pct, 2),
                "move_dollars": round(move_dollars, 2),
                "start_idx":    low_idx,
                "end_idx":      last_high_idx,
            })

        else:  # bearish
            if not highs or not lows:
                return result

            last_low_idx, last_low_price = lows[-1]

            # Find the swing high that preceded this low
            preceding_highs = [(i, p) for i, p in highs if i < last_low_idx]
            if not preceding_highs:
                return result

            high_idx, high_price = preceding_highs[-1]

            move_dollars = high_price - last_low_price
            move_pct     = (move_dollars / high_price) * 100

            if move_pct < 0.5:
                return result

            result.update({
                "valid":        True,
                "move_start":   round(high_price, 2),  # top of the move
                "move_end":     round(last_low_price, 2), # bottom
                "move_pct":     round(move_pct, 2),
                "move_dollars": round(move_dollars, 2),
                "start_idx":    high_idx,
                "end_idx":      last_low_idx,
            })

        return result

    except Exception as _e:
        print("[breakout_move] error: %s" % str(_e)[:120])
        return result


def classify_fib_retrace(move, current_price, direction):
    """
    Given a confirmed breakout move and the current price,
    classify where price is in the fib retrace.

    Returns:
        dict with level (str), pct (float), verdict (str), color (str),
        meaning (str), action (str), trade_valid (bool)
    """
    result = {
        "valid":       False,
        "level":       None,
        "pct":         None,
        "verdict":     "",
        "color":       "#A1A1A6",
        "meaning":     "",
        "action":      "",
        "trade_valid": True,
        "fib_levels":  {},
    }

    try:
        if not move or not move.get("valid"):
            return result

        move_start = move["move_start"]
        move_end   = move["move_end"]
        move_size  = abs(move_end - move_start)

        if move_size < 0.01:
            return result

        # Calculate all fib levels
        # For bullish: retrace is measured from move_end (top) DOWN toward move_start (bottom)
        # For bearish: retrace is measured from move_end (bottom) UP toward move_start (top)
        if direction == "bullish":
            fib_levels = {
                "0.0%":   move_end,
                "23.6%":  move_end - (move_size * 0.236),
                "38.2%":  move_end - (move_size * 0.382),
                "50.0%":  move_end - (move_size * 0.500),
                "61.8%":  move_end - (move_size * 0.618),
                "78.6%":  move_end - (move_size * 0.786),
                "100.0%": move_start,
            }
            # How far has price retraced from the top?
            retrace_amount = move_end - current_price
        else:
            fib_levels = {
                "0.0%":   move_end,
                "23.6%":  move_end + (move_size * 0.236),
                "38.2%":  move_end + (move_size * 0.382),
                "50.0%":  move_end + (move_size * 0.500),
                "61.8%":  move_end + (move_size * 0.618),
                "78.6%":  move_end + (move_size * 0.786),
                "100.0%": move_start,
            }
            retrace_amount = current_price - move_end

        retrace_pct = (retrace_amount / move_size) * 100 if move_size > 0 else 0
        retrace_pct = max(0, retrace_pct)  # no negative retrace

        result["fib_levels"] = {k: round(v, 2) for k, v in fib_levels.items()}
        result["pct"] = round(retrace_pct, 1)

        # Classify the retrace level
        if retrace_pct <= 5:
            result.update({
                "valid":       True,
                "level":       "At Highs",
                "verdict":     "No meaningful retrace yet",
                "color":       "#22C55E",
                "meaning":     "Price hasn't pulled back at all. Momentum is strong but don't chase — wait for a pullback entry.",
                "action":      "Wait for a pullback to the .236 or .382 zone before entering.",
                "trade_valid": True,
            })
        elif retrace_pct <= 27:
            result.update({
                "valid":       True,
                "level":       ".236",
                "verdict":     "Shallow Pullback",
                "color":       "#22C55E",
                "meaning":     "Price barely pulled back. Trend is very strong — buyers stepped in immediately.",
                "action":      "Valid entry zone. The move has plenty of room left.",
                "trade_valid": True,
            })
        elif retrace_pct <= 43:
            result.update({
                "valid":       True,
                "level":       ".382",
                "verdict":     "Normal Pullback",
                "color":       "#D4AF37",
                "meaning":     "Classic healthy pullback. Trend is intact. This is the textbook entry zone.",
                "action":      "Strong entry zone. Most trending moves bounce from here and continue.",
                "trade_valid": True,
            })
        elif retrace_pct <= 55:
            result.update({
                "valid":       True,
                "level":       ".500",
                "verdict":     "Trend Weakening",
                "color":       "#FF6B35",
                "meaning":     "Price has retraced half the move. The trend is losing conviction. Bulls and bears are balanced here.",
                "action":      "Caution. If price cannot bounce strongly from here, the trend is likely over. Quick trades only — no swings.",
                "trade_valid": True,  # still tradeable but with caution
            })
        elif retrace_pct <= 70:
            result.update({
                "valid":       True,
                "level":       ".618",
                "verdict":     "Trend Gone — Wait for Reset",
                "color":       "#C1121F",
                "meaning":     "Price has retraced 61.8% of the move. The original trend is over. Smart money has exited.",
                "action":      "Do not trade in the original direction. Wait for a new structure to form — a new confirmed high or low before re-entering.",
                "trade_valid": False,
            })
        else:
            result.update({
                "valid":       True,
                "level":       "Full Retrace",
                "verdict":     "Full Reversal — New Direction Forming",
                "color":       "#C1121F",
                "meaning":     "Price has given back nearly the entire move. This is no longer a pullback — it is a reversal. The thesis has changed.",
                "action":      "Stay out until a new swing structure forms. A new confirmed high (bullish) or new confirmed low (bearish) is your signal to re-enter.",
                "trade_valid": False,
            })

        return result

    except Exception as _e:
        print("[fib_classify] error: %s" % str(_e)[:120])
        return result


def detect_multi_timeframe_fib(ticker, current_price, direction, style):
    """
    Run fib detection across the correct timeframes for the trade style.

    Quick  → 1-hour fib (last 5 days of hourly data)
    Swing  → Daily fib  (last 60 days of daily data)
    Leap   → Weekly fib (last 2 years of weekly data)

    Returns dict with fib reads for each relevant timeframe.
    """
    result = {
        "available": False,
        "intraday":  None,   # 1H fib (for quick trades)
        "daily":     None,   # Daily fib (for swing trades)
        "weekly":    None,   # Weekly fib (for leaps)
        "verdict":   "",     # plain English synthesis
        "trade_valid": True, # overall — if ANY higher TF says no, this is False
        "conflict":  False,  # timeframes disagree
        "conflict_note": "",
    }

    try:
        # Pull data for each timeframe
        df_1h = None
        df_1d = None
        df_1w = None

        try:
            df_1h = _fmp_download(ticker, "10d", "1h")
        except Exception:
            pass

        try:
            df_1d = _fmp_download(ticker, "90d", "1d")
        except Exception:
            pass

        try:
            df_1w = _fmp_download(ticker, "2y", "1wk")
        except Exception:
            pass

        # Run fib on each
        any_available = False

        if df_1h is not None and len(df_1h) >= 10 and style in ("quick", "swing", "leap"):
            move_1h = find_most_recent_breakout_move(df_1h, direction, n_confirm=2, min_swing_pct=0.001)
            if move_1h["valid"]:
                fib_1h = classify_fib_retrace(move_1h, current_price, direction)
                fib_1h["move"] = move_1h
                result["intraday"] = fib_1h
                any_available = True

        if df_1d is not None and len(df_1d) >= 20 and style in ("quick", "swing", "leap"):
            move_1d = find_most_recent_breakout_move(df_1d, direction, n_confirm=3, min_swing_pct=0.01)
            if move_1d["valid"]:
                fib_1d = classify_fib_retrace(move_1d, current_price, direction)
                fib_1d["move"] = move_1d
                result["daily"] = fib_1d
                any_available = True

        if df_1w is not None and len(df_1w) >= 20 and style == "leap":
            move_1w = find_most_recent_breakout_move(df_1w, direction, n_confirm=2, min_swing_pct=0.02)
            if move_1w["valid"]:
                fib_1w = classify_fib_retrace(move_1w, current_price, direction)
                fib_1w["move"] = move_1w
                result["weekly"] = fib_1w
                any_available = True

        result["available"] = any_available

        if not any_available:
            return result

        # === SYNTHESIS — determine overall verdict ===
        reads = []
        if result["intraday"]: reads.append(("Intraday", result["intraday"]))
        if result["daily"]:    reads.append(("Daily",    result["daily"]))
        if result["weekly"]:   reads.append(("Weekly",   result["weekly"]))

        # If ANY higher timeframe says trend gone — overall is invalid for swing/leap
        higher_tf_broken = False
        if result["daily"] and not result["daily"]["trade_valid"]:
            higher_tf_broken = True
        if result["weekly"] and not result["weekly"]["trade_valid"]:
            higher_tf_broken = True

        result["trade_valid"] = not higher_tf_broken

        # Check for conflict — intraday valid but daily broken
        if (result["intraday"] and result["intraday"]["trade_valid"] and
            result["daily"] and not result["daily"]["trade_valid"]):
            result["conflict"] = True
            result["conflict_note"] = (
                "Short-term (hourly) shows a valid pullback but the daily trend has broken down. "
                "A quick bounce toward $%.2f is possible but the bigger move is gone. "
                "If you trade this — quick in and out only, no overnight holds. "
                "Target the hourly resistance, take profit, move on." % (
                    result["intraday"].get("move", {}).get("move_end", current_price)
                )
            )

        # Build overall verdict string
        if higher_tf_broken and result["conflict"]:
            result["verdict"] = "TIMEFRAME CONFLICT — Quick only"
        elif higher_tf_broken:
            result["verdict"] = "TREND GONE — Wait for reset"
        elif all(r.get("level") in (".236", "At Highs") for _, r in reads):
            result["verdict"] = "ALL TIMEFRAMES STRONG — High conviction entry"
        elif all(r.get("level") in (".236", ".382", "At Highs") for _, r in reads):
            result["verdict"] = "ALIGNED — Entry valid across timeframes"
        elif all(r.get("trade_valid", True) for _, r in reads):
            result["verdict"] = "MIXED — Entry valid but watch the levels"
        else:
            result["verdict"] = "CAUTION — Timeframes not aligned"

        return result

    except Exception as _e:
        print("[multi_tf_fib] error: %s" % str(_e)[:120])
        return result


# FEATURE 1: VOLATILITY CLASSIFIER + PREDICTED MOVE + STRIKE GUIDANCE

@_thread_cache(ttl=1800)
def classify_stock_volatility(ticker, current_price=None):
    """
    Classify stock as LOW / MODERATE / HIGH mover based on ATR% of price.
    Returns dollar range, % classification, and DTE/delta guidance.
    """
    result = {
        "available":      False,
        "tier":           "MODERATE",
        "atr_dollar":     None,
        "atr_pct":        None,
        "daily_range_lo": None,
        "daily_range_hi": None,
        "label":          "",
        "dte_min":        21,
        "dte_max":        45,
        "delta_lo":       0.35,
        "delta_hi":       0.55,
        "guidance":       "",
    }
    ETF_TICKERS = {
        'SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLY','XLI',
        'GLD','SLV','TLT','HYG','IBIT','VXX','UVXY','SQQQ','TQQQ','SPXL','SPXU',
    }
    is_etf = ticker.upper() in ETF_TICKERS

    try:
        df = _fmp_download(ticker, "30d", "1d")
        if df is None or len(df) < 14:
            return result

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        tr  = (high - low).copy()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = tr.combine(tr2, max).combine(tr3, max)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        price = current_price or float(close.iloc[-1])
        atr_pct = (atr_14 / price) * 100

        result["atr_dollar"]     = round(atr_14, 2)
        result["daily_range_lo"] = round(atr_14 * 0.7, 2)
        result["daily_range_hi"] = round(atr_14 * 1.3, 2)
        result["atr_pct"]        = round(atr_pct, 2)

        if is_etf:
            result.update({
                "tier": "ETF", "label": "INDEX ETF",
                "dte_min": 0, "dte_max": 30, "delta_lo": 0.40, "delta_hi": 0.65,
                "guidance": (
                    "Index ETF \u2014 moves $%.2f/day (%.1f%%). "
                    "0DTE is heavily traded here and perfectly valid. "
                    "Same-day plays: 0-5 DTE, delta 0.40-0.65. "
                    "Multi-day swings: 21-30 DTE." % (atr_14, atr_pct)
                ),
            })
            result["available"] = True
            return result

        # Classify by ATR% internally, display in dollars externally
        if atr_pct < 1.5:
            result.update({
                "tier":      "LOW",
                "label":     "LOW MOVER",
                "dte_min":   30,
                "dte_max":   60,
                "delta_lo":  0.40,
                "delta_hi":  0.65,
                "guidance":  (
                    "This stock moves about $%.2f/day (%.1f%% of price). "
                    "It needs time — short DTE options will decay before the move develops. "
                    "Target 30-60 DTE and a delta of 0.40-0.65 so your option moves "
                    "meaningfully when the stock does." % (atr_14, atr_pct)
                ),
            })
        elif atr_pct < 3.0:
            result.update({
                "tier":      "MODERATE",
                "label":     "MODERATE MOVER",
                "dte_min":   21,
                "dte_max":   45,
                "delta_lo":  0.35,
                "delta_hi":  0.55,
                "guidance":  (
                    "This stock moves about $%.2f/day (%.1f%% of price). "
                    "Standard options approach — 21-45 DTE, 0.35-0.55 delta. "
                    "Room to catch the move without overpaying for time." % (atr_14, atr_pct)
                ),
            })
        else:
            result.update({
                "tier":      "HIGH",
                "label":     "HIGH MOVER",
                "dte_min":   7,
                "dte_max":   21,
                "delta_lo":  0.25,
                "delta_hi":  0.50,
                "guidance":  (
                    "This stock moves about $%.2f/day (%.1f%% of price). "
                    "Moves fast — shorter DTE (7-21) is fine since the move happens quickly. "
                    "Lower delta (0.25-0.50) gives you better leverage on the big moves." % (atr_14, atr_pct)
                ),
            })

        result["available"] = True
        return result
    except Exception as _e:
        print("[vol_classify] error: %s" % str(_e)[:100])
        return result


def calc_predicted_move(ticker, current_price, direction, atr, sq_state, sq_compression, block_detected, style):
    """
    Predict today's move range and the next 15-min candle projection.
    Two separate outputs as requested.
    Returns dict with daily_lo, daily_hi, candle_lo, candle_hi, candle_time, confidence.
    """
    result = {
        "available":       False,
        "daily_lo":        None,
        "daily_hi":        None,
        "daily_lo_price":  None,
        "daily_hi_price":  None,
        "candle_proj":     None,
        "candle_lo":       None,
        "candle_hi":       None,
        "candle_time":     None,
        "candle_state":    None,  # WAITING / CURRENT_IS_ENTRY / WATCH_CLOSE
        "candle_minutes_left": None,
        "candle_time_range":   None,
        "confirm_level":   None,
        "confidence":      "MODERATE",
        "scan_time":       None,
    }
    try:
        if not atr or atr <= 0:
            return result

        # === DAILY MOVE PREDICTION ===
        # Base multiplier from ATR
        multiplier = 1.0

        # Squeeze modifier
        if sq_state == "firing" and sq_compression >= 60:
            multiplier = 1.55
            result["confidence"] = "HIGH"
        elif sq_state == "firing":
            multiplier = 1.35
            result["confidence"] = "HIGH"
        elif sq_state == "squeeze" and sq_compression >= 50:
            multiplier = 1.20
            result["confidence"] = "MODERATE-HIGH"
        elif sq_state == "squeeze":
            multiplier = 1.10
            result["confidence"] = "MODERATE"

        # Block modifier (institutional footprint)
        if block_detected:
            multiplier *= 1.15
            if result["confidence"] == "MODERATE":
                result["confidence"] = "MODERATE-HIGH"

        # Both firing together
        if sq_state == "firing" and block_detected:
            result["confidence"] = "HIGH"

        predicted_move = atr * multiplier
        daily_lo = predicted_move * 0.65
        daily_hi = predicted_move * 1.35

        result["daily_lo"] = round(daily_lo, 2)
        result["daily_hi"] = round(daily_hi, 2)

        if direction == "bullish":
            result["daily_lo_price"] = round(current_price + daily_lo, 2)
            result["daily_hi_price"] = round(current_price + daily_hi, 2)
        else:
            result["daily_lo_price"] = round(current_price - daily_hi, 2)
            result["daily_hi_price"] = round(current_price - daily_lo, 2)

        # === NEXT 15-MIN CANDLE PROJECTION ===
        try:
            et = pytz.timezone("America/New_York")
            now = datetime.now(et)
            current_minute = now.hour * 60 + now.minute

            # Find which 15-min candle we're in
            market_open_min = 9 * 60 + 30   # 9:30 AM
            mins_since_open = current_minute - market_open_min

            if mins_since_open < 0:
                # Pre-market
                candle_start_min = market_open_min
                mins_into_candle = 0
            else:
                candle_num = mins_since_open // 15
                candle_start_min = market_open_min + (candle_num * 15)
                mins_into_candle = mins_since_open % 15

            mins_left = 15 - mins_into_candle
            candle_end_min = candle_start_min + 15

            def fmt_time(total_minutes):
                h = total_minutes // 60
                m = total_minutes % 60
                ampm = "AM" if h < 12 else "PM"
                h12 = h if h <= 12 else h - 12
                if h12 == 0: h12 = 12
                return "%d:%02d %s" % (h12, m, ampm)

            candle_start_str = fmt_time(candle_start_min)
            candle_end_str   = fmt_time(candle_end_min)
            next_candle_str  = fmt_time(candle_end_min)

            result["candle_time_range"] = "%s - %s ET" % (candle_start_str, candle_end_str)
            result["candle_minutes_left"] = mins_left
            result["scan_time"] = now.strftime("%I:%M %p ET")

            # Pull 15-min data for candle size projection
            df_15m = _fmp_download(ticker, "5d", "15min")
            if df_15m is not None and len(df_15m) >= 10:
                c15 = df_15m["close"].astype(float)
                h15 = df_15m["high"].astype(float)
                l15 = df_15m["low"].astype(float)
                v15 = df_15m["volume"].astype(float)

                # Average 15-min candle range
                avg_candle_range = float((h15 - l15).rolling(10).mean().iloc[-1])

                # Volume ratio on most recent candle
                avg_vol_15 = float(v15.iloc[-10:].mean())
                cur_vol_15 = float(v15.iloc[-1])
                vol_ratio_15 = cur_vol_15 / avg_vol_15 if avg_vol_15 > 0 else 1.0

                # Projected next candle range
                projected_range = avg_candle_range * min(vol_ratio_15, 2.0)
                candle_lo_proj  = projected_range * 0.6
                candle_hi_proj  = projected_range * 1.4

                result["candle_proj"]  = round(projected_range, 2)
                result["candle_lo"]    = round(candle_lo_proj, 2)
                result["candle_hi"]    = round(candle_hi_proj, 2)

                # Determine candle state and confirm level
                last_15m_close = float(c15.iloc[-1])
                last_15m_high  = float(h15.iloc[-1])
                last_15m_low   = float(l15.iloc[-1])

                if direction == "bullish":
                    confirm_level = round(last_15m_high + (avg_candle_range * 0.1), 2)
                else:
                    confirm_level = round(last_15m_low - (avg_candle_range * 0.1), 2)
                result["confirm_level"] = confirm_level

                # State logic
                if mins_into_candle <= 2:
                    result["candle_state"] = "NEW_CANDLE"
                    result["candle_time"] = (
                        "New candle just opened (%s). Give it 3-5 minutes to show direction. "
                        "Watch for price to %s $%.2f — that confirms the move is on." % (
                            candle_start_str,
                            "hold above" if direction == "bullish" else "hold below",
                            confirm_level
                        )
                    )
                elif mins_left <= 3:
                    result["candle_state"] = "WATCH_CLOSE"
                    result["candle_time"] = (
                        "Current candle (%s) has %d minute%s left. "
                        "If it closes %s $%.2f the next candle (%s) is your entry. "
                        "This is the decision point." % (
                            result["candle_time_range"],
                            mins_left,
                            "s" if mins_left != 1 else "",
                            "above" if direction == "bullish" else "below",
                            confirm_level,
                            next_candle_str
                        )
                    )
                elif mins_into_candle >= 7 and vol_ratio_15 >= 1.3:
                    result["candle_state"] = "CURRENT_IS_ENTRY"
                    result["candle_time"] = (
                        "We are %d minutes into the %s candle and volume is %.1fx average. "
                        "Price is %s $%.2f — this IS the entry candle. "
                        "Enter now or on any dip back to $%.2f." % (
                            mins_into_candle,
                            result["candle_time_range"],
                            vol_ratio_15,
                            "holding above" if direction == "bullish" else "holding below",
                            confirm_level,
                            confirm_level
                        )
                    )
                else:
                    result["candle_state"] = "WAITING"
                    result["candle_time"] = (
                        "Current candle: %s | %d minutes left. "
                        "Watch for a close %s $%.2f to confirm entry on the next candle (%s)." % (
                            result["candle_time_range"],
                            mins_left,
                            "above" if direction == "bullish" else "below",
                            confirm_level,
                            next_candle_str
                        )
                    )
        except Exception:
            pass

        result["available"] = True
        return result

    except Exception as _e:
        print("[predicted_move] error: %s" % str(_e)[:120])
        return result


def calc_strike_guidance(vol_class, predicted_move, current_price, direction):
    """
    Generate 3 strike guidance options (Conservative / Balanced / Aggressive)
    based on volatility classification and predicted move.
    Uses Black-Scholes delta approximation — no live options chain needed.
    """
    result = {
        "available": False,
        "options":   [],  # list of 3 dicts
        "summary":   "",
    }
    try:
        if not vol_class.get("available") or not predicted_move.get("available"):
            return result

        tier    = vol_class["tier"]
        delta_lo = vol_class["delta_lo"]
        delta_hi = vol_class["delta_hi"]
        dte_min  = vol_class["dte_min"]
        dte_max  = vol_class["dte_max"]
        move_lo  = predicted_move["daily_lo"]
        move_hi  = predicted_move["daily_hi"]

        # Three options — Conservative, Balanced, Aggressive
        options = [
            {
                "label":     "Conservative",
                "delta_lo":  round(delta_lo - 0.05, 2),
                "delta_hi":  round(delta_lo + 0.05, 2),
                "dte_lo":    dte_max - 5,
                "dte_hi":    dte_max + 15,
                "why":       "Lower premium cost, more time for the move to develop. Lower risk, lower reward.",
                "gain_lo":   round(move_lo * (delta_lo - 0.05) * 100, 0),
                "gain_hi":   round(move_hi * (delta_lo + 0.05) * 100, 0),
            },
            {
                "label":     "Balanced",
                "delta_lo":  round((delta_lo + delta_hi) / 2 - 0.05, 2),
                "delta_hi":  round((delta_lo + delta_hi) / 2 + 0.05, 2),
                "dte_lo":    dte_min + 5,
                "dte_hi":    dte_max,
                "why":       "Best risk/reward balance. Enough delta to profit from the predicted move, enough time to not panic.",
                "gain_lo":   round(move_lo * ((delta_lo + delta_hi) / 2 - 0.05) * 100, 0),
                "gain_hi":   round(move_hi * ((delta_lo + delta_hi) / 2 + 0.05) * 100, 0),
            },
            {
                "label":     "Aggressive",
                "delta_lo":  round(delta_hi - 0.05, 2),
                "delta_hi":  round(delta_hi + 0.05, 2),
                "dte_lo":    dte_min,
                "dte_hi":    dte_min + 10,
                "why":       "Higher premium, faster decay. Only works if the move happens quickly. Higher reward, higher risk.",
                "gain_lo":   round(move_lo * (delta_hi - 0.05) * 100, 0),
                "gain_hi":   round(move_hi * (delta_hi + 0.05) * 100, 0),
            },
        ]

        result["options"]  = options
        result["available"] = True
        result["summary"]  = (
            "Based on predicted move of $%.2f-$%.2f and %s volatility profile — "
            "look for a strike with these delta/DTE characteristics at your broker." % (
                move_lo, move_hi, tier.lower()
            )
        )
        return result
    except Exception as _e:
        print("[strike_guidance] error: %s" % str(_e)[:100])
        return result


def render_volatility_block_html(vol, pred, strike, fib, scan_time, style='swing'):
    """
    Full render block combining:
    - Volatility classification chip
    - Predicted move (daily + candle)
    - Strike guidance (3 options)
    - Fibonacci multi-timeframe read

    Plain English throughout. No jargon.
    """
    if not vol.get("available") and not pred.get("available") and not fib.get("available"):
        return ""

    parts = []

    # === SECTION 1: VOLATILITY + PREDICTED MOVE ===
    if vol.get("available") or pred.get("available"):
        tier_color = {"LOW": "#F6E27A", "MODERATE": "#D4AF37", "HIGH": "#22C55E"}.get(
            vol.get("tier", "MODERATE"), "#D4AF37"
        )
        vol_chip = ""
        if vol.get("available"):
            if style == "quick":
                _dte_guidance = (
                    "Quick trade \u2014 use 0-5 DTE. Stock moves $%.2f/day (%.1f%% of price). "
                    "For a same-day move, target delta %.2f-%.2f \u2014 enough sensitivity "
                    "to profit without overpaying on premium." % (
                        vol.get("atr_dollar", 0), vol.get("atr_pct", 0),
                        vol.get("delta_lo", 0.35), vol.get("delta_hi", 0.55)
                    )
                )
            else:
                _dte_guidance = vol.get("guidance", "")
            vol_chip = (
                "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                "<span style='background:" + tier_color + "22;color:" + tier_color + ";"
                "border:1px solid " + tier_color + ";border-radius:4px;padding:2px 8px;"
                "font-size:0.65rem;font-weight:700;letter-spacing:1px'>"
                + vol.get("label", "") + "</span>"
                "<span style='color:#A1A1A6;font-size:0.72rem'>"
                "$%.2f avg daily range (%.1f%% of price)</span>" % (
                    vol.get("atr_dollar", 0), vol.get("atr_pct", 0)
                ) +
                "</div>"
                "<div style='font-size:0.73rem;color:#F5F5F5;line-height:1.5;margin-bottom:8px'>"
                + _dte_guidance + "</div>"
            )

        move_html = ""
        if pred.get("available"):
            conf = pred.get("confidence", "MODERATE")
            conf_color = "#22C55E" if "HIGH" in conf else "#D4AF37"

            daily_html = ""
            if pred.get("daily_lo") and pred.get("daily_hi"):
                daily_html = (
                    "<div style='background:#111115;border-radius:6px;padding:8px 10px;margin-bottom:6px'>"
                    "<div style='font-size:0.6rem;color:#A1A1A6;letter-spacing:1px'>TODAY'S EXPECTED MOVE</div>"
                    "<div style='font-size:1.0rem;font-weight:700;color:#F5F5F5'>"
                    "$%.2f - $%.2f" % (pred["daily_lo"], pred["daily_hi"]) +
                    " <span style='font-size:0.65rem;color:" + conf_color + ";margin-left:6px'>"
                    + conf + " CONFIDENCE</span></div>"
                )
                if pred.get("daily_lo_price") and pred.get("daily_hi_price"):
                    daily_html += (
                        "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:2px'>"
                        "Price range: $%.2f - $%.2f</div>" % (
                            pred["daily_lo_price"], pred["daily_hi_price"]
                        )
                    )
                daily_html += "</div>"

            candle_html = ""
            if pred.get("candle_time"):
                state = pred.get("candle_state", "WAITING")
                state_color = {
                    "CURRENT_IS_ENTRY": "#22C55E",
                    "WATCH_CLOSE":      "#D4AF37",
                    "NEW_CANDLE":       "#F6E27A",
                    "WAITING":          "#A1A1A6",
                }.get(state, "#A1A1A6")

                candle_html = (
                    "<div style='background:#111115;border-radius:6px;padding:8px 10px;margin-bottom:6px'>"
                    "<div style='font-size:0.6rem;color:#A1A1A6;letter-spacing:1px'>15-MIN CANDLE TIMING</div>"
                    "<div style='font-size:0.7rem;color:" + state_color + ";font-weight:700;margin:3px 0'>"
                    + {"CURRENT_IS_ENTRY":"* ENTRY CANDLE PRINTING NOW","WATCH_CLOSE":"* WATCH THIS CLOSE",
                       "NEW_CANDLE":"NEW CANDLE OPENED","WAITING":"MONITORING"}.get(state, "") +
                    "</div>"
                    "<div style='font-size:0.73rem;color:#F5F5F5;line-height:1.5'>" + pred["candle_time"] + "</div>"
                )
                if pred.get("candle_lo") and pred.get("candle_hi"):
                    candle_html += (
                        "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:4px'>"
                        "Projected next candle range: $%.2f - $%.2f</div>" % (
                            pred["candle_lo"], pred["candle_hi"]
                        )
                    )
                candle_html += (
                    "<div style='font-size:0.62rem;color:#4a5568;margin-top:4px'>"
                    "Data as of %s — rescan for latest 15-min</div>" % (scan_time or "") +
                    "</div>"
                )

            move_html = daily_html + candle_html

        parts.append(
            "<div style='background:#0B0B0C;border:1px solid #2A2A2D;border-radius:10px;"
            "padding:12px 14px;margin-top:8px'>"
            "<div style='color:#D4AF37;font-family:monospace;font-size:0.68rem;"
            "letter-spacing:2px;font-weight:700;margin-bottom:8px'>📊 MOVE ANALYSIS</div>"
            + vol_chip + move_html +
            "</div>"
        )

    # === SECTION 2: STRIKE GUIDANCE ===
    if strike and strike.get("available"):
        opts_html = ""
        for i, opt in enumerate(strike.get("options", [])):
            badge_colors = ["#22C55E", "#D4AF37", "#FF6B35"]
            bc = badge_colors[i]
            opts_html += (
                "<div style='background:#111115;border-radius:6px;padding:8px 10px;margin-bottom:6px'>"
                "<div style='display:flex;justify-content:space-between;align-items:center'>"
                "<span style='color:" + bc + ";font-size:0.72rem;font-weight:700'>" + opt["label"] + "</span>"
                "<span style='color:#A1A1A6;font-size:0.68rem'>"
                "Delta %.2f-%.2f | %d-%d DTE</span>" % (
                    opt["delta_lo"], opt["delta_hi"], opt["dte_lo"], opt["dte_hi"]
                ) +
                "</div>"
                "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:3px'>" + opt["why"] + "</div>"
                "<div style='font-size:0.7rem;color:#F5F5F5;margin-top:2px'>"
                "Estimated option gain on predicted move: "
                "$%.0f - $%.0f per contract</div>" % (opt["gain_lo"], opt["gain_hi"]) +
                "</div>"
            )

        parts.append(
            "<div style='background:#0B0B0C;border:1px solid #2A2A2D;border-radius:10px;"
            "padding:12px 14px;margin-top:8px'>"
            "<div style='color:#D4AF37;font-family:monospace;font-size:0.68rem;"
            "letter-spacing:2px;font-weight:700;margin-bottom:6px'>🎯 STRIKE GUIDANCE</div>"
            "<div style='font-size:0.72rem;color:#A1A1A6;margin-bottom:8px'>"
            + strike.get("summary", "") + "</div>"
            + opts_html +
            "<div style='font-size:0.65rem;color:#4a5568;margin-top:6px'>"
            "Find these strikes at your broker. Delta is shown in the options chain. "
            "Estimated gains are approximate — actual results depend on IV at entry.</div>"
            "</div>"
        )

    # === SECTION 3: FIBONACCI MULTI-TIMEFRAME ===
    if fib and fib.get("available"):
        fib_rows = ""
        tf_map = [
            ("intraday", "Intraday (1H)", "Quick trade timeframe"),
            ("daily",    "Daily",         "Swing trade timeframe"),
            ("weekly",   "Weekly",        "Leap timeframe"),
        ]
        for key, label, sublabel in tf_map:
            read = fib.get(key)
            if not read or not read.get("valid"):
                continue
            level   = read.get("level", "")
            verdict = read.get("verdict", "")
            color   = read.get("color", "#A1A1A6")
            valid   = read.get("trade_valid", True)
            icon    = "OK" if valid else "!!"
            move    = read.get("move", {})
            move_str = ""
            if move.get("move_start") and move.get("move_end"):
                move_str = " (Move: $%.2f to $%.2f)" % (move["move_start"], move["move_end"])

            fib_rows += (
                "<div style='display:flex;justify-content:space-between;align-items:center;"
                "padding:6px 0;border-bottom:1px solid #1A1A1D'>"
                "<div>"
                "<div style='font-size:0.75rem;color:#F5F5F5;font-weight:700'>" + label + "</div>"
                "<div style='font-size:0.62rem;color:#4a5568'>" + sublabel + move_str + "</div>"
                "</div>"
                "<div style='text-align:right'>"
                "<span style='color:" + color + ";font-size:0.72rem;font-weight:700'>"
                + icon + " " + level + "</span>"
                "<div style='font-size:0.65rem;color:" + color + "'>" + verdict + "</div>"
                "</div>"
                "</div>"
            )

        # Verdict + meaning
        overall = fib.get("verdict", "")
        is_conflict = fib.get("conflict", False)
        trade_valid = fib.get("trade_valid", True)

        verdict_color = "#22C55E" if trade_valid and not is_conflict else "#FF6B35" if is_conflict else "#C1121F"
        verdict_icon  = "OK" if trade_valid and not is_conflict else "!!" if is_conflict else "X"

        conflict_html = ""
        if is_conflict and fib.get("conflict_note"):
            conflict_html = (
                "<div style='background:#1a0f00;border-left:3px solid #FF6B35;"
                "border-radius:4px;padding:8px 12px;margin-top:8px'>"
                "<div style='font-size:0.6rem;color:#FF6B35;letter-spacing:1px;"
                "font-weight:700;margin-bottom:3px'>!! TIMEFRAME CONFLICT — READ CAREFULLY</div>"
                "<div style='font-size:0.73rem;color:#F5F5F5;line-height:1.6'>"
                + fib["conflict_note"] + "</div>"
                "</div>"
            )

        action_html = ""
        highest_read = fib.get("daily") or fib.get("intraday")
        if highest_read and highest_read.get("action"):
            action_html = (
                "<div style='font-size:0.73rem;color:#F5F5F5;margin-top:6px;line-height:1.5'>"
                + highest_read["action"] + "</div>"
            )

        parts.append(
            "<div style='background:#0B0B0C;border:1px solid #2A2A2D;border-radius:10px;"
            "padding:12px 14px;margin-top:8px'>"
            "<div style='color:#D4AF37;font-family:monospace;font-size:0.68rem;"
            "letter-spacing:2px;font-weight:700;margin-bottom:8px'>📐 FIBONACCI READ</div>"
            + fib_rows +
            "<div style='margin-top:8px;display:flex;align-items:center;gap:8px'>"
            "<span style='color:" + verdict_color + ";font-weight:700;font-size:0.78rem'>"
            + verdict_icon + " " + overall + "</span>"
            "</div>"
            + action_html + conflict_html +
            "</div>"
        )

    if not parts:
        return ""

    return "\n".join(parts)


# ORB ENGINE — Opening Range Breakout scanner core
# Timeframes: 15m range (9:30-9:45 ET) + 5m break/retest tracking. No 1m.

ORB_OPEN_H,  ORB_OPEN_M  = 9, 30
ORB_CLOSE_H, ORB_CLOSE_M = 9, 45
PM_START_H,  PM_START_M  = 4, 0
RTH_END_H,   RTH_END_M   = 16, 0

# Retest wick may penetrate at most this fraction of range depth back inside
ORB_WICK_DEPTH_MAX = 0.33
# Range width must sit inside this band as a fraction of daily ATR
ORB_ATR_MIN_FRAC = 0.15
ORB_ATR_MAX_FRAC = 0.50
# Minimum reward:risk to the next structural level
ORB_MIN_RR = 2.0
# A retest counts as confluence if VWAP/9EMA sits within this % of the level
ORB_CONFLUENCE_PCT = 0.0025
# TRIGGER_LIVE window: retest within this many 5m bars of the latest bar
ORB_LIVE_BARS = 2


def _orb_minutes(ts):
    """Minutes since midnight for a pandas Timestamp."""
    try:
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:
        return -1


def orb_calc_vwap(df):
    """Session VWAP over the supplied bars. Returns a list aligned to df rows."""
    try:
        out, cum_pv, cum_v = [], 0.0, 0.0
        for _, row in df.iterrows():
            tp = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3.0
            v  = float(row["volume"]) or 0.0
            cum_pv += tp * v
            cum_v  += v
            out.append(cum_pv / cum_v if cum_v > 0 else tp)
        return out
    except Exception:
        return []


def orb_calc_ema(values, period=9):
    """Simple EMA over a list of floats. Returns list aligned to input."""
    try:
        if not values:
            return []
        k = 2.0 / (period + 1.0)
        out = [float(values[0])]
        for v in values[1:]:
            out.append(float(v) * k + out[-1] * (1 - k))
        return out
    except Exception:
        return []


def orb_session_frames(df_5m, session_date=None):
    """
    Split a 5m frame into premarket and RTH bars for the most recent session.
    Returns (df_pm, df_rth, session_date) — any may be None/empty.
    """
    try:
        if df_5m is None or len(df_5m) == 0:
            return None, None, None
        d = df_5m.copy()
        d["_date"] = d["datetime"].dt.date
        if session_date is None:
            session_date = d["_date"].max()
        d = d[d["_date"] == session_date]
        if len(d) == 0:
            return None, None, session_date
        d["_min"] = d["datetime"].apply(_orb_minutes)
        pm_lo  = PM_START_H * 60 + PM_START_M
        rth_lo = ORB_OPEN_H * 60 + ORB_OPEN_M
        rth_hi = RTH_END_H * 60 + RTH_END_M
        df_pm  = d[(d["_min"] >= pm_lo)  & (d["_min"] < rth_lo)].reset_index(drop=True)
        df_rth = d[(d["_min"] >= rth_lo) & (d["_min"] < rth_hi)].reset_index(drop=True)
        return df_pm, df_rth, session_date
    except Exception:
        return None, None, None


def orb_calc_levels(ticker, df_5m=None, df_daily=None):
    """
    Build the day's structural levels.
    Returns dict — always includes 'available' and 'state'.
    state: NO_DATA | RANGE_BUILDING | RANGE_SET
    """
    res = {
        "available": False, "state": "NO_DATA", "ticker": ticker,
        "orb_high": 0.0, "orb_low": 0.0, "range_width": 0.0, "range_pct": 0.0,
        "orb_volume": 0.0, "pm_high": None, "pm_low": None,
        "pdh": None, "pdl": None, "atr": 0.0, "range_vs_atr": 0.0,
        "ma50": None, "ma200": None, "ma400": None,
        "last_price": 0.0, "last_bar_time": "", "session_date": None,
        "pm_available": False, "note": "",
    }
    try:
        if df_5m is None:
            df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is None or len(df_5m) == 0:
            res["note"] = "no intraday data"
            return res

        df_pm, df_rth, sess = orb_session_frames(df_5m)
        res["session_date"] = str(sess) if sess else None
        if df_rth is None or len(df_rth) == 0:
            res["note"] = "no RTH bars yet"
            return res

        res["last_price"]    = float(df_rth["close"].iloc[-1])
        res["last_bar_time"] = df_rth["datetime"].iloc[-1].strftime("%-I:%M %p")

        # Premarket — only report if bars actually exist (FMP coverage varies)
        if df_pm is not None and len(df_pm) > 0:
            res["pm_high"] = float(df_pm["high"].max())
            res["pm_low"]  = float(df_pm["low"].min())
            res["pm_available"] = True

        # Opening range = 9:30 through 9:45 (three 5m bars), wick to wick
        orb_end = ORB_CLOSE_H * 60 + ORB_CLOSE_M
        df_orb  = df_rth[df_rth["_min"] < orb_end]
        if len(df_orb) == 0:
            res["note"] = "range not started"
            return res

        res["orb_high"]   = float(df_orb["high"].max())
        res["orb_low"]    = float(df_orb["low"].min())
        res["orb_volume"] = float(df_orb["volume"].sum())
        res["range_width"] = res["orb_high"] - res["orb_low"]
        if res["orb_high"] > 0:
            res["range_pct"] = (res["range_width"] / res["orb_high"]) * 100.0

        # Range is only final once 9:45 has printed
        last_min = int(df_rth["_min"].iloc[-1])
        if last_min < orb_end:
            res["state"] = "RANGE_BUILDING"
            res["available"] = True
            res["note"] = "range completes 9:45 ET"
            return res

        # Daily context — prev day levels, ATR, moving averages
        if df_daily is None:
            df_daily = _fmp_download(ticker, "2y", "1d")
        if df_daily is not None and len(df_daily) >= 15:
            dd = df_daily.copy()
            dd["_date"] = dd["datetime"].dt.date
            prior = dd[dd["_date"] < sess] if sess else dd.iloc[:-1]
            if len(prior) > 0:
                res["pdh"] = float(prior["high"].iloc[-1])
                res["pdl"] = float(prior["low"].iloc[-1])
            h, l, c = prior["high"], prior["low"], prior["close"]
            tr  = (h - l).copy()
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr  = tr.combine(tr2, max).combine(tr3, max)
            if len(tr) >= 14:
                res["atr"] = float(tr.rolling(14).mean().iloc[-1])
            for per, key in ((50, "ma50"), (200, "ma200"), (400, "ma400")):
                if len(prior) >= per:
                    res[key] = float(prior["close"].rolling(per).mean().iloc[-1])

        if res["atr"] > 0:
            res["range_vs_atr"] = res["range_width"] / res["atr"]

        res["state"] = "RANGE_SET"
        res["available"] = True
        return res
    except Exception as e:
        res["note"] = "levels error: " + str(e)[:60]
        return res


def orb_detect_events(levels, df_5m=None):
    """
    Walk 5m bars after 9:45 and build the day's event timeline for BOTH boundaries.
    Break  = 5m CLOSE beyond the level.
    Retest = later bar touches the level (wick or body) but does NOT close back through it,
             and the wick stays within ORB_WICK_DEPTH_MAX of range depth.
    Returns dict with a 'high' and a 'low' side, each carrying its own timeline.
    """
    out = {
        "available": False, "timeline": [],
        "high": {"broken": False, "break_time": "", "break_price": 0.0, "break_vol_ratio": 0.0,
                 "break_close_strength": 0.0, "retested": False, "retest_time": "",
                 "retest_grade": "", "retest_level": "", "failed": False, "fail_time": "",
                 "bars_since_retest": 999, "proximity": False, "retest_price": 0.0},
        "low":  {"broken": False, "break_time": "", "break_price": 0.0, "break_vol_ratio": 0.0,
                 "break_close_strength": 0.0, "retested": False, "retest_time": "",
                 "retest_grade": "", "retest_level": "", "failed": False, "fail_time": "",
                 "bars_since_retest": 999, "proximity": False, "retest_price": 0.0},
    }
    try:
        if not levels.get("available") or levels.get("state") != "RANGE_SET":
            return out
        if df_5m is None:
            df_5m = _fmp_download(levels["ticker"], "5d", "5m")
        _, df_rth, _ = orb_session_frames(df_5m)
        if df_rth is None or len(df_rth) == 0:
            return out

        vwaps = orb_calc_vwap(df_rth)
        ema9  = orb_calc_ema([float(x) for x in df_rth["close"].tolist()], 9)

        orb_end = ORB_CLOSE_H * 60 + ORB_CLOSE_M
        post    = df_rth[df_rth["_min"] >= orb_end].reset_index(drop=True)
        if len(post) == 0:
            return out
        offset = len(df_rth) - len(post)

        oh, ol = levels["orb_high"], levels["orb_low"]
        depth  = levels["range_width"] * ORB_WICK_DEPTH_MAX
        # Proximity zone: a rejection this far short of the level still
        # counts as a retest (price often rejects just before tagging).
        rz_h = max(oh * 0.006, levels["range_width"] * 0.22)
        rz_l = max(ol * 0.006, levels["range_width"] * 0.22)
        # Baseline volume = average 5m bar volume inside the opening range
        base_vol = levels.get("orb_volume", 0) / 3.0 if levels.get("orb_volume") else 0.0

        tl = []
        for i, row in post.iterrows():
            gi   = offset + i
            t    = row["datetime"].strftime("%-I:%M")
            hi   = float(row["high"]);  lo = float(row["low"])
            cl   = float(row["close"]); op = float(row["open"])
            vol  = float(row["volume"]) or 0.0
            vwap = vwaps[gi] if gi < len(vwaps) else cl
            ema  = ema9[gi]  if gi < len(ema9)  else cl
            vr   = (vol / base_vol) if base_vol > 0 else 0.0
            bar_rng = max(hi - lo, 1e-9)

            # ---- HIGH boundary (bullish) ----
            H = out["high"]
            if not H["broken"]:
                if cl > oh:
                    H["broken"] = True
                    H["break_time"] = t
                    H["break_price"] = cl
                    H["break_vol_ratio"] = round(vr, 2)
                    H["break_close_strength"] = round((cl - lo) / bar_rng, 2)
                    tl.append({"t": t, "side": "high", "kind": "BREAK",
                               "txt": "Broke ORB high $%.2f on %.1fx vol" % (oh, vr)})
            elif not H["failed"] and not H["retested"]:
                if cl < oh:
                    H["failed"] = True
                    H["fail_time"] = t
                    tl.append({"t": t, "side": "high", "kind": "FAIL",
                               "txt": "Closed back inside range — high break failed"})
                elif lo <= oh:
                    if lo >= oh - depth:
                        near_vwap = abs(vwap - oh) / oh <= ORB_CONFLUENCE_PCT
                        near_ema  = abs(ema  - oh) / oh <= ORB_CONFLUENCE_PCT
                        if near_vwap or near_ema:
                            grade, lvl = "A+", ("ORB high + VWAP" if near_vwap else "ORB high + 9EMA")
                        elif lo >= oh - (depth * 0.35):
                            grade, lvl = "A", "ORB high"
                        else:
                            grade, lvl = "B", "ORB high (deep wick)"
                        H["retested"] = True
                        H["retest_time"] = t
                        H["retest_grade"] = grade
                        H["retest_level"] = lvl
                        H["bars_since_retest"] = len(post) - 1 - i
                        tl.append({"t": t, "side": "high", "kind": "RETEST",
                                   "txt": "Retest held at %s — grade %s" % (lvl, grade)})
                    else:
                        H["failed"] = True
                        H["fail_time"] = t
                        tl.append({"t": t, "side": "high", "kind": "FAIL",
                                   "txt": "Wick cut too deep into range — break failed"})
                elif lo <= oh + rz_h and cl > op:
                    # Proximity retest: price pulled back to within the zone above the
                    # ORB high and bounced (bullish close) without tagging the level.
                    H["retested"] = True
                    H["proximity"] = True
                    H["retest_time"] = t
                    H["retest_grade"] = "B"
                    H["retest_level"] = "near ORB high (bounced $%.2f above)" % (lo - oh)
                    H["retest_price"] = lo
                    H["bars_since_retest"] = len(post) - 1 - i
                    tl.append({"t": t, "side": "high", "kind": "RETEST",
                               "txt": "Near retest — bounced at $%.2f, just above ORB high" % lo})
            elif H["retested"]:
                H["bars_since_retest"] = len(post) - 1 - i if i >= i else H["bars_since_retest"]

            # ---- LOW boundary (bearish) ----
            L = out["low"]
            if not L["broken"]:
                if cl < ol:
                    L["broken"] = True
                    L["break_time"] = t
                    L["break_price"] = cl
                    L["break_vol_ratio"] = round(vr, 2)
                    L["break_close_strength"] = round((hi - cl) / bar_rng, 2)
                    tl.append({"t": t, "side": "low", "kind": "BREAK",
                               "txt": "Broke ORB low $%.2f on %.1fx vol" % (ol, vr)})
            elif not L["failed"] and not L["retested"]:
                if cl > ol:
                    L["failed"] = True
                    L["fail_time"] = t
                    tl.append({"t": t, "side": "low", "kind": "FAIL",
                               "txt": "Closed back inside range — low break failed"})
                elif hi >= ol:
                    if hi <= ol + depth:
                        near_vwap = abs(vwap - ol) / ol <= ORB_CONFLUENCE_PCT
                        near_ema  = abs(ema  - ol) / ol <= ORB_CONFLUENCE_PCT
                        if near_vwap or near_ema:
                            grade, lvl = "A+", ("ORB low + VWAP" if near_vwap else "ORB low + 9EMA")
                        elif hi <= ol + (depth * 0.35):
                            grade, lvl = "A", "ORB low"
                        else:
                            grade, lvl = "B", "ORB low (deep wick)"
                        L["retested"] = True
                        L["retest_time"] = t
                        L["retest_grade"] = grade
                        L["retest_level"] = lvl
                        L["bars_since_retest"] = len(post) - 1 - i
                        tl.append({"t": t, "side": "low", "kind": "RETEST",
                                   "txt": "Retest held at %s — grade %s" % (lvl, grade)})
                    else:
                        L["failed"] = True
                        L["fail_time"] = t
                        tl.append({"t": t, "side": "low", "kind": "FAIL",
                                   "txt": "Wick cut too deep into range — break failed"})
                elif hi >= ol - rz_l and cl < op:
                    # Proximity retest: price rallied to within the zone below the
                    # ORB low and rejected (bearish close) without tagging the level.
                    L["retested"] = True
                    L["proximity"] = True
                    L["retest_time"] = t
                    L["retest_grade"] = "B"
                    L["retest_level"] = "near ORB low (rejected $%.2f below)" % (ol - hi)
                    L["retest_price"] = hi
                    L["bars_since_retest"] = len(post) - 1 - i
                    tl.append({"t": t, "side": "low", "kind": "RETEST",
                               "txt": "Near retest — rejected at $%.2f, just below ORB low" % hi})

        # Recompute bars_since_retest cleanly from stored times
        for side in ("high", "low"):
            S = out[side]
            if S["retested"]:
                for j, row in post.iterrows():
                    if row["datetime"].strftime("%-I:%M") == S["retest_time"]:
                        S["bars_since_retest"] = len(post) - 1 - j
                        break

        out["timeline"] = tl
        out["last_vwap"] = vwaps[-1] if vwaps else 0.0
        out["last_ema9"] = ema9[-1] if ema9 else 0.0
        out["bars_post"] = len(post)
        out["available"] = True
        return out
    except Exception as e:
        out["note"] = "events error: " + str(e)[:60]
        return out


def orb_evaluate_structure(levels, events, side, df_5m=None):
    """
    Is the trade still alive? Structure decides, not the clock.
    TRIGGER_LIVE     — retest held within the last ORB_LIVE_BARS bars
    TREND_INTACT     — retest held earlier, price still respecting 9EMA and correct side of VWAP
    STRUCTURE_BROKEN — closed through VWAP, or lost the 9EMA on 2+ consecutive closes
    NO_TRIGGER       — no break, or broke but has not retested yet
    """
    res = {"state": "NO_TRIGGER", "reason": "", "vwap_ok": False, "ema_ok": False,
           "last_vwap": 0.0, "last_ema9": 0.0, "ema_breach_count": 0}
    try:
        S = events.get(side, {})
        if not S.get("broken"):
            res["reason"] = "no break yet"
            return res
        if S.get("failed"):
            res["state"] = "STRUCTURE_BROKEN"
            res["reason"] = "break failed at " + (S.get("fail_time") or "")
            return res
        if not S.get("retested"):
            res["state"] = "NO_TRIGGER"
            res["reason"] = "broke, awaiting retest"
            return res

        if df_5m is None:
            df_5m = _fmp_download(levels["ticker"], "5d", "5m")
        _, df_rth, _ = orb_session_frames(df_5m)
        if df_rth is None or len(df_rth) == 0:
            return res

        vwaps = orb_calc_vwap(df_rth)
        ema9  = orb_calc_ema([float(x) for x in df_rth["close"].tolist()], 9)
        res["last_vwap"] = vwaps[-1] if vwaps else 0.0
        res["last_ema9"] = ema9[-1]  if ema9  else 0.0

        # Bars from the retest forward
        start = 0
        for j, row in df_rth.iterrows():
            if row["datetime"].strftime("%-I:%M") == S.get("retest_time"):
                start = j
                break
        bullish = (side == "high")

        vwap_ok, ema_streak, worst_streak = True, 0, 0
        for j in range(start, len(df_rth)):
            cl = float(df_rth["close"].iloc[j])
            vw = vwaps[j] if j < len(vwaps) else cl
            em = ema9[j]  if j < len(ema9)  else cl
            if (bullish and cl < vw) or ((not bullish) and cl > vw):
                vwap_ok = False
            if (bullish and cl < em) or ((not bullish) and cl > em):
                ema_streak += 1
                worst_streak = max(worst_streak, ema_streak)
            else:
                ema_streak = 0

        res["vwap_ok"] = vwap_ok
        res["ema_ok"]  = worst_streak < 2
        res["ema_breach_count"] = worst_streak

        if not vwap_ok:
            res["state"] = "STRUCTURE_BROKEN"
            res["reason"] = "closed through VWAP"
        elif worst_streak >= 2:
            res["state"] = "STRUCTURE_BROKEN"
            res["reason"] = "lost the 9EMA (%d consecutive closes)" % worst_streak
        elif S.get("bars_since_retest", 999) <= ORB_LIVE_BARS:
            res["state"] = "TRIGGER_LIVE"
            res["reason"] = "retest just held — entry at the level"
        else:
            res["state"] = "TREND_INTACT"
            res["reason"] = "riding 9EMA on the right side of VWAP — entry on pullback to 9EMA"
        return res
    except Exception as e:
        res["reason"] = "structure error: " + str(e)[:50]
        return res


def orb_next_level_up(levels, price):
    """Nearest structural level above price. Returns (value, label) or (None, '')."""
    cands = []
    for key, lbl in (("pdh", "PDH"), ("pm_high", "PM High"),
                     ("ma50", "50MA"), ("ma200", "200MA"), ("ma400", "400MA")):
        v = levels.get(key)
        if v and v > price:
            cands.append((float(v), lbl))
    if not cands:
        return None, ""
    cands.sort(key=lambda x: x[0])
    return cands[0]


def orb_next_level_down(levels, price):
    """Nearest structural level below price. Returns (value, label) or (None, '')."""
    cands = []
    for key, lbl in (("pdl", "PDL"), ("pm_low", "PM Low"),
                     ("ma50", "50MA"), ("ma200", "200MA"), ("ma400", "400MA")):
        v = levels.get(key)
        if v and v < price:
            cands.append((float(v), lbl))
    if not cands:
        return None, ""
    cands.sort(key=lambda x: -x[0])
    return cands[0]


def orb_calc_trade_geometry(levels, side, events=None):
    """
    Entry / stop / target for a retest entry on the given boundary.
    Stop sits at the same depth that invalidates the retest, so the
    invalidation level and the stop are the same price.
    """
    g = {"entry": 0.0, "stop": 0.0, "target": 0.0, "risk": 0.0, "reward": 0.0,
         "rr": 0.0, "target_label": "", "rr_ok": False, "atr_ok": False,
         "atr_note": "", "ma_in_path": "", "ma_in_path_pct": 0.0,
         "target2": 0.0, "target2_label": "", "rr2": 0.0}
    try:
        rw = levels.get("range_width", 0)
        if rw <= 0:
            return g
        risk = rw * ORB_WICK_DEPTH_MAX
        atr  = levels.get("atr", 0) or 0
        # Ceilings: an intraday trade can't realistically travel more than ~1x
        # the daily ATR from the break. The runner target may reach ~1.5x ATR.
        # Without ATR, fall back to range multiples.
        cap_primary = atr * 1.0 if atr > 0 else rw * 3.0
        cap_ext     = atr * 1.5 if atr > 0 else rw * 4.0

        bull  = (side == "high")
        orb_level = levels["orb_high"] if bull else levels["orb_low"]
        # Invalidation stop is anchored to the ORB level regardless of entry.
        stop  = orb_level - risk if bull else orb_level + risk
        # Default entry is the ORB level. For a proximity retest, price rejected
        # short of the level — enter at the rejection, not the untagged level.
        entry = orb_level
        _prox = False
        if events:
            _sd = events.get(side, {})
            if _sd.get("proximity") and _sd.get("retest_price"):
                entry = float(_sd["retest_price"])
                _prox = True

        # 1x measured move — the opening range projected from the break point.
        # This scales with the stock's own volatility that day and is the
        # natural, conservative first target for an ORB break.
        mm = orb_level + rw if bull else orb_level - rw

        # Structural levels in the trade direction (from daily/prev-day data).
        struct = []
        for key, slbl in (("pdh", "PDH"), ("pdl", "PDL"), ("pm_high", "PM High"),
                          ("pm_low", "PM Low"), ("ma50", "50MA"),
                          ("ma200", "200MA"), ("ma400", "400MA")):
            v = levels.get(key)
            if not v:
                continue
            if (bull and v > orb_level) or ((not bull) and v < orb_level):
                struct.append((float(v), slbl))

        def _dist(v):
            return (v - entry) if bull else (entry - v)

        # Candidate targets: measured move + in-direction structural levels,
        # each with its positive in-direction distance.
        cands = [(mm, "measured move")] + struct
        cands = [(v, l, _dist(v)) for (v, l) in cands if _dist(v) > 0]

        # Primary target = nearest candidate within the 1x-ATR ceiling.
        # This is what the R:R is judged on — a level too far to reach
        # intraday (e.g. the 200MA $17 away) is filtered out here.
        prim_pool = [c for c in cands if c[2] <= cap_primary] or cands
        prim_pool.sort(key=lambda c: c[2])
        target, lbl, reward = prim_pool[0][0], prim_pool[0][1], prim_pool[0][2]

        # Runner target = nearest candidate beyond the primary, within 1.5x ATR;
        # if none, default to 2x the measured move.
        ext_pool = [c for c in cands if c[2] > reward and c[2] <= cap_ext]
        ext_pool.sort(key=lambda c: c[2])
        if ext_pool:
            t2, l2, r2 = ext_pool[0]
        else:
            t2 = entry + rw * 2 if bull else entry - rw * 2
            l2, r2 = "2x measured move", (rw * 2)

        g.update({"entry": entry, "stop": stop, "target": target,
                  "risk": risk, "reward": reward, "target_label": lbl,
                  "rr": (reward / risk) if risk > 0 else 0.0,
                  "target2": t2, "target2_label": l2,
                  "rr2": (r2 / risk) if risk > 0 else 0.0})
        g["rr_ok"] = g["rr"] >= ORB_MIN_RR
        g["proximity"] = _prox
        g["orb_level"] = orb_level

        # ATR band — range must be meaningful but not already exhausted
        rva = levels.get("range_vs_atr", 0)
        if rva <= 0:
            g["atr_note"] = "ATR unavailable"
        elif rva < ORB_ATR_MIN_FRAC:
            g["atr_note"] = "Range only %.0f%% of ATR — chop risk, no room" % (rva * 100)
        elif rva > ORB_ATR_MAX_FRAC:
            g["atr_note"] = "Range already %.0f%% of ATR — day's move may be spent" % (rva * 100)
        else:
            g["atr_ok"] = True
            g["atr_note"] = "Range %.0f%% of ATR — healthy" % (rva * 100)

        # Is a major MA sitting in the path between entry and target?
        lo_b, hi_b = (min(entry, target), max(entry, target))
        for key, lbl2 in (("ma50", "50MA"), ("ma200", "200MA"), ("ma400", "400MA")):
            v = levels.get(key)
            if v and lo_b < v < hi_b:
                g["ma_in_path"] = lbl2
                g["ma_in_path_pct"] = abs(v - entry) / entry * 100 if entry else 0
                break
        return g
    except Exception:
        return g


def orb_time_modifier(bar_time_str):
    """Prime-window multiplier. Same break at 9:50 and 12:30 are not the same trade."""
    try:
        parts = bar_time_str.replace("AM", "").replace("PM", "").strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if "PM" in bar_time_str and h != 12:
            h += 12
        mins = h * 60 + m
        if mins < 11 * 60:        return 1.00, "prime window"
        if mins < 14 * 60:        return 0.85, "lunch chop — discounted"
        if mins < 15 * 60 + 30:   return 0.92, "afternoon"
        return 0.80, "late day — discounted"
    except Exception:
        return 1.00, ""


def orb_score_setup(levels, events, structure, side, geo, context=None):
    """
    70 structure / 30 context. Context can subtract but never manufacture a signal.
    Returns dict with score, bucket, and a full component breakdown.
    """
    ctx = context or {}
    res = {"score": 0, "bucket": "", "structure_pts": 0, "context_pts": 0,
           "components": [], "penalties": [], "direction": "",
           "time_note": "", "time_mult": 1.0, "gated": False, "gate_reason": ""}
    try:
        S = events.get(side, {})
        res["direction"] = "bullish" if side == "high" else "bearish"
        if not S.get("broken"):
            res["gate_reason"] = "no break"
            return res

        # ---------- STRUCTURE (70) ----------
        pts = 0

        # Break confirmation — 22
        bp = 16
        cs = S.get("break_close_strength", 0)
        bp += min(6, int(round(cs * 6)))
        if S.get("failed"):
            bp = 0
        pts += bp
        res["components"].append(("Break confirmed (5m close)", bp, 22))

        # Volume on break — 22
        vr = S.get("break_vol_ratio", 0)
        if   vr >= 2.0: vp, vtxt = 22, "%.1fx — heavy" % vr
        elif vr >= 1.5: vp, vtxt = 17, "%.1fx — solid" % vr
        elif vr >= 1.2: vp, vtxt = 12, "%.1fx — adequate" % vr
        elif vr >= 1.0: vp, vtxt = 7,  "%.1fx — light" % vr
        else:           vp, vtxt = 0,  "%.1fx — no participation" % vr
        pts += vp
        res["components"].append(("Break volume " + vtxt, vp, 22))

        # Retest quality — 16
        grade = S.get("retest_grade", "")
        rp = {"A+": 16, "A": 13, "B": 9}.get(grade, 0)
        if S.get("failed"):
            rp = 0
        pts += rp
        rlbl = ("Retest %s at %s" % (grade, S.get("retest_level", ""))) if grade else "No retest yet"
        res["components"].append((rlbl, rp, 16))

        # Range geometry — 10
        gp = (6 if geo.get("atr_ok") else (3 if levels.get("range_vs_atr", 0) > 0 else 0))
        gp += 4 if geo.get("rr_ok") else 0
        pts += gp
        res["components"].append(("Range geometry / %.1f:1 to %s" % (geo.get("rr", 0), geo.get("target_label", "")), gp, 10))

        res["structure_pts"] = pts

        # ---------- CONTEXT (30) — modifies only ----------
        cpts = 0
        want_bull = (side == "high")

        news_dir = ctx.get("news_direction", "")
        news_str = ctx.get("news_strength", 0)
        if news_dir and news_dir != "neutral":
            aligned = (news_dir == "bullish") == want_bull
            if aligned:
                np_ = min(10, int(round(news_str * 10)))
                cpts += np_
                res["components"].append(("News catalyst aligned", np_, 10))
            else:
                res["penalties"].append(("News catalyst points against the trade", -6))
                cpts -= 6
        else:
            res["components"].append(("No news catalyst", 0, 10))

        if ctx.get("squeeze_active") and ctx.get("squeeze_direction", "") in ("", "both",
                "bullish" if want_bull else "bearish"):
            cpts += 6
            res["components"].append(("Squeeze firing in break direction", 6, 6))
        else:
            res["components"].append(("No squeeze", 0, 6))

        lc = 0
        if geo.get("rr_ok"):        lc += 3
        if not geo.get("ma_in_path"): lc += 3
        cpts += lc
        lctxt = "Clear path to target" if lc == 6 else (
                "%s sits in the path" % geo["ma_in_path"] if geo.get("ma_in_path") else "Limited running room")
        res["components"].append((lctxt, lc, 6))

        mb = ctx.get("market_bias", "")
        if mb in ("bullish", "bearish"):
            if (mb == "bullish") == want_bull:
                cpts += 4
                res["components"].append(("Market regime backs the trade", 4, 4))
            else:
                res["components"].append(("Trading against market regime", 0, 4))
        else:
            cpts += 2
            res["components"].append(("Market regime neutral", 2, 4))

        vc = ctx.get("vol_tier", "")
        if vc in ("MODERATE", "HIGH", "ETF"):
            cpts += 4
            res["components"].append(("Volatility profile fits intraday", 4, 4))
        else:
            res["components"].append(("Low volatility — limited follow-through", 0, 4))

        if ctx.get("block_in_path"):
            cpts -= 10
            res["penalties"].append(("Institutional block sits in the path", -10))

        res["context_pts"] = cpts

        # ---------- TIME MODIFIER ----------
        anchor = S.get("retest_time") or S.get("break_time") or ""
        tm, tnote = orb_time_modifier(anchor)
        res["time_mult"], res["time_note"] = tm, tnote

        raw = max(0, res["structure_pts"] + res["context_pts"])
        res["score"] = int(round(raw * tm))

        # ---------- BUCKET (retest-gated) ----------
        st = structure.get("state", "NO_TRIGGER")
        if st == "STRUCTURE_BROKEN":
            res["bucket"] = "ON DECK"
            res["gated"] = True
            res["gate_reason"] = structure.get("reason", "structure broken")
            res["score"] = min(res["score"], 45)
        elif not S.get("retested"):
            res["bucket"] = "WATCHING" if res["score"] >= 50 else "ON DECK"
            res["gated"] = True
            res["gate_reason"] = "broke but has not retested — retest is the entry"
            res["score"] = min(res["score"], 74)
        elif st in ("TRIGGER_LIVE", "TREND_INTACT"):
            if   res["score"] >= 75: res["bucket"] = "GO NOW"
            elif res["score"] >= 50: res["bucket"] = "WATCHING"
            else:                    res["bucket"] = "ON DECK"
        else:
            res["bucket"] = "ON DECK"
        return res
    except Exception as e:
        res["gate_reason"] = "score error: " + str(e)[:50]
        return res


ORB_STATE_STYLE = {
    "TRIGGER_LIVE":     ("#22C55E", "TRIGGER LIVE",     "Entry at the level right now"),
    "TREND_INTACT":     ("#D4AF37", "TREND INTACT",     "Entry on pullback to the 9EMA"),
    "STRUCTURE_BROKEN": ("#6B6B6B", "STRUCTURE BROKEN", "No entry — timeline only"),
    "NO_TRIGGER":       ("#7AA2F7", "AWAITING RETEST",  "Broke — wait for the pullback"),
}
ORB_BUCKET_COLOR = {"GO NOW": "#22C55E", "WATCHING": "#D4AF37", "ON DECK": "#7AA2F7"}


def render_orb_card_html(levels, events, structure, score, geo, side):
    """Full ORB card — range, state, geometry, timeline, levels, score breakdown."""
    try:
        S     = events.get(side, {})
        bull  = (side == "high")
        arrow = "CALLS" if bull else "PUTS"
        acol  = "#22C55E" if bull else "#C1121F"
        scol, slabel, saction = ORB_STATE_STYLE.get(
            structure.get("state", "NO_TRIGGER"), ORB_STATE_STYLE["NO_TRIGGER"])
        bcol = ORB_BUCKET_COLOR.get(score.get("bucket", ""), "#7AA2F7")

        h = ("<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-left:3px solid "
             + bcol + ";border-radius:6px;padding:14px 16px;margin:10px 0'>")

        # Header
        h += ("<div style='display:flex;align-items:center;justify-content:space-between;"
              "flex-wrap:wrap;gap:8px;margin-bottom:12px'>"
              "<div style='display:flex;align-items:center;gap:10px'>"
              "<span style='color:#F5F5F5;font-size:1.15rem;font-weight:800;letter-spacing:1px'>"
              + str(levels.get("ticker", "")) + "</span>"
              "<span style='background:" + acol + "22;color:" + acol + ";border:1px solid " + acol
              + ";border-radius:3px;padding:2px 7px;font-size:0.62rem;font-weight:700;letter-spacing:1px'>"
              + arrow + "</span></div>"
              "<div style='display:flex;align-items:center;gap:8px'>"
              "<span style='background:" + bcol + "22;color:" + bcol + ";border:1px solid " + bcol
              + ";border-radius:3px;padding:2px 8px;font-size:0.62rem;font-weight:700;letter-spacing:1px'>"
              + str(score.get("bucket", "")) + "</span>"
              "<span style='color:" + bcol + ";font-size:1.15rem;font-weight:800'>"
              + str(score.get("score", 0)) + "</span></div></div>")

        # State banner
        h += ("<div style='background:" + scol + "14;border:1px solid " + scol
              + "55;border-radius:4px;padding:8px 11px;margin-bottom:11px'>"
              "<div style='color:" + scol + ";font-size:0.68rem;font-weight:700;letter-spacing:1.2px;"
              "margin-bottom:3px'>" + slabel + "</div>"
              "<div style='color:#F5F5F5;font-size:0.74rem;line-height:1.45'>" + saction
              + " &middot; " + str(structure.get("reason", "")) + "</div></div>")

        # Opening range
        rng_note = levels.get("range_vs_atr", 0)
        h += ("<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:11px'>"
              "<div style='flex:1;min-width:88px;background:#111;border:1px solid #1F1F1F;"
              "border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>ORB HIGH</div>"
              "<div style='color:#22C55E;font-size:0.92rem;font-weight:700'>$%.2f</div></div>" % levels.get("orb_high", 0)
              + "<div style='flex:1;min-width:88px;background:#111;border:1px solid #1F1F1F;"
              "border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>ORB LOW</div>"
              "<div style='color:#C1121F;font-size:0.92rem;font-weight:700'>$%.2f</div></div>" % levels.get("orb_low", 0)
              + "<div style='flex:1;min-width:88px;background:#111;border:1px solid #1F1F1F;"
              "border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>WIDTH</div>"
              "<div style='color:#F5F5F5;font-size:0.92rem;font-weight:700'>$%.2f</div></div></div>"
              % levels.get("range_width", 0))

        if geo.get("atr_note"):
            acolor = "#22C55E" if geo.get("atr_ok") else "#F6E27A"
            h += ("<div style='color:" + acolor + ";font-size:0.7rem;margin-bottom:11px'>"
                  + geo["atr_note"] + "</div>")

        # Trade geometry
        rr_col = "#22C55E" if geo.get("rr_ok") else "#C1121F"
        h += ("<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:4px;"
              "padding:9px 11px;margin-bottom:11px'>"
              "<div style='color:#D4AF37;font-size:0.62rem;font-weight:700;letter-spacing:1.2px;"
              "margin-bottom:6px'>TRADE GEOMETRY</div>"
              "<div style='display:flex;gap:12px;flex-wrap:wrap;font-size:0.73rem'>"
              "<span style='color:#A1A1A6'>Entry <b style='color:#F5F5F5'>$%.2f</b></span>"
              "<span style='color:#A1A1A6'>Stop <b style='color:#C1121F'>$%.2f</b></span>"
              "<span style='color:#A1A1A6'>Target <b style='color:#22C55E'>$%.2f</b></span>"
              "<span style='color:#A1A1A6'>R:R <b style='color:%s'>%.1f:1</b></span></div>"
              % (geo.get("entry", 0), geo.get("stop", 0), geo.get("target", 0),
                 rr_col, geo.get("rr", 0))
              + "<div style='color:#6B6B6B;font-size:0.66rem;margin-top:5px'>Target is "
              + str(geo.get("target_label", "")) + " (nearest realistic level) &middot; "
              "stop is the retest invalidation level</div>")
        if geo.get("proximity"):
            h += ("<div style='color:#F6E27A;font-size:0.66rem;margin-top:2px'>"
                  "Proximity entry &mdash; price rejected before tagging the level; "
                  "stop still sits at the ORB-level invalidation, so R:R is tighter</div>")
        if geo.get("target2") and geo.get("rr2", 0) > 0:
            h += ("<div style='color:#6B6B6B;font-size:0.66rem;margin-top:2px'>"
                  "Runner: ${t2:.2f} ({l2}) &mdash; {rr2:.1f}:1 if it trends</div>").format(
                      t2=geo.get("target2", 0), l2=geo.get("target2_label", ""),
                      rr2=geo.get("rr2", 0))
        if geo.get("ma_in_path"):
            h += ("<div style='color:#F6E27A;font-size:0.68rem;margin-top:4px'>"
                  + geo["ma_in_path"] + " sits between entry and target — expect resistance there</div>")
        h += "</div>"

        # Strike & expiration — driven by the trade geometry
        try:
            _sd = orb_strike_expiration(levels, geo, side)
            h += render_orb_strike_html(_sd, side)
        except Exception:
            pass

        # Timeline
        tl = [e for e in events.get("timeline", []) if e.get("side") == side]
        if tl:
            h += ("<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:4px;"
                  "padding:9px 11px;margin-bottom:11px'>"
                  "<div style='color:#D4AF37;font-size:0.62rem;font-weight:700;letter-spacing:1.2px;"
                  "margin-bottom:6px'>SESSION TIMELINE</div>")
            for e in tl:
                kcol = {"BREAK": "#7AA2F7", "RETEST": "#22C55E", "FAIL": "#C1121F"}.get(e["kind"], "#A1A1A6")
                h += ("<div style='display:flex;gap:9px;font-size:0.72rem;line-height:1.65'>"
                      "<span style='color:#6B6B6B;min-width:44px'>" + e["t"] + "</span>"
                      "<span style='color:" + kcol + ";min-width:52px;font-weight:700;font-size:0.62rem;"
                      "letter-spacing:0.8px;padding-top:2px'>" + e["kind"] + "</span>"
                      "<span style='color:#F5F5F5'>" + e["txt"] + "</span></div>")
            h += "</div>"

        # Level context
        def _lvl(lbl, val):
            if val is None or not val:
                return ""
            return ("<span style='color:#A1A1A6'>" + lbl
                    + " <b style='color:#F5F5F5'>$%.2f</b></span>" % float(val))
        chips = [_lvl("PDH", levels.get("pdh")), _lvl("PDL", levels.get("pdl"))]
        if levels.get("pm_available"):
            chips += [_lvl("PM Hi", levels.get("pm_high")), _lvl("PM Lo", levels.get("pm_low"))]
        chips += [_lvl("VWAP", structure.get("last_vwap")), _lvl("9EMA", structure.get("last_ema9")),
                  _lvl("50MA", levels.get("ma50")), _lvl("200MA", levels.get("ma200")),
                  _lvl("400MA", levels.get("ma400"))]
        chips = [c for c in chips if c]
        if chips:
            h += ("<div style='display:flex;gap:11px;flex-wrap:wrap;font-size:0.71rem;"
                  "padding:8px 0;border-top:1px solid #1A1A1A;margin-bottom:8px'>"
                  + "".join(chips) + "</div>")
        if not levels.get("pm_available"):
            h += ("<div style='color:#6B6B6B;font-size:0.66rem;margin-bottom:8px'>"
                  "Premarket bars unavailable from the data feed for this ticker</div>")

        # Score breakdown
        h += ("<details style='margin-top:4px'>"
              "<summary style='color:#6B6B6B;font-size:0.68rem;cursor:pointer;letter-spacing:0.6px'>"
              "SCORE BREAKDOWN &middot; structure " + str(score.get("structure_pts", 0))
              + "/70 &middot; context " + str(score.get("context_pts", 0)) + "/30</summary>"
              "<div style='padding:8px 0 2px 0'>")
        for name, p, mx in score.get("components", []):
            pcol = "#22C55E" if p >= mx * 0.75 else ("#D4AF37" if p > 0 else "#6B6B6B")
            h += ("<div style='display:flex;justify-content:space-between;font-size:0.7rem;"
                  "line-height:1.7'><span style='color:#A1A1A6'>" + name + "</span>"
                  "<span style='color:" + pcol + ";font-weight:700'>%d/%d</span></div>" % (p, mx))
        for name, p in score.get("penalties", []):
            h += ("<div style='display:flex;justify-content:space-between;font-size:0.7rem;"
                  "line-height:1.7'><span style='color:#C1121F'>" + name + "</span>"
                  "<span style='color:#C1121F;font-weight:700'>%d</span></div>" % p)
        if score.get("time_mult", 1.0) != 1.0:
            h += ("<div style='color:#F6E27A;font-size:0.68rem;margin-top:5px'>Time-of-day x%.2f — %s</div>"
                  % (score.get("time_mult", 1.0), score.get("time_note", "")))
        if score.get("gated"):
            h += ("<div style='color:#F6E27A;font-size:0.68rem;margin-top:5px'>Capped: "
                  + str(score.get("gate_reason", "")) + "</div>")
        h += "</div></details></div>"
        return h
    except Exception as e:
        return ("<div style='color:#C1121F;font-size:0.72rem'>ORB card render error: "
                + str(e)[:80] + "</div>")


def render_orb_building_html(levels):
    """Pre-9:45 state — range not final yet, show context so you're prepped."""
    try:
        h = ("<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-left:3px solid #7AA2F7;"
             "border-radius:6px;padding:13px 16px;margin:10px 0'>"
             "<div style='display:flex;align-items:center;gap:10px;margin-bottom:9px'>"
             "<span style='color:#F5F5F5;font-size:1.05rem;font-weight:800;letter-spacing:1px'>"
             + str(levels.get("ticker", "")) + "</span>"
             "<span style='background:#7AA2F722;color:#7AA2F7;border:1px solid #7AA2F7;border-radius:3px;"
             "padding:2px 8px;font-size:0.6rem;font-weight:700;letter-spacing:1px'>RANGE BUILDING</span></div>"
             "<div style='color:#A1A1A6;font-size:0.75rem;line-height:1.5;margin-bottom:9px'>"
             "Opening range completes at 9:45 ET. Running high/low so far: "
             "<b style='color:#22C55E'>$%.2f</b> / <b style='color:#C1121F'>$%.2f</b></div>"
             % (levels.get("orb_high", 0), levels.get("orb_low", 0)))
        chips = []
        for lbl, key in (("PDH", "pdh"), ("PDL", "pdl"), ("PM Hi", "pm_high"), ("PM Lo", "pm_low")):
            v = levels.get(key)
            if v:
                chips.append("<span style='color:#A1A1A6'>" + lbl
                             + " <b style='color:#F5F5F5'>$%.2f</b></span>" % float(v))
        if chips:
            h += ("<div style='display:flex;gap:12px;flex-wrap:wrap;font-size:0.71rem'>"
                  + "".join(chips) + "</div>")
        h += "</div>"
        return h
    except Exception:
        return ""


def scan_orb_ticker(ticker, context_fn=None):
    """
    Full ORB read for one ticker. Returns a list of candidate dicts (0, 1, or 2 —
    one per boundary that actually broke). context_fn(ticker, direction) -> ctx dict.
    """
    out = []
    try:
        df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is None or len(df_5m) == 0:
            return out
        levels = orb_calc_levels(ticker, df_5m=df_5m)
        if not levels.get("available"):
            return out
        if levels.get("state") == "RANGE_BUILDING":
            return [{"ticker": ticker, "state": "RANGE_BUILDING", "levels": levels,
                     "score": 0, "bucket": "ON DECK", "side": ""}]

        events = orb_detect_events(levels, df_5m=df_5m)
        if not events.get("available"):
            return out

        for side in ("high", "low"):
            S = events.get(side, {})
            if not S.get("broken"):
                continue
            structure = orb_evaluate_structure(levels, events, side, df_5m=df_5m)
            structure["last_vwap"] = structure.get("last_vwap") or events.get("last_vwap", 0)
            structure["last_ema9"] = structure.get("last_ema9") or events.get("last_ema9", 0)
            geo = orb_calc_trade_geometry(levels, side, events=events)
            ctx = {}
            if context_fn:
                try:
                    ctx = context_fn(ticker, "bullish" if side == "high" else "bearish") or {}
                except Exception:
                    ctx = {}
            score = orb_score_setup(levels, events, structure, side, geo, ctx)
            out.append({
                "ticker": ticker, "side": side, "state": structure.get("state", ""),
                "levels": levels, "events": events, "structure": structure,
                "geo": geo, "score_data": score, "score": score.get("score", 0),
                "bucket": score.get("bucket", ""),
                "direction": "bullish" if side == "high" else "bearish",
                "last_price": levels.get("last_price", 0),
            })
        return out
    except Exception:
        return out


def orb_lookup_ticker(ticker, context_fn=None):
    """
    Single-ticker deep read. Unlike the scan, this ALWAYS returns a result even
    when nothing has broken — the pre-trigger state is the whole point of a lookup.
    Returns a dict with 'mode' one of: NO_DATA | RANGE_BUILDING | WATCHING_PRE |
    BROKE (has candidates). For BROKE, 'candidates' holds the same rows the scan builds.
    """
    res = {"ticker": ticker, "mode": "NO_DATA", "levels": {}, "candidates": [],
           "pre": {}, "note": ""}
    try:
        df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is None or len(df_5m) == 0:
            res["note"] = "No intraday data for " + ticker
            return res

        levels = orb_calc_levels(ticker, df_5m=df_5m)
        res["levels"] = levels
        if not levels.get("available"):
            res["note"] = levels.get("note", "No range data")
            return res

        if levels.get("state") == "RANGE_BUILDING":
            res["mode"] = "RANGE_BUILDING"
            return res

        events = orb_detect_events(levels, df_5m=df_5m)
        broke_any = events.get("high", {}).get("broken") or events.get("low", {}).get("broken")

        if broke_any:
            res["mode"] = "BROKE"
            for side in ("high", "low"):
                if not events.get(side, {}).get("broken"):
                    continue
                structure = orb_evaluate_structure(levels, events, side, df_5m=df_5m)
                if not structure.get("last_vwap"):
                    structure["last_vwap"] = events.get("last_vwap", 0)
                if not structure.get("last_ema9"):
                    structure["last_ema9"] = events.get("last_ema9", 0)
                geo = orb_calc_trade_geometry(levels, side, events=events)
                ctx = {}
                if context_fn:
                    try:
                        ctx = context_fn(ticker, "bullish" if side == "high" else "bearish") or {}
                    except Exception:
                        ctx = {}
                score = orb_score_setup(levels, events, structure, side, geo, ctx)
                res["candidates"].append({
                    "ticker": ticker, "side": side, "state": structure.get("state", ""),
                    "levels": levels, "events": events, "structure": structure,
                    "geo": geo, "score_data": score, "score": score.get("score", 0),
                    "bucket": score.get("bucket", ""),
                    "direction": "bullish" if side == "high" else "bearish",
                    "last_price": levels.get("last_price", 0),
                })
            res["candidates"].sort(key=lambda r: -r.get("score", 0))
            return res

        # No break yet — build the pre-trigger watch read
        res["mode"] = "WATCHING_PRE"
        price = levels.get("last_price", 0)
        oh, ol = levels.get("orb_high", 0), levels.get("orb_low", 0)
        vwap = events.get("last_vwap", 0)
        ema9 = events.get("last_ema9", 0)
        dist_high = ((oh - price) / price * 100) if price else 0
        dist_low  = ((price - ol) / price * 100) if price else 0
        # Which boundary is price nearer to?
        if abs(oh - price) <= abs(price - ol):
            near = "high"; trigger = oh; trig_dist = dist_high
        else:
            near = "low"; trigger = ol; trig_dist = dist_low
        res["pre"] = {
            "price": price, "orb_high": oh, "orb_low": ol,
            "dist_to_high_pct": dist_high, "dist_to_low_pct": dist_low,
            "near": near, "trigger": trigger, "trigger_dist_pct": trig_dist,
            "vwap": vwap, "ema9": ema9,
            "above_vwap": price >= vwap if vwap else None,
            "inside_range": ol <= price <= oh,
        }
        return res
    except Exception as e:
        res["note"] = "lookup error: " + str(e)[:60]
        return res


def render_orb_pre_html(res):
    """Pre-trigger watch card — price hasn't broken the range yet, but here's the read."""
    try:
        levels = res.get("levels", {})
        pre    = res.get("pre", {})
        tk     = res.get("ticker", "")
        near   = pre.get("near", "high")
        ncol   = "#22C55E" if near == "high" else "#C1121F"
        ndir   = "CALLS on a break above" if near == "high" else "PUTS on a break below"

        vwap_txt = ""
        if pre.get("above_vwap") is not None:
            vwap_txt = ("above VWAP" if pre["above_vwap"] else "below VWAP")

        h = ("<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-left:3px solid #7AA2F7;"
             "border-radius:6px;padding:14px 16px;margin:10px 0'>"
             "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:11px'>"
             "<span style='color:#F5F5F5;font-size:1.15rem;font-weight:800;letter-spacing:1px'>" + tk + "</span>"
             "<span style='background:#7AA2F722;color:#7AA2F7;border:1px solid #7AA2F7;border-radius:3px;"
             "padding:2px 8px;font-size:0.6rem;font-weight:700;letter-spacing:1px'>NO BREAK YET</span></div>")

        h += ("<div style='color:#F5F5F5;font-size:0.78rem;line-height:1.5;margin-bottom:11px'>"
              "Price <b>${price:.2f}</b>{inside}. Watching for a 5m close "
              "<b style='color:{ncol}'>{trigdir} ${trigger:.2f}</b> "
              "&mdash; that&rsquo;s <b>{tdist:.2f}%</b> away. {vwap}</div>").format(
                  price=pre.get("price", 0),
                  inside=(" sitting inside the range" if pre.get("inside_range") else " outside the range"),
                  ncol=ncol, trigdir=("above" if near == "high" else "below"),
                  trigger=pre.get("trigger", 0), tdist=abs(pre.get("trigger_dist_pct", 0)),
                  vwap=("Currently " + vwap_txt + "." if vwap_txt else ""))

        # Range + levels row
        h += ("<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>"
              "<div style='flex:1;min-width:84px;background:#111;border:1px solid #1F1F1F;border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>ORB HIGH</div>"
              "<div style='color:#22C55E;font-size:0.9rem;font-weight:700'>${oh:.2f}</div></div>"
              "<div style='flex:1;min-width:84px;background:#111;border:1px solid #1F1F1F;border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>ORB LOW</div>"
              "<div style='color:#C1121F;font-size:0.9rem;font-weight:700'>${ol:.2f}</div></div>"
              "<div style='flex:1;min-width:84px;background:#111;border:1px solid #1F1F1F;border-radius:4px;padding:7px 9px'>"
              "<div style='color:#6B6B6B;font-size:0.58rem;letter-spacing:1px'>PRICE</div>"
              "<div style='color:#F5F5F5;font-size:0.9rem;font-weight:700'>${px:.2f}</div></div></div>").format(
                  oh=pre.get("orb_high", 0), ol=pre.get("orb_low", 0), px=pre.get("price", 0))

        def _lvl(lbl, val):
            if not val:
                return ""
            return "<span style='color:#A1A1A6'>{l} <b style='color:#F5F5F5'>${v:.2f}</b></span>".format(l=lbl, v=float(val))
        chips = [_lvl("VWAP", pre.get("vwap")), _lvl("9EMA", pre.get("ema9")),
                 _lvl("PDH", levels.get("pdh")), _lvl("PDL", levels.get("pdl")),
                 _lvl("200MA", levels.get("ma200")), _lvl("400MA", levels.get("ma400"))]
        chips = [c for c in chips if c]
        if chips:
            h += ("<div style='display:flex;gap:11px;flex-wrap:wrap;font-size:0.71rem;"
                  "padding-top:8px;border-top:1px solid #1A1A1A'>" + "".join(chips) + "</div>")
        h += "</div>"
        return h
    except Exception as e:
        return "<div style='color:#C1121F;font-size:0.7rem'>pre-card error: " + str(e)[:70] + "</div>"




def orb_strike_expiration(levels, geo, side, vol_class=None):
    """
    Strike + expiration + estimated contract move, driven by the ORB trade geometry.
    Entry = ORB level, Target = next structural level, Move = target - entry.
    Uses a delta approximation (no live chain). Returns three sized options.
    NOTE: contract-move dollars are ESTIMATES from delta, not live option prices.
    """
    res = {"available": False, "entry": 0.0, "target": 0.0, "stop": 0.0,
           "move": 0.0, "move_pct": 0.0, "options": [], "note": ""}
    try:
        entry  = geo.get("entry", 0.0)
        target = geo.get("target", 0.0)
        stop   = geo.get("stop", 0.0)
        if entry <= 0 or target <= 0:
            res["note"] = "geometry unavailable"
            return res

        bull = (side == "high")
        move = (target - entry) if bull else (entry - target)
        if move <= 0:
            res["note"] = "no room to target"
            return res

        res.update({"entry": entry, "target": target, "stop": stop,
                    "move": move, "move_pct": (move / entry) * 100.0})

        def _incr(p):
            if p >= 500: return 5.0
            if p >= 100: return 1.0
            if p >= 25:  return 0.5
            return 1.0
        inc = _incr(entry)
        def _round(v):
            return round(round(v / inc) * inc, 2)

        mp = res["move_pct"]
        if mp <= 0.6:
            dte_c, dte_b, dte_a = "1-2 DTE", "0-1 DTE", "0 DTE"
        elif mp <= 1.2:
            dte_c, dte_b, dte_a = "2-3 DTE", "1-2 DTE", "0-1 DTE"
        else:
            dte_c, dte_b, dte_a = "3-5 DTE", "2-3 DTE", "1-2 DTE"

        if bull:
            k_cons, k_bal, k_aggr = _round(entry - inc), _round(entry), _round(entry + inc)
        else:
            k_cons, k_bal, k_aggr = _round(entry + inc), _round(entry), _round(entry - inc)

        specs = [
            ("Conservative", k_cons, 0.65, dte_c,
             "Slightly ITM — higher delta, less theta risk, costs more."),
            ("Balanced",     k_bal,  0.50, dte_b,
             "At the money — balanced cost and sensitivity."),
            ("Aggressive",   k_aggr, 0.38, dte_a,
             "Slightly OTM — cheaper, needs the move to actually hit."),
        ]
        opts = []
        for label, strike, delta, dte, why in specs:
            opts.append({
                "label": label, "strike": strike, "delta": delta,
                "dte": dte, "why": why,
                "est_contract_move": round(move * delta * 100.0, 0),
                "type": "CALL" if bull else "PUT",
            })
        res["options"] = opts
        res["available"] = True
        return res
    except Exception as e:
        res["note"] = "strike calc error: " + str(e)[:50]
        return res


def render_orb_strike_html(strike_data, side):
    """Render the strike/expiration block. Uses .format() throughout to avoid
    the %-operator precedence trap with string concatenation."""
    try:
        if not strike_data.get("available"):
            return ""
        acol = "#22C55E" if side == "high" else "#C1121F"

        rows = ""
        for o in strike_data.get("options", []):
            rows += (
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "gap:8px;padding:5px 0;border-top:1px solid #151515;font-size:0.72rem'>"
                "<span style='color:#A1A1A6;min-width:78px'>{label}</span>"
                "<span style='color:{acol};font-weight:700;min-width:92px'>${strike:.2f} {typ}</span>"
                "<span style='color:#6B6B6B;min-width:64px'>{dte}</span>"
                "<span style='color:#6B6B6B'>~{delta:.0f}&Delta;</span>"
                "<span style='color:#22C55E;font-weight:700'>&asymp;${est}/contract</span></div>"
            ).format(label=o["label"], acol=acol, strike=o["strike"], typ=o["type"],
                     dte=o["dte"], delta=o["delta"] * 100,
                     est="{:,.0f}".format(o["est_contract_move"]))

        html = (
            "<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:4px;"
            "padding:9px 11px;margin-bottom:11px'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:7px'>"
            "<span style='color:#D4AF37;font-size:0.62rem;font-weight:700;letter-spacing:1.2px'>"
            "STRIKE &amp; EXPIRATION</span>"
            "<span style='color:#6B6B6B;font-size:0.62rem'>move to target ${move:.2f} ({movepct:.1f}%)</span></div>"
            "{rows}"
            "<div style='color:#6B6B6B;font-size:0.63rem;margin-top:6px;line-height:1.4'>"
            "Contract move is estimated from delta for the full run to target &mdash; "
            "not a live option quote. Strikes rounded to tradeable increments.</div></div>"
        ).format(move=strike_data.get("move", 0), movepct=strike_data.get("move_pct", 0), rows=rows)
        return html
    except Exception as e:
        return "<div style='color:#C1121F;font-size:0.7rem'>strike render error: " + str(e)[:70] + "</div>"




def orb_build_context(ticker, direction, df_5m=None, last_price=0.0):
    """
    Bridge the ORB scorer to the features already in the screener.
    Every source is isolated — a failure records itself in _errors instead of
    silently returning empty. Context modifies the score; it never creates a signal.
    """
    ctx = {"news_direction": "", "news_strength": 0.0, "squeeze_active": False,
           "squeeze_direction": "", "market_bias": "", "vol_tier": "",
           "block_in_path": False, "_errors": []}

    try:
        untradeable, _reason, news, _adj, _flip, _fr = run_news_check(ticker, direction)
        sent = (news or {}).get("sentiment", "neutral")
        raw  = abs(float((news or {}).get("score", 0)))
        ctx["news_direction"] = sent
        ctx["news_strength"]  = max(0.0, min(1.0, raw / 60.0))
        if untradeable:
            ctx["news_direction"] = "bearish" if direction == "bullish" else "bullish"
            ctx["news_strength"]  = 1.0
    except Exception as e:
        ctx["_errors"].append("news: " + str(e)[:45])

    try:
        mb = get_weekly_macro_bias() or {}
        ov = str(mb.get("overall", "NEUTRAL")).lower()
        ctx["market_bias"] = ov if ov in ("bullish", "bearish") else "neutral"
    except Exception as e:
        ctx["_errors"].append("bias: " + str(e)[:45])

    try:
        vc = classify_stock_volatility(ticker, last_price or None) or {}
        ctx["vol_tier"] = vc.get("tier", "")
    except Exception as e:
        ctx["_errors"].append("vol: " + str(e)[:45])

    try:
        if df_5m is None:
            df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is not None and len(df_5m) >= 25:
            state, comp = detect_squeeze(df_5m, direction)
            ctx["squeeze_active"]    = state in ("fired", "firing", "confirmed", "squeeze")
            ctx["squeeze_direction"] = direction if ctx["squeeze_active"] else ""
    except Exception as e:
        ctx["_errors"].append("squeeze: " + str(e)[:45])

    try:
        _, df_rth, _ = orb_session_frames(df_5m)
        if df_rth is not None and len(df_rth) >= 6:
            vols = [float(v) for v in df_rth["volume"].tolist()]
            avg  = sum(vols[:-1]) / max(1, len(vols) - 1)
            recent = vols[-3:]
            if avg > 0 and max(recent) >= avg * 2.5:
                idx  = vols.index(max(recent))
                bar  = df_rth.iloc[idx]
                rng  = float(bar["high"]) - float(bar["low"])
                body = abs(float(bar["close"]) - float(bar["open"]))
                # Heavy volume with a small body = absorption sitting in the path
                if rng > 0 and (body / rng) < 0.35:
                    ctx["block_in_path"] = True
    except Exception as e:
        ctx["_errors"].append("block: " + str(e)[:45])

    return ctx


def run_orb_scan(tickers, max_workers=4, progress_cb=None):
    """Scan the universe for ORB structure. Returns (candidates, building, stats)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cands, building = [], []
    done, errors = 0, 0

    def _one(tk):
        try:
            df_5m = _fmp_download(tk, "5d", "5m")
            if df_5m is None or len(df_5m) == 0:
                return tk, []
            levels = orb_calc_levels(tk, df_5m=df_5m)
            if not levels.get("available"):
                return tk, []
            if levels.get("state") == "RANGE_BUILDING":
                return tk, [{"_building": True, "levels": levels}]

            events = orb_detect_events(levels, df_5m=df_5m)
            if not events.get("available"):
                return tk, []
            rows = []
            for side in ("high", "low"):
                if not events.get(side, {}).get("broken"):
                    continue
                structure = orb_evaluate_structure(levels, events, side, df_5m=df_5m)
                if not structure.get("last_vwap"):
                    structure["last_vwap"] = events.get("last_vwap", 0)
                if not structure.get("last_ema9"):
                    structure["last_ema9"] = events.get("last_ema9", 0)
                geo = orb_calc_trade_geometry(levels, side, events=events)
                ctx = orb_build_context(tk, "bullish" if side == "high" else "bearish",
                                        df_5m=df_5m, last_price=levels.get("last_price", 0))
                score = orb_score_setup(levels, events, structure, side, geo, ctx)
                rows.append({
                    "ticker": tk, "side": side, "state": structure.get("state", ""),
                    "levels": levels, "events": events, "structure": structure,
                    "geo": geo, "score_data": score, "score": score.get("score", 0),
                    "bucket": score.get("bucket", ""), "ctx": ctx,
                    "direction": "bullish" if side == "high" else "bearish",
                    "last_price": levels.get("last_price", 0),
                })
            return tk, rows
        except Exception:
            return tk, None

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pbp_orb") as ex:
        futs = {ex.submit(_one, t): t for t in tickers}
        for f in as_completed(futs):
            tk, rows = f.result()
            done += 1
            if rows is None:
                errors += 1
            elif rows:
                for r in rows:
                    (building if r.get("_building") else cands).append(r)
            if progress_cb:
                try:
                    progress_cb(done, len(tickers), tk)
                except Exception:
                    pass

    cands.sort(key=lambda r: (-r.get("score", 0), r.get("ticker", "")))
    stats = {"scanned": len(tickers), "errors": errors,
             "with_breaks": len(set(c["ticker"] for c in cands)),
             "building": len(building)}
    return cands, building, stats


# OPENING MOMENTUM ENGINE — pre-range thrust detection (9:30–9:45 window only)
# Calibrated on 7/27/2026 exact 09:30 candles:
#   AMD  range $8.41, closed $1 below VWAP  -> ran ~$48  (Signal A, immediate thrust)
#   TSLA range $4.38, closed above VWAP, broke by 9:50   (Signal B, delayed break)
#   SPY  range $1.23, pinned to VWAP        -> bled ~$6  (null / lowest conviction)
# This mode SHOWS momentum with loud caution labeling — it does not gate it out.

MOM_WINDOW_START = 9 * 60 + 30      # 9:30
MOM_WINDOW_END   = 9 * 60 + 45      # 9:45 — mode goes dormant after this
# Candle-one range must be this multiple of the name's normal opening range to count as a thrust.
MOM_RANGE_MULT_STRONG = 2.5         # AMD-class
MOM_RANGE_MULT_MOD    = 1.6         # meaningful but not violent
# Relative volume multiples (vs the name's own recent opening-bar volume).
MOM_RVOL_STRONG = 3.0
MOM_RVOL_MOD    = 1.8
# How decisively candle one must close beyond VWAP, as a fraction of that candle's range.
MOM_VWAP_DECISIVE = 0.15            # close is ≥15% of the bar's range beyond VWAP
# Suggested starter size as a PERCENTAGE of the trader's normal ORB size.
MOM_SIZE_STRONG = 50                # STRONG conviction -> up to 50%
MOM_SIZE_MOD    = 33
MOM_SIZE_SPEC   = 20                # SPECULATIVE -> 20% or less


def _mom_minutes(ts):
    try:
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:
        return -1


def mom_normal_open_range(ticker, df_5m, session_date):
    """
    The name's TYPICAL 9:30 candle range and volume, from prior sessions.
    Returns (avg_range, avg_vol) — the baseline the live open is judged against.
    Falls back to (0,0) if not enough history.
    """
    try:
        d = df_5m.copy()
        d["_date"] = d["datetime"].dt.date
        d["_min"]  = d["datetime"].apply(_mom_minutes)
        opens = d[(d["_min"] == MOM_WINDOW_START) & (d["_date"] < session_date)]
        if len(opens) == 0:
            return 0.0, 0.0
        rng = (opens["high"].astype(float) - opens["low"].astype(float))
        vol = opens["volume"].astype(float)
        # Use median — robust to the odd gap day skewing the mean.
        return float(rng.median()), float(vol.median())
    except Exception:
        return 0.0, 0.0


def mom_scan_open(ticker, df_5m=None, market_ctx=None):
    """
    Read the 9:30–9:45 window for an opening-momentum setup.
    market_ctx (optional): {"spy_dir": "up"|"down"|"flat", "sector_dir": ...,
                            "catalyst": bool, "catalyst_dir": "bullish"|"bearish"}
    Returns dict; 'available' False when outside the window or no data.
    'signal' one of: NONE | A_THRUST | B_DELAYED
    """
    res = {"available": False, "ticker": ticker, "signal": "NONE",
           "direction": "", "conviction": "", "score": 0, "size_pct": 0,
           "factors": [], "candle1": {}, "note": "", "in_window": False,
           "vwap_c1": 0.0, "range_c1": 0.0, "rvol": 0.0}
    try:
        if df_5m is None:
            df_5m = _fmp_download(ticker, "5d", "5m")
        if df_5m is None or len(df_5m) == 0:
            res["note"] = "no intraday data"
            return res

        d = df_5m.copy()
        d["_date"] = d["datetime"].dt.date
        sess = d["_date"].max()
        d = d[d["_date"] == sess].copy()
        d["_min"] = d["datetime"].apply(_mom_minutes)
        rth = d[(d["_min"] >= MOM_WINDOW_START) & (d["_min"] < 16 * 60)].reset_index(drop=True)
        if len(rth) == 0:
            res["note"] = "no RTH bars"
            return res

        last_min = int(rth["_min"].iloc[-1])
        # Mode is only live during the opening window. After 9:45 it's dormant
        # (ORB takes over) — but we still report what it WOULD have flagged for the handoff.
        res["in_window"] = last_min < MOM_WINDOW_END

        c1 = rth.iloc[0]
        o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
        v1 = float(c1["volume"]) or 0.0
        rng1 = h1 - l1
        res["candle1"] = {"open": o1, "high": h1, "low": l1, "close": cl1, "volume": v1}
        res["range_c1"] = rng1

        # VWAP over the window so far
        vwaps = orb_calc_vwap(rth)  # reuse ORB's VWAP
        vwap1 = vwaps[0] if vwaps else cl1
        res["vwap_c1"] = vwap1

        # Baselines from prior sessions
        norm_rng, norm_vol = mom_normal_open_range(ticker, df_5m, sess)
        rvol = (v1 / norm_vol) if norm_vol > 0 else 0.0
        range_mult = (rng1 / norm_rng) if norm_rng > 0 else 0.0
        res["rvol"] = round(rvol, 2)

        bar_rng = max(rng1, 1e-9)
        # Direction + decisiveness of candle-one VWAP close
        below = cl1 < vwap1
        above = cl1 > vwap1
        vwap_dist_frac = abs(cl1 - vwap1) / bar_rng
        decisive = vwap_dist_frac >= MOM_VWAP_DECISIVE
        direction = "bearish" if below else ("bullish" if above else "")
        res["direction"] = direction

        # ---- Factor scorecard (lit/dark) ----
        factors = []
        # X — relative volume
        if rvol >= MOM_RVOL_STRONG:
            factors.append(("X", True,  "RVOL %.1fx — heavy" % rvol))
        elif rvol >= MOM_RVOL_MOD:
            factors.append(("X", True,  "RVOL %.1fx — elevated" % rvol))
        elif rvol > 0:
            factors.append(("X", False, "RVOL %.1fx — ordinary" % rvol))
        else:
            factors.append(("X", False, "RVOL unavailable"))
        # (range multiple rides alongside X as the "how big is candle one" read)
        if range_mult >= MOM_RANGE_MULT_STRONG:
            factors.append(("range", True,  "Opening candle %.1fx normal — violent" % range_mult))
        elif range_mult >= MOM_RANGE_MULT_MOD:
            factors.append(("range", True,  "Opening candle %.1fx normal" % range_mult))
        elif range_mult > 0:
            factors.append(("range", False, "Opening candle %.1fx normal — quiet" % range_mult))
        else:
            factors.append(("range", False, "No range baseline"))
        # Y — VWAP held / decisive close
        if direction and decisive:
            factors.append(("Y", True,  "Closed decisively %s VWAP" % ("below" if below else "above")))
        elif direction:
            factors.append(("Y", False, "Near VWAP — not decisive"))
        else:
            factors.append(("Y", False, "Pinned to VWAP — no commit"))
        # Z — one-directional candle (open near one end, close near the other)
        if direction == "bearish":
            body_pos = (o1 - l1) / bar_rng   # open near high, close near low = strong
            clean = (o1 >= h1 - bar_rng * 0.35) and (cl1 <= l1 + bar_rng * 0.35)
        elif direction == "bullish":
            clean = (o1 <= l1 + bar_rng * 0.35) and (cl1 >= h1 - bar_rng * 0.35)
        else:
            clean = False
        factors.append(("Z", clean, "Clean one-way candle" if clean else "Wicky / two-sided candle"))
        # Z1 — catalyst
        mc = market_ctx or {}
        cat = bool(mc.get("catalyst"))
        cat_aligned = cat and (mc.get("catalyst_dir", "") == direction)
        factors.append(("Z1", cat_aligned,
                        "Catalyst aligned" if cat_aligned else ("Catalyst opposes" if cat else "No catalyst")))
        # Z2 — broad-market agreement
        want = "down" if direction == "bearish" else ("up" if direction == "bullish" else "")
        spy_ok = want and (mc.get("spy_dir", "") == want)
        sec_ok = want and (mc.get("sector_dir", "") == want)
        market_agree = bool(spy_ok or sec_ok)
        factors.append(("Z2", market_agree,
                        "Market/sector agrees" if market_agree else "No broad-market confirmation"))
        # Z3 — room to run (not already exhausted on candle one)
        # If candle one already moved > ~1.2x the name's normal FULL opening range, it may be spent.
        exhausted = norm_rng > 0 and (rng1 > norm_rng * 3.5)
        factors.append(("Z3", not exhausted,
                        "Room to run" if not exhausted else "May be exhausted — big candle already"))
        res["factors"] = factors

        # ---- Signal classification ----
        core_ok = (rvol >= MOM_RVOL_MOD or range_mult >= MOM_RANGE_MULT_MOD)
        if not direction:
            res["signal"] = "NONE"
            res["note"] = "Opening candle pinned to VWAP — no directional thrust"
            # SPY-type null still returns available so the board can show "nothing here"
            res["available"] = True
            return res

        # Signal A — immediate thrust: big candle, decisive VWAP break, whole move in candle one
        strong_range = range_mult >= MOM_RANGE_MULT_STRONG or rng1 >= (norm_rng * 2.5 if norm_rng else rng1)
        if strong_range and decisive and clean:
            res["signal"] = "A_THRUST"
        elif core_ok and direction:
            # Signal B — delayed/forming: meaningful candle, direction set, but not the
            # violent one-candle break. Often resolves into the ORB break by 9:45–9:50.
            res["signal"] = "B_DELAYED"
        else:
            res["signal"] = "NONE"
            res["note"] = "Directional but below momentum thresholds — likely just noise"
            res["available"] = True
            return res

        # ---- Conviction + score from factors lit ----
        lit = sum(1 for _, ok, _ in factors if ok)
        total = len(factors)
        res["score"] = int(round(lit / total * 100))
        if res["signal"] == "A_THRUST" and lit >= 5:
            res["conviction"], res["size_pct"] = "STRONG", MOM_SIZE_STRONG
        elif lit >= 4:
            res["conviction"], res["size_pct"] = "MODERATE", MOM_SIZE_MOD
        else:
            res["conviction"], res["size_pct"] = "SPECULATIVE", MOM_SIZE_SPEC

        res["available"] = True
        return res
    except Exception as e:
        res["note"] = "momentum error: " + str(e)[:60]
        return res


MOM_SIG_STYLE = {
    "A_THRUST":  ("#C1121F", "IMMEDIATE THRUST"),
    "B_DELAYED": ("#F6E27A", "DELAYED / FORMING"),
    "NONE":      ("#6B6B6B", "NO THRUST"),
}
MOM_CONV_COLOR = {"STRONG": "#22C55E", "MODERATE": "#D4AF37", "SPECULATIVE": "#F6E27A"}


def render_momentum_card_html(m):
    """Opening-momentum card — loud caution, factor scorecard, two-leg sizing."""
    try:
        sig = m.get("signal", "NONE")
        if sig == "NONE":
            return ""  # nulls handled by the tab as a summary line, not a card
        scol, slabel = MOM_SIG_STYLE.get(sig, MOM_SIG_STYLE["NONE"])
        direction = m.get("direction", "")
        acol = "#22C55E" if direction == "bullish" else "#C1121F"
        adir = "CALLS" if direction == "bullish" else "PUTS"
        conv = m.get("conviction", "")
        ccol = MOM_CONV_COLOR.get(conv, "#6B6B6B")

        h = ("<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-left:3px solid "
             + scol + ";border-radius:6px;padding:14px 16px;margin:10px 0'>")

        # Header
        h += ("<div style='display:flex;align-items:center;justify-content:space-between;"
              "flex-wrap:wrap;gap:8px;margin-bottom:10px'>"
              "<div style='display:flex;align-items:center;gap:9px'>"
              "<span style='color:#F5F5F5;font-size:1.12rem;font-weight:800;letter-spacing:1px'>"
              + str(m.get("ticker", "")) + "</span>"
              "<span style='background:" + acol + "22;color:" + acol + ";border:1px solid " + acol
              + ";border-radius:3px;padding:2px 7px;font-size:0.6rem;font-weight:700;letter-spacing:1px'>"
              + adir + "</span>"
              "<span style='background:" + scol + "22;color:" + scol + ";border:1px solid " + scol
              + ";border-radius:3px;padding:2px 7px;font-size:0.58rem;font-weight:700;letter-spacing:1px'>"
              "\u26A1 " + slabel + "</span></div>"
              "<div style='display:flex;align-items:center;gap:8px'>"
              "<span style='background:" + ccol + "22;color:" + ccol + ";border:1px solid " + ccol
              + ";border-radius:3px;padding:2px 8px;font-size:0.6rem;font-weight:700;letter-spacing:1px'>"
              + conv + "</span></div></div>")

        # LOUD caution line — on every card, every time
        h += ("<div style='background:#C1121F14;border:1px solid #C1121F55;border-radius:4px;"
              "padding:7px 10px;margin-bottom:11px'>"
              "<div style='color:#F87171;font-size:0.7rem;line-height:1.45'>"
              "\u26A0 <b>EARLY &middot; UNCONFIRMED &middot; CAN REVERSE.</b> The open is the most "
              "violent part of the day. This has NOT been confirmed by the opening range yet. "
              "Size small, confirm with the range.</div></div>")

        # Two-leg plan
        h += ("<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:4px;"
              "padding:9px 11px;margin-bottom:11px'>"
              "<div style='color:#D4AF37;font-size:0.62rem;font-weight:700;letter-spacing:1.2px;"
              "margin-bottom:6px'>TWO-LEG PLAN</div>"
              "<div style='font-size:0.73rem;line-height:1.55'>"
              "<div style='color:#F5F5F5'><b style='color:" + ccol + "'>Leg 1 (now):</b> starter at "
              "<b>" + str(m.get("size_pct", 0)) + "% of your normal ORB size</b> — this is the "
              "unconfirmed momentum entry.</div>"
              "<div style='color:#F5F5F5;margin-top:3px'><b style='color:#22C55E'>Leg 2 (on ORB "
              "confirm):</b> if the range breaks the same direction after 9:45, add the second "
              "leg — your starter is already working.</div></div></div>")

        # Factor scorecard (lit/dark)
        h += ("<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:4px;"
              "padding:9px 11px;margin-bottom:8px'>"
              "<div style='color:#D4AF37;font-size:0.62rem;font-weight:700;letter-spacing:1.2px;"
              "margin-bottom:6px'>MOMENTUM FACTORS &middot; " + str(m.get("score", 0)) + "/100</div>")
        for tag, ok, txt in m.get("factors", []):
            dot = "#22C55E" if ok else "#3A3A3A"
            tcol = "#F5F5F5" if ok else "#6B6B6B"
            mark = "\u2713" if ok else "\u00B7"
            h += ("<div style='display:flex;align-items:center;gap:8px;font-size:0.71rem;line-height:1.7'>"
                  "<span style='color:" + dot + ";font-weight:700;min-width:12px'>" + mark + "</span>"
                  "<span style='color:#6B6B6B;min-width:40px;font-size:0.6rem;letter-spacing:0.5px'>"
                  + str(tag).upper() + "</span>"
                  "<span style='color:" + tcol + "'>" + txt + "</span></div>")
        h += "</div>"

        # Candle-one facts
        c1 = m.get("candle1", {})
        h += ("<div style='display:flex;gap:11px;flex-wrap:wrap;font-size:0.68rem;color:#6B6B6B;"
              "padding-top:6px;border-top:1px solid #1A1A1A'>"
              "<span>Open candle range <b style='color:#F5F5F5'>$%.2f</b></span>"
              "<span>Close <b style='color:%s'>$%.2f</b></span>"
              "<span>VWAP <b style='color:#F5F5F5'>$%.2f</b></span>"
              "<span>RVOL <b style='color:#F5F5F5'>%.1fx</b></span></div>"
              % (m.get("range_c1", 0), acol, c1.get("close", 0),
                 m.get("vwap_c1", 0), m.get("rvol", 0)))
        h += "</div>"
        return h
    except Exception as e:
        return ("<div style='color:#C1121F;font-size:0.72rem'>momentum card error: "
                + str(e)[:80] + "</div>")




def mom_market_context(direction_hint=None):
    """
    Broad-market read for the momentum factors, computed ONCE per scan and reused
    for every ticker (SPY/sector direction doesn't change per name).
    Returns {"spy_dir","sector_dir","catalyst_by_ticker_fn"...}. Isolated failures.
    """
    ctx = {"spy_dir": "flat", "qqq_dir": "flat", "_errors": []}
    try:
        # SPY opening-candle direction from its own 5m
        spy = _fmp_download("SPY", "5d", "5m")
        if spy is not None and len(spy) > 0:
            d = spy.copy(); d["_date"] = d["datetime"].dt.date
            d = d[d["_date"] == d["_date"].max()]
            if len(d) > 0:
                c1 = d.iloc[0]
                op, cl = float(c1["open"]), float(c1["close"])
                ch = (cl - op) / op * 100 if op else 0
                ctx["spy_dir"] = "up" if ch > 0.05 else ("down" if ch < -0.05 else "flat")
    except Exception as e:
        ctx["_errors"].append("spy: " + str(e)[:40])
    try:
        qqq = _fmp_download("QQQ", "5d", "5m")
        if qqq is not None and len(qqq) > 0:
            d = qqq.copy(); d["_date"] = d["datetime"].dt.date
            d = d[d["_date"] == d["_date"].max()]
            if len(d) > 0:
                c1 = d.iloc[0]
                op, cl = float(c1["open"]), float(c1["close"])
                ch = (cl - op) / op * 100 if op else 0
                ctx["qqq_dir"] = "up" if ch > 0.05 else ("down" if ch < -0.05 else "flat")
    except Exception as e:
        ctx["_errors"].append("qqq: " + str(e)[:40])
    return ctx


def mom_ticker_context(ticker, direction, base_ctx):
    """Per-ticker momentum context: sector proxy = QQQ for tech names, plus news catalyst."""
    mc = {"spy_dir": base_ctx.get("spy_dir", "flat"),
          "sector_dir": base_ctx.get("qqq_dir", "flat"),
          "catalyst": False, "catalyst_dir": ""}
    try:
        untradeable, _r, news, _a, _f, _fr = run_news_check(ticker, direction)
        sent = (news or {}).get("sentiment", "neutral")
        strong = abs(float((news or {}).get("score", 0))) >= 25
        if sent in ("bullish", "bearish") and strong:
            mc["catalyst"] = True
            mc["catalyst_dir"] = sent
    except Exception:
        pass
    return mc


def run_momentum_scan(tickers, max_workers=4, progress_cb=None):
    """Scan the universe for opening-momentum setups. Returns (thrust, delayed, nulls, stats)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    base = mom_market_context()
    thrust, delayed, nulls = [], [], []
    done, errors = 0, 0

    def _one(tk):
        try:
            df5 = _fmp_download(tk, "5d", "5m")
            if df5 is None or len(df5) == 0:
                return tk, None
            # Direction hint from candle one so news context can align
            d = df5.copy(); d["_date"] = d["datetime"].dt.date
            d = d[d["_date"] == d["_date"].max()].reset_index(drop=True)
            hint = ""
            if len(d) > 0:
                c1 = d.iloc[0]
                hint = "bearish" if float(c1["close"]) < float(c1["open"]) else "bullish"
            mc = mom_ticker_context(tk, hint, base)
            m = mom_scan_open(tk, df_5m=df5, market_ctx=mc)
            return tk, m
        except Exception:
            return tk, "ERR"

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pbp_mom") as ex:
        futs = {ex.submit(_one, t): t for t in tickers}
        for f in as_completed(futs):
            tk, m = f.result()
            done += 1
            if m == "ERR" or m is None:
                if m == "ERR":
                    errors += 1
            elif m.get("available"):
                sig = m.get("signal")
                if sig == "A_THRUST":
                    thrust.append(m)
                elif sig == "B_DELAYED":
                    delayed.append(m)
                # NONE = null, we don't surface cards but count them
            if progress_cb:
                try:
                    progress_cb(done, len(tickers), tk)
                except Exception:
                    pass

    thrust.sort(key=lambda x: -x.get("score", 0))
    delayed.sort(key=lambda x: -x.get("score", 0))
    stats = {"scanned": len(tickers), "errors": errors,
             "thrust": len(thrust), "delayed": len(delayed),
             "spy_dir": base.get("spy_dir", "flat"), "qqq_dir": base.get("qqq_dir", "flat")}
    return thrust, delayed, nulls, stats


def render_news_sentiment_html(news_data, ticker, signal_direction=None,
                                flip_signal=False, flip_reason="", conf_adj=0):
    if not news_data or not news_data.get("article_count"):
        return (
            "<div style='background:#1A1A1D;border:1px solid #2A2A2D;"
            "border-radius:8px;padding:10px 14px;margin-top:8px'>"
            "<div style='color:#A1A1A6;font-family:monospace;font-size:0.68rem;"
            "letter-spacing:1px;margin-bottom:4px'>NEWS SENTIMENT</div>"
            "<div style='color:#4a5568;font-size:0.75rem'>"
            "No recent news — trade on technicals only</div>"
            "</div>"
        )

    sentiment    = news_data.get("sentiment", "neutral")
    score        = news_data.get("score", 0)
    flags        = news_data.get("flags", [])
    headlines    = news_data.get("headlines", [])
    count        = news_data.get("article_count", 0)
    untradeable  = news_data.get("untradeable", False)
    is_breaking  = news_data.get("is_breaking", False)
    recent_count = news_data.get("recent_count", 0)

    if untradeable:
        border = "#C1121F"; label_color = "#C1121F"
        label  = "🚫 UNTRADEABLE — HALT/DELISTED"; bg = "#1a0505"
    elif sentiment == "bullish":
        border = "#D4AF37"; label_color = "#D4AF37"
        label  = "📈 BULLISH NEWS · CALL EDGE"; bg = "#1A1500"
    elif sentiment == "bearish":
        border = "#C1121F"; label_color = "#C1121F"
        label  = "📉 BEARISH NEWS · PUT EDGE"; bg = "#1a0a0a"
    else:
        border = "#2A2A2D"; label_color = "#A1A1A6"
        label  = "↔️ NEUTRAL NEWS"; bg = "#1A1A1D"

    breaking_badge = ""
    if is_breaking:
        breaking_badge = (
            "<span style='background:#C1121F;color:#F5F5F5;font-size:0.6rem;"
            "font-weight:700;padding:1px 7px;border-radius:4px;margin-left:6px'>"
            "🔴 BREAKING · %s articles/15min</span>" % recent_count
        )

    adj_html = ""
    if conf_adj > 0:
        adj_html = ("<span style='background:#D4AF3722;color:#D4AF37;font-size:0.62rem;"
                    "padding:1px 7px;border-radius:4px;margin-left:6px'>+%s conf</span>" % conf_adj)
    elif conf_adj < 0:
        adj_html = ("<span style='background:#C1121F22;color:#C1121F;font-size:0.62rem;"
                    "padding:1px 7px;border-radius:4px;margin-left:6px'>%s conf</span>" % conf_adj)

    bar_pct   = min(100, abs(score))
    bar_color = "#D4AF37" if score > 0 else "#C1121F" if score < 0 else "#A1A1A6"
    score_str = "%s%s" % ("+" if score > 0 else "", score)

    flags_html = ""
    for f in flags[:5]:
        fc = "#C1121F" if (f.startswith("🔴") or f.startswith("🚫")) else "#D4AF37"
        flags_html += (
            "<span style='background:%s22;color:%s;border:1px solid %s44;"
            "padding:1px 6px;border-radius:4px;font-size:0.6rem;"
            "margin:2px 2px;display:inline-block'>%s</span>" % (fc, fc, fc, f)
        )

    headlines_html = ""
    for h in headlines[:3]:
        site  = h.get("site", "")
        title = h.get("title", "")
        title = title[:75] + "…" if len(title) > 75 else title
        headlines_html += (
            "<div style='margin:3px 0;font-size:0.69rem;line-height:1.4'>"
            "<span style='color:#4a5568;font-size:0.62rem'>%s</span> "
            "<span style='color:#A1A1A6'>%s</span></div>" % (site, title)
        )

    flip_html = ""
    if flip_signal and flip_reason:
        flip_action = "PUT" if signal_direction == "bullish" else "CALL"
        flip_color  = "#C1121F" if flip_action == "PUT" else "#D4AF37"
        flip_html = (
            "<div style='background:%s22;border:1px solid %s;border-radius:6px;"
            "padding:8px 12px;margin-top:8px'>"
            "<div style='color:%s;font-weight:700;font-size:0.75rem;margin-bottom:3px'>"
            "💡 NEWS SUGGESTS %s</div>"
            "<div style='color:#A1A1A6;font-size:0.72rem'>%s</div>"
            "</div>" % (flip_color, flip_color, flip_color, flip_action, flip_reason)
        )

    return (
        "<div style='background:%s;border:1px solid %s;border-radius:8px;"
        "padding:10px 14px;margin-top:8px'>"
        "<div style='display:flex;justify-content:space-between;"
        "align-items:center;margin-bottom:6px'>"
        "<span style='color:#A1A1A6;font-family:monospace;font-size:0.68rem;"
        "letter-spacing:1px'>NEWS SENTIMENT</span>"
        "<div><span style='color:%s;font-weight:700;font-size:0.75rem'>%s</span>%s%s</div>"
        "</div>"
        "<div style='background:#2A2A2D;border-radius:4px;height:4px;margin-bottom:8px'>"
        "<div style='background:%s;height:4px;border-radius:4px;width:%s%%'></div>"
        "</div>"
        "<div style='font-size:0.69rem;color:#A1A1A6;margin-bottom:6px'>"
        "%s articles · Score <b style='color:%s'>%s</b></div>"
        "<div style='margin-bottom:6px'>%s</div>"
        "<div style='padding-top:6px;border-top:1px solid #2A2A2D'>%s</div>"
        "%s"
        "</div>"
    ) % (
        bg, border,
        label_color, label, adj_html, breaking_badge,
        bar_color, bar_pct,
        count, bar_color, score_str,
        flags_html or "<span style='color:#4a5568;font-size:0.69rem'>No strong keyword signals</span>",
        headlines_html or "<span style='color:#4a5568;font-size:0.69rem'>No headlines available</span>",
        flip_html,
    )

def precision_score(ticker, direction, df_primary, df_confirm,
                    iv_rank, earnings_days, market_bias,
                    sector_bias, atr, dte, account_size, risk_pct,
                    trade_style, current_price=None, signals_only=False):
    import pytz
    from datetime import datetime as _dt

    # signals_only=True skips all hard stops — SIGNALS tab enrichment path.
    # The pattern already surfaced; we just want 6-signal scoring, not re-gating.
    if not signals_only:
        if earnings_days is not None and earnings_days <= 5:
            return None, "Earnings within 5 days"

        if iv_rank is not None and iv_rank > 70:
            return None, "IV too high (%s%%)" % iv_rank

    # Block zero volume — market closed or no liquidity
    if df_primary is not None and len(df_primary) > 0:
        try:
            _cur_vol = float(df_primary["volume"].iloc[-3:].mean()) if "volume" in df_primary.columns else 1
            if _cur_vol == 0:
                if not signals_only:
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
            if not signals_only:
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
                    if not signals_only:
                        return None, "Gapped up %.1f%% against PUT signal - invalidated" % gap_pct
                if direction == "bullish" and gap_pct < -3.0:
                    if not signals_only:
                        return None, "Gapped down %.1f%% against CALL signal - invalidated" % gap_pct
    except Exception:
        pass

    # Sector ETF must be above 20 EMA for bullish, below for bearish
    try:
        sector_etf = SECTOR_ETF.get(ticker, "SPY")
        if sector_etf and sector_etf != ticker:
            _bull_sector, _bear_sector, _sector_detail = check_sector_etf_trend(sector_etf)
            if direction == "bullish" and not _bull_sector:
                if not signals_only:
                    return None, "Sector %s in downtrend — CALL signal blocked. %s" % (sector_etf, _sector_detail)
            if direction == "bearish" and not _bear_sector:
                if not signals_only:
                    return None, "Sector %s in uptrend — PUT signal blocked. %s" % (sector_etf, _sector_detail)
    except Exception:
        pass

    # Stock must outperform its sector ETF over 10 days for bullish
    try:
        sector_etf = SECTOR_ETF.get(ticker, "SPY")
        if sector_etf and sector_etf != ticker:
            _bull_rs, _bear_rs, _rs_detail = check_relative_strength(ticker, sector_etf, days=10)
            if direction == "bullish" and not _bull_rs:
                if not signals_only:
                    return None, "Relative strength weak — %s underperforming %s. %s" % (ticker, sector_etf, _rs_detail)
            if direction == "bearish" and not _bear_rs:
                if not signals_only:
                    return None, "Relative strength strong — %s outperforming %s (bad for PUT). %s" % (ticker, sector_etf, _rs_detail)
    except Exception:
        pass

    # Positive 5-day AND 10-day returns required for bullish
    # Negative 5-day AND 10-day returns required for bearish
    try:
        _bull_mom, _bear_mom, _mom_detail = check_momentum_filter(ticker)
        if direction == "bullish" and not _bull_mom:
            if not signals_only:
                return None, "Momentum negative — %s needs positive 5D & 10D returns for CALL. %s" % (ticker, _mom_detail)
        if direction == "bearish" and not _bear_mom:
            if not signals_only:
                return None, "Momentum positive — %s needs negative 5D & 10D returns for PUT. %s" % (ticker, _mom_detail)
    except Exception:
        pass

    # Energy/materials names must have commodity trend aligned
    try:
        _comm_ok, _comm_detail = check_commodity_trend(ticker, direction)
        if not _comm_ok:
            if not signals_only:
                return None, "Commodity trend opposing signal — %s" % _comm_detail
    except Exception:
        pass

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
                        if not signals_only:
                            return None, "Sector %s up %.1f%% today - fighting PUT signal" % (sector_etf, sec_move)
                    if direction == "bullish" and sec_move < -2.0:
                        if not signals_only:
                            return None, "Sector %s down %.1f%% today - fighting CALL signal" % (sector_etf, sec_move)
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
                    if not signals_only:
                        return None, "Price dropped %.2f (%.1fx ATR) from recent high - CALL invalidated" % (drop, drop/atr)
            else:
                recent_low = float(df_primary["low"].iloc[-10:].min())
                rip = entry_price - recent_low
                if rip > atr * 1.5:
                    if not signals_only:
                        return None, "Price ripped %.2f (%.1fx ATR) from recent low - PUT invalidated" % (rip, rip/atr)
    except Exception:
        pass

    # Only hard blocks: trading halt and delisted. Everything else is intel.
    _news_block, _news_reason, _news_data, _news_conf_adj, _flip, _flip_reason = (
        run_news_check(ticker, direction)
    )
    if _news_block:
        return None, _news_reason

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

    # Signal 7: 200-Day MA alignment
    _ma200_above  = None
    _ma200_val    = None
    _ma200_rising = None
    _ma200_pct    = None
    try:
        _ma200_above, _ma200_val, _ma200_rising, _ma200_pct = fetch_200ma(ticker)
        if _ma200_above is not None:
            if direction == "bullish" and _ma200_above and _ma200_rising:
                signals_hit += 1
                signal_detail.append("\u2705 Above Rising 200MA ($%.2f, +%.1f%%)" % (_ma200_val, _ma200_pct or 0))
            elif direction == "bearish" and not _ma200_above and not _ma200_rising:
                signals_hit += 1
                signal_detail.append("\u2705 Below Falling 200MA ($%.2f, %.1f%%)" % (_ma200_val, _ma200_pct or 0))
            elif direction == "bullish" and _ma200_above and not _ma200_rising:
                signal_detail.append("\u274c Above Flat 200MA \u2014 trend weakening")
            elif direction == "bullish" and not _ma200_above:
                signal_detail.append("\u274c Below 200MA \u2014 structural headwind")
            elif direction == "bearish" and not _ma200_above and _ma200_rising:
                signal_detail.append("\u274c Below Rising 200MA \u2014 potential support")
            elif direction == "bearish" and _ma200_above:
                signal_detail.append("\u274c Above 200MA \u2014 structural tailwind (headwind for PUT)")
        else:
            signal_detail.append("\u274c 200MA Unavailable")
    except Exception:
        signal_detail.append("\u274c 200MA Check Failed")


    # 4/5 required for GO NOW - but still score 3/5 for WATCHING/ON DECK
    # Bucket assignment in full_scan uses signals_hit to separate tiers
    if signals_hit < 2 and not signals_only:
        return None, "Only %s/5 quality signals aligned (need 2+ minimum)" % signals_hit
    # 2/5 signals: low confidence, will land in ON DECK only

    score = 50
    score += signals_hit * 5
    score += min(exh_score * 5, 18)  # exhaustion quality multiplier — raised weight

    # Fibonacci confluence boost — up to +8 max (reduced to prevent score inflation)
    fib_boost = fib_result.get("boost", 0) if isinstance(fib_result, dict) else 0
    score += min(fib_boost, 8)

    # 200MA confidence boost
    try:
        if _ma200_above is not None:
            if direction == "bullish" and _ma200_above and _ma200_rising:
                score += 8
            elif direction == "bullish" and _ma200_above and not _ma200_rising:
                score += 2
            elif direction == "bullish" and not _ma200_above:
                score -= 6
            elif direction == "bearish" and not _ma200_above and not _ma200_rising:
                score += 8
            elif direction == "bearish" and not _ma200_above and _ma200_rising:
                score += 2
            elif direction == "bearish" and _ma200_above:
                score -= 6
    except Exception:
        pass

    # S/R Level scoring
    _sr_data  = {}
    _sr_label = ""
    try:
        _sr_data  = detect_sr_levels(ticker, float(current_price) if current_price else 0, direction)
        score    += _sr_data.get("conf_boost", 0)
        _sr_label = _sr_data.get("label", "")
    except Exception:
        pass

    # Confluence Intel scoring
    _confluence_data = {}
    try:
        _confluence_data = detect_confluence_setup(
            ticker, float(current_price) if current_price else 0, direction, atr=atr,
        )
        _cfl_align = _confluence_data.get("alignment_count", 0)
        if _cfl_align >= 4:   score += 6
        elif _cfl_align >= 3: score += 4
        elif _cfl_align >= 2: score += 2
    except Exception:
        pass

    # Volatility classification
    _vol_class = {}
    try:
        _vol_class = classify_stock_volatility(ticker, float(current_price) if current_price else None)
    except Exception:
        pass

    # Fibonacci multi-timeframe
    _fib_data = {}
    try:
        _fib_data = detect_multi_timeframe_fib(ticker, float(current_price) if current_price else 0, direction, trade_style or "swing")
        if not _fib_data.get("trade_valid", True):
            score = max(0, score - 10)
    except Exception:
        pass



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
    elif _against_bias:             score -= 3   # soft flag — strong setups still surface
    if sector_bias  == direction:   score += 4
    elif sector_bias == "neutral":  score += 2

    if iv_rank is not None:
        if 20 <= iv_rank <= 50:   score += 5
        elif 15 <= iv_rank <= 65: score += 2

    # News sentiment adjustment — aligning news boosts, opposing news penalizes
    score += _news_conf_adj

    liq_ok, liq_vol, liq_oi, _ = check_liquidity(ticker)
    if liq_ok:
        score += 3 if liq_vol >= 500 else 2 if liq_vol >= 100 else 1

    final = min(97, max(50, score))  # cap at 97 — reserve 100% for exceptional setups only

    # Exhaustion confirmed = minimum 72% floor
    # Exhaustion missing = cap at 84% — strong but not HIGH CONVICTION
    if exh_confirmed:
        final = max(final, 72)
    else:
        final = min(final, 84)

    # Signal count hard caps — score cannot exceed these limits regardless of
    # how strong individual signals score. Prevents 3/6 signals from showing 90%+.
    _signal_caps = {0: 55, 1: 62, 2: 72, 3: 80, 4: 88, 5: 94, 6: 97, 7: 97}
    _sig_cap = _signal_caps.get(min(signals_hit, 7), 97)
    final = min(final, _sig_cap)

    return final, {
        "exhaustion_confirmed": exh_confirmed,
        "exhaustion_score":     exh_score,
        "exhaustion_reasons":   exh_reasons,
        "signals_hit":          signals_hit,
        "signal_detail":        signal_detail,
        "news_data":            _news_data,
        "news_sentiment":       _news_data.get("sentiment", "neutral"),
        "news_score":           _news_data.get("score", 0),
        "news_conf_adj":        _news_conf_adj,
        "flip_signal":          _flip,
        "flip_reason":          _flip_reason,
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
        "ma200_above":          _ma200_above,
        "ma200_val":            _ma200_val,
        "ma200_rising":         _ma200_rising,
        "ma200_pct":            _ma200_pct,
        "sr_data":              _sr_data,
        "sr_label":             _sr_label,
        "confluence":           _confluence_data,
    }

def scan_single_ticker(ticker, toggles, account_size, risk_pct,
                        dte_quick, dte_swing, max_premium,
                        trade_style_filter, market_bias):
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

            # Stock-level override: strong volume surge at a low price
            # can still be a valid CALL even in a bearish market
            _effective_bias = market_bias
            if direction == "bullish" and market_bias == "bearish":
                try:
                    _avg_vol = float(df_pri["volume"].iloc[-20:].mean())
                    _cur_vol = float(df_pri["volume"].iloc[-3:].mean())
                    _price_pct_from_low = (
                        (float(df_pri["close"].iloc[-1]) - float(df_pri["low"].iloc[-20:].min()))
                        / max(float(df_pri["low"].iloc[-20:].min()), 0.01) * 100
                    )
                    # Volume surging + price near 20-day low = bounce candidate
                    if _cur_vol > _avg_vol * 1.5 and _price_pct_from_low < 5:
                        _effective_bias = "neutral"
                except Exception:
                    pass

            conf, detail = precision_score(
                ticker, direction, df_pri, df_con,
                iv_rank, earn_days, _effective_bias,
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
                "news_data":     detail.get("news_data", {}),
                "news_sentiment":detail.get("news_sentiment", "neutral"),
                "flip_signal":   detail.get("flip_signal", False),
                "flip_reason":   detail.get("flip_reason", ""),
                "rel_vol":        rel_vol,
                "vol_spike":      vol_spike,
                "block_detected": block_detected,
                "sq_state":       sq_state,
                "sq_compression": sq_compression,
                "confluence":     detail.get("confluence", {}),
                "vol_class":      detail.get("vol_class", {}),
                "fib_data":       detail.get("fib_data", {}),
                "atr":            float(atr) if atr else 0,
            })
    except Exception as _e:
        results.append({"ticker": ticker, "_rejected": True, "_reason": "Exception: " + str(_e)[:80]})
    return results

def full_scan(scan_list, toggles, account_size, risk_pct,
              dte_quick, dte_swing, max_premium, trade_style_filter,
              progress_cb=None):
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

            try:
                _wbias = get_weekly_macro_bias()
                _wbias_overall = _wbias.get("overall", "NEUTRAL")
                _sig_type = "CALL" if r.get("direction") == "bullish" else "PUT"
                if _wbias_overall == "BULLISH" and _sig_type == "CALL":
                    conf = min(97, conf + 3)   # macro aligned — boost
                    r["macro_bias_label"] = "✅ MACRO ALIGNED"
                elif _wbias_overall == "BEARISH" and _sig_type == "PUT":
                    conf = min(97, conf + 3)
                    r["macro_bias_label"] = "✅ MACRO ALIGNED"
                elif _wbias_overall == "BEARISH" and _sig_type == "CALL":
                    conf = max(50, conf - 5)   # macro headwind — penalize
                    r["macro_bias_label"] = "⚠️ MACRO HEADWIND"
                elif _wbias_overall == "BULLISH" and _sig_type == "PUT":
                    conf = max(50, conf - 5)
                    r["macro_bias_label"] = "⚠️ MACRO HEADWIND"
                else:
                    r["macro_bias_label"] = "➖ MACRO NEUTRAL"
                r["confidence"] = conf
            except Exception:
                r["macro_bias_label"] = "➖ MACRO NEUTRAL"

            low_rr = r.get("low_rr", False)

            _rr_val = r.get("opt", {}).get("rr_option", 0) or 0
            _min_rr = 1.5 if r.get("style") == "quick" else 2.5
            if _rr_val < _min_rr:
                r["_on_deck_reason"] = "Low RR (%.1fx) - need %.1f minimum" % (_rr_val, _min_rr)
                on_deck.append(r)
                continue

            # Tier 1: conf>=85%, gates>=4, signals>=2
            # Tier 2: conf>=75%, gates>=5, signals>=3
            # Both: CONFIRMED + exhaustion
            # Block GO NOW if volume is zero — market closed or no liquidity
            _vol_detail = r.get("detail", {}) or {}
            _has_volume = "No Activity" not in str(_vol_detail.get("signal_detail", []))
            # Signal/Gate alignment check — they must agree within 2 points
            # Gate score out of 7, signal check out of 6 — normalize both to percentage
            _gate_pct   = (gates_passed / 7) * 100
            _signal_pct = (signals_hit / 7) * 100
            _divergence = abs(_gate_pct - _signal_pct)
            _aligned    = _divergence <= 28  # ~2 gates worth of divergence allowed

            # HIGH CONVICTION: conf>=85, gates>=5, signals>=5, aligned
            _high_conf = conf >= 85 and gates_passed >= 5 and signals_hit >= 4 and _aligned
            _med_conf  = conf >= 75 and gates_passed >= 5 and signals_hit >= 4 and _aligned

            # Exhaustion confirmed = lower bar to GO NOW (74% conf, 4 gates)
            # Exhaustion missing   = higher bar required (80% conf, 5 gates)
            if exh_ok:
                _go_now_ok = (_high_conf or _med_conf or conf >= 74) and entry_status == "CONFIRMED"
            else:
                _go_now_ok = (conf >= 80 and gates_passed >= 5 and signals_hit >= 4 and _aligned) and entry_status == "CONFIRMED"

            if _go_now_ok:
                go_now.append(r)
                # Log to signal_outcomes for all users
                try:
                    log_signal_outcome(r)
                except Exception:
                    pass

            elif conf >= 65 and gates_passed >= 4 and entry_status == "CONFIRMED" and signals_hit >= 2:
                watching.append(r)

            elif conf >= 60 and gates_passed >= 3 and signals_hit >= 2:
                watching.append(r)

            elif conf >= 45 or signals_hit >= 2:
                r["_on_deck_reason"] = (
                    "%s/5 signals" % signals_hit if signals_hit < 3
                    else "Building (%s%%)" % conf
                )
                on_deck.append(r)
    try:
        _macro_bear, _macro_bull, _macro_triggers = check_macro_sentiment()
    except Exception:
        _macro_bear, _macro_bull, _macro_triggers = False, False, []

    # Submit all futures first
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="pbp_scan") as executor:
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

    # Store macro triggers in BG_RESULTS for regime banner display
    try:
        with _BG_LOCK:
            _BG_RESULTS["macro_triggers"] = _macro_triggers
            _BG_RESULTS["macro_bear"]     = _macro_bear
            _BG_RESULTS["macro_bull"]     = _macro_bull
    except Exception:
        pass
    return go_now, watching, on_deck, market_bias, _macro_triggers


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

        with st.expander("🔑 Change Password"):
            _cp_new  = st.text_input("New Password", type="password", key="cp_new", placeholder="Min 6 characters")
            _cp_conf = st.text_input("Confirm Password", type="password", key="cp_conf", placeholder="Repeat password")
            if st.button("Update Password", use_container_width=True, key="cp_btn"):
                if len(_cp_new) < 6:
                    st.error("Password must be at least 6 characters")
                elif _cp_new != _cp_conf:
                    st.error("Passwords don't match")
                else:
                    try:
                        from supabase import create_client
                        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
                        _at = st.session_state.get("_access_token", "")
                        _rt = st.session_state.get("_refresh_token", "")
                        if _at and _rt:
                            _sb.auth.set_session(_at, _rt)
                        _sb.auth.update_user({"password": _cp_new})
                        st.success("✅ Password updated!")
                    except Exception as _cpe:
                        st.error("Error: %s" % str(_cpe)[:80])
    st.markdown("---")
    _ticker_options = ["— Select a ticker —"] + list(st.session_state.user_watchlist or DEFAULT_WATCHLIST)
    _ticker_choice  = st.selectbox("TICKER", _ticker_options, index=0)
    selected_ticker = None if _ticker_choice.startswith("—") else _ticker_choice
    custom = st.text_input("Or type ticker symbol", "", placeholder="e.g. NVDA").upper().strip()
    if custom:
        import re as _re
        if _re.match(r'^[A-Z]{1,5}$', custom):
            selected_ticker = custom
        else:
            st.error("Enter a ticker symbol only (e.g. NVDA, AAPL) - not a company name")
    selected_tf = st.selectbox("CHART TIMEFRAME", list(TIMEFRAMES.keys()), index=2)
    st.caption("Signals use automatic timeframes per mode.")
    st.markdown("---")
    toggles = {"db": True, "dt": True, "br": True, "vwap": True, "flag": True, "orb": True, "mom": True, "tri": True, "hs": True}
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
    if DISCORD_WEBHOOK_URL:
        st.success("📲 TELEGRAM CONNECTED")
    else:
        st.info("📲 Add DISCORD_WEBHOOK_URL in Railway to enable alerts")

# Auto-refresh - NEVER fire while a scan is running.
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

if not selected_ticker:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:40px 20px 20px;text-align:center'>
        <svg width="220" height="80" viewBox="0 0 300 100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="goldG" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#F6E27A"/>
              <stop offset="50%" stop-color="#D4AF37"/>
              <stop offset="100%" stop-color="#9A7D1E"/>
            </linearGradient>
          </defs>
          <text x="50%" y="72%" text-anchor="middle"
                font-family="Barlow Condensed, Arial Black, sans-serif"
                font-size="78" font-weight="900" fill="url(#goldG)"
                letter-spacing="2">PBP</text>
        </svg>
        <div style='color:#A1A1A6;font-size:0.85rem;margin-top:8px;letter-spacing:2px'>PAIDBUTPRESSURED</div>
        <div style='color:#4a5568;font-size:0.75rem;margin-top:8px'>Select a ticker from the sidebar to view signals · Or run a scan below</div>
    </div>
    """, unsafe_allow_html=True)
    selected_ticker = "SPY"
    df            = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)
    current_price = 0.0
    prev_close    = 0.0
    pct_change    = 0.0
    iv_rank       = None
    hv            = None
    earnings_days = None
    atr           = None
    htf_trend     = None
    htf_rsi       = None
    htf_ema       = None
    liq_ok        = True
    liq_vol       = 0
    liq_oi        = 0
    liq_msg       = ""
    _blank_state  = True
else:
    _blank_state  = False
    df            = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)

if not _blank_state:
    current_price = fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
    prev_close    = float(df["close"].iloc[-2]) if len(df)>1 else current_price
    pct_change    = ((current_price-prev_close)/prev_close)*100
    iv_rank, hv   = fetch_iv_rank(selected_ticker)
    earnings_days = check_earnings(selected_ticker)

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

if not _blank_state:
    atr = calc_atr(df)
    htf_trend, htf_rsi, htf_ema = fetch_htf_trend(selected_ticker)
    liq_ok, liq_vol, liq_oi, liq_msg = check_liquidity(selected_ticker)
else:
    atr = None

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

SCAN_INTERVAL = 300  # 5 minutes

def should_run_auto_scan():
    if not st.session_state.auto_scan_enabled: return False
    last = st.session_state.auto_scan_last_run
    if last is None: return True
    return (datetime.now() - last).total_seconds() >= SCAN_INTERVAL

# BACKGROUND SCAN ENGINE
# Runs in a daemon thread completely separate from Streamlit's render cycle.

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


def send_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        return
    import urllib.request, json as _j, re
    # Convert HTML to Discord markdown
    text = msg
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text)
    text = re.sub(r'<[^>]+>', '', text)  # strip remaining tags
    payload = _j.dumps({"content": text}).encode("utf-8")
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception:
        pass

# Keep send_telegram_text as alias so existing calls don't break
def send_telegram_text(msg):
    send_discord(msg)


_news_seen_articles = {}  # module-level — persists across bg loop iterations

def _check_watchlist_news_alerts(watchlist):
    global _news_seen_articles
    for ticker in watchlist:
        try:
            articles = fetch_ticker_news(ticker, hours=1, limit=5)
            if not articles:
                continue
            velocity_score, is_breaking, recent_count = calculate_news_velocity(
                articles, window_minutes=15
            )
            if not is_breaking:
                continue
            seen = _news_seen_articles.get(ticker, set())
            new_articles = [a for a in articles if a.get("url", "") not in seen]
            if not new_articles:
                continue
            news = score_news_sentiment(new_articles, ticker)
            if news["sentiment"] == "neutral":
                continue
            direction_emoji = "📈" if news["sentiment"] == "bullish" else "📉"
            action          = "CALL" if news["sentiment"] == "bullish" else "PUT"
            top_headline    = new_articles[0].get("title", "")[:100]
            source          = new_articles[0].get("site", "")
            msg = (
                "📰 <b>NEWS ALERT — %s</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "%s Sentiment: <b>%s</b> · Score: <b>%s</b>\n"
                "💡 Consider: <b>%s</b>\n"
                "📊 %s articles in last 15 min\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<i>%s</i>\n"
                "<i>Source: %s</i>"
            ) % (
                ticker, direction_emoji, news["sentiment"].upper(),
                news["score"], action, recent_count, top_headline, source,
            )
            send_telegram_text(msg)
            _news_seen_articles[ticker] = seen | {
                a.get("url", "") for a in new_articles
            }
        except Exception:
            pass

def _bg_scan_loop():
    import time as _time

    while True:
        # Wait for trigger or 5-minute auto interval
        triggered = _BG_TRIGGER.wait(timeout=300)
        _BG_TRIGGER.clear()
        try:
            _wl_for_news = _BG_RESULTS.get("scan_list", ["SPY", "QQQ", "IWM"])
            _check_watchlist_news_alerts(_wl_for_news)
        except Exception:
            pass

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

            go, watching, deck, mkt, _macro_triggers = full_scan(
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


# SUPABASE PERSISTENCE ENGINE

def get_supabase(service=False):
    """
    Returns a Supabase client.
      service=True  -> uses SUPABASE_SERVICE_KEY (bypasses RLS, server-side use)
      service=False -> uses SUPABASE_KEY + attaches user session token (RLS-aware)
    """
    if not SUPABASE_URL:
        return None
    try:
        from supabase import create_client
        if service:
            # Service key bypasses RLS — use for cross-user writes (signal_outcomes, etc)
            if not SUPABASE_SERVICE_KEY:
                # Fallback to anon key if service key not configured — will fail RLS
                if not SUPABASE_KEY:
                    return None
                client = create_client(SUPABASE_URL, SUPABASE_KEY)
            else:
                client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            return client
        # Regular client — attach user session for RLS
        if not SUPABASE_KEY:
            return None
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        try:
            _at = st.session_state.get("_access_token", "")
            _rt = st.session_state.get("_refresh_token", "")
            if _at and _rt:
                client.auth.set_session(_at, _rt)
        except Exception:
            pass  # fall through with anon client
        return client
    except Exception as _ge:
        print("[get_supabase] error: %s" % str(_ge)[:120])
        return None

def get_user_id():
    import uuid
    params = st.query_params
    uid = params.get("uid", None)
    if not uid:
        uid = str(uuid.uuid4())[:8]  # short 8-char ID, easy to share
        st.query_params["uid"] = uid
    return uid

def load_user_data(user_id):
    """Load all user data from Supabase user_data table."""
    if not user_id:
        return {}
    sb = get_supabase(service=True)
    if not sb:
        return {}
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
    except Exception as _le:
        print("[load_user_data] error: %s" % str(_le)[:200])
    return {}

def save_user_data(user_id, watchlist=None, watch_queue=None, preferences=None):
    """Save user data to Supabase. Uses service key to bypass RLS upsert quirks."""
    if not user_id:
        print("[save_user_data] skipped — no user_id")
        return False
    sb = get_supabase(service=True)
    if not sb:
        print("[save_user_data] skipped — no supabase client")
        return False
    try:
        import json as _j
        row = {"user_id": str(user_id), "updated_at": datetime.now(tz=pytz.UTC).isoformat()}
        if watchlist   is not None: row["watchlist"]   = _j.dumps(watchlist)
        if watch_queue is not None: row["watch_queue"] = _j.dumps(watch_queue)
        if preferences is not None: row["preferences"] = _j.dumps(preferences)
        resp = sb.table("user_data").upsert(row, on_conflict="user_id").execute()
        if not resp.data:
            print("[save_user_data] upsert returned no data for user_id=%s" % user_id)
        return True
    except Exception as e:
        print("[save_user_data] FAILED: %s" % str(e)[:200])
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


def log_signal_outcome(r):
    sb = get_supabase(service=True)
    if not sb:
        return
    try:
        opt         = r.get("opt", {})
        ticker      = r.get("ticker", "")
        signal_type = "CALL" if r.get("direction") == "bullish" else "PUT"
        today_start = datetime.now(tz=pytz.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        existing = sb.table("signal_outcomes")             .select("id")             .eq("ticker", ticker)             .eq("signal_type", signal_type)             .gte("logged_at", today_start)             .execute()
        if existing.data:
            return  # already logged this ticker+direction today
        _regime_state = ""
        try:
            _regime_state = r.get("regime_alignment", "") or ""
        except Exception:
            pass
        _macro_b = ""
        try:
            _mblbl = r.get("macro_bias_label", "") or ""
            if "ALIGNED" in _mblbl:   _macro_b = "ALIGNED"
            elif "HEADWIND" in _mblbl: _macro_b = "HEADWIND"
            elif "NEUTRAL" in _mblbl:  _macro_b = "NEUTRAL"
        except Exception:
            pass
        _cfl_align = 0
        try:
            _cfl = r.get("confluence", {}) or r.get("detail", {}).get("confluence", {}) or {}
            _cfl_align = int(_cfl.get("alignment_count", 0) or 0)
        except Exception:
            pass
        _tw = ""
        try:
            _tw = get_time_of_day_context().get("window", "")
        except Exception:
            pass

        _delta = 0.5
        try:
            _delta = abs(float(opt.get("delta", 0.5) or 0.5))
            _delta = max(0.1, min(0.9, _delta))
        except Exception:
            pass

        _exp_str = ""
        try:
            _exp_str = opt.get("expiration", "") or ""
        except Exception:
            pass

        row = {
            "ticker":       ticker,
            "signal_type":  signal_type,
            "direction":    r.get("direction"),
            "pattern":      r.get("pattern"),
            "style":        r.get("style"),
            "confidence":   r.get("confidence"),
            "gates_passed": r.get("gates_passed"),
            "signals_hit":  r.get("signals_hit", 0),
            "entry_price":  round(float(r.get("price", 0) or 0), 2),
            "target":       round(float(opt.get("target", 0) or 0), 2),
            "stop":         round(float(opt.get("stop", 0) or 0), 2),
            "strike":       round(float(opt.get("strike", 0) or 0), 2),
            "premium":      round(float(opt.get("premium", 0) or 0), 2),
            "delta":        round(_delta, 3),
            "option_type":  opt.get("type", signal_type),
            "outcome_1d":   None,
            "outcome_3d":   None,
            "outcome_5d":   None,
            "peak_pnl_pct": None,
            "exit_premium": None,
            "exit_reason":  None,
            "result":       "OPEN",
            "logged_at":    datetime.now(tz=pytz.UTC).isoformat(),
            "resolved_at":  None,
            "market_regime":        _regime_state,
            "macro_bias":           _macro_b,
            "confluence_alignment": _cfl_align,
            "time_window":          _tw,
        }
        resp = sb.table("signal_outcomes").insert(row).execute()
        if not resp.data:
            print("[signal_outcomes] Insert returned no data for %s %s" % (ticker, signal_type))
    except Exception as _soe:
        print("[signal_outcomes] Insert failed: %s" % str(_soe)[:200])
def update_signal_outcomes():
    """Resolve open signal outcomes using same delta-adjusted P&L formula as paper trades.
    WIN  = option P&L >= +20% at any check point OR stock hit target
    LOSS = option P&L <= -20% at any check point OR stock hit stop
    EXPIRED = past expiration date, close at current P&L
    OPEN = still within thresholds and not expired"""
    sb = get_supabase(service=True)
    if not sb:
        return
    WIN_THRESHOLD  =  20.0
    LOSS_THRESHOLD = -20.0
    try:
        cutoff = (datetime.now(tz=pytz.UTC) - timedelta(hours=16)).isoformat()
        res = (sb.table("signal_outcomes")
               .select("*")
               .eq("result", "OPEN")
               .lt("logged_at", cutoff)
               .limit(30)
               .execute())
        if not res.data:
            return
        today = datetime.now(tz=pytz.UTC).date()
        for row in res.data:
            try:
                ticker       = row["ticker"]
                entry_price  = float(row.get("entry_price") or 0)
                entry_premium= float(row.get("premium") or 0)
                direction    = row.get("direction", "bullish")
                target       = float(row.get("target") or 0)
                stop         = float(row.get("stop") or 0)
                delta        = float(row.get("delta") or 0.5)
                delta        = max(0.1, min(0.9, delta))
                is_bull      = direction == "bullish"
                peak_pnl     = float(row.get("peak_pnl_pct") or 0)

                if entry_price <= 0 or entry_premium <= 0:
                    continue

                exp_str = row.get("expiration") or ""
                is_expired = False
                if exp_str:
                    try:
                        for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%y"):
                            try:
                                exp_date = datetime.strptime(exp_str, fmt).date()
                                if today > exp_date:
                                    is_expired = True
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                df = _fmp_download(ticker, "60d", "1d")
                if df is None or len(df) < 1:
                    continue
                df = df.sort_values("datetime").reset_index(drop=True)

                logged_at = row.get("logged_at", "")
                try:
                    log_date = datetime.fromisoformat(
                        logged_at.replace("Z", "+00:00")
                    ).date()
                except Exception:
                    log_date = (today - timedelta(days=30))

                df["_date"] = df["datetime"].apply(
                    lambda x: x.date() if hasattr(x, "date") else
                    datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
                )
                after_df = df[df["_date"] > log_date].reset_index(drop=True)
                closes = after_df["close"].astype(float).tolist()

                if not closes:
                    continue

                def option_pnl(stock_close):
                    stock_move   = stock_close - entry_price
                    premium_move = stock_move * delta if is_bull else -stock_move * delta
                    cur_premium  = max(0.01, entry_premium + premium_move)
                    return round((cur_premium - entry_premium) / entry_premium * 100, 2), round(cur_premium, 2)

                o1d_pnl, _ = option_pnl(closes[0]) if len(closes) >= 1 else (None, None)
                o3d_pnl, _ = option_pnl(closes[2]) if len(closes) >= 3 else (None, None)
                o5d_pnl, o5d_prem = option_pnl(closes[4]) if len(closes) >= 5 else (None, None)

                latest_close = closes[-1]
                latest_pnl, latest_prem = option_pnl(latest_close)

                all_pnls = [p for p in [o1d_pnl, o3d_pnl, o5d_pnl, latest_pnl] if p is not None]
                if all_pnls:
                    peak_pnl = max(peak_pnl, max(all_pnls))

                result = "OPEN"
                exit_reason = None
                exit_premium = None

                if target > 0 and stop > 0:
                    for c in closes:
                        if is_bull and c >= target:
                            result = "WIN"; exit_reason = "TARGET HIT"
                            _, exit_premium = option_pnl(c)
                            break
                        elif not is_bull and c <= target:
                            result = "WIN"; exit_reason = "TARGET HIT"
                            _, exit_premium = option_pnl(c)
                            break
                        elif is_bull and c <= stop:
                            result = "LOSS"; exit_reason = "STOP HIT"
                            _, exit_premium = option_pnl(c)
                            break
                        elif not is_bull and c >= stop:
                            result = "LOSS"; exit_reason = "STOP HIT"
                            _, exit_premium = option_pnl(c)
                            break

                if result == "OPEN":
                    for pnl in all_pnls:
                        if pnl >= WIN_THRESHOLD:
                            result = "WIN"
                            exit_reason = "+%.0f%% PROFIT TARGET" % WIN_THRESHOLD
                            exit_premium = latest_prem
                            break
                        elif pnl <= LOSS_THRESHOLD:
                            result = "LOSS"
                            exit_reason = "%.0f%% STOP LOSS" % LOSS_THRESHOLD
                            exit_premium = latest_prem
                            break

                if result == "OPEN" and is_expired:
                    result = "WIN" if latest_pnl >= 0 else "LOSS"
                    exit_reason = "EXPIRED"
                    exit_premium = latest_prem

                update_payload = {
                    "outcome_1d":   o1d_pnl,
                    "outcome_3d":   o3d_pnl,
                    "outcome_5d":   o5d_pnl,
                    "peak_pnl_pct": round(peak_pnl, 1),
                    "result":       result,
                }
                if result != "OPEN":
                    update_payload["exit_premium"] = exit_premium
                    update_payload["exit_reason"]  = exit_reason
                    update_payload["resolved_at"]  = datetime.now(tz=pytz.UTC).isoformat()

                sb.table("signal_outcomes").update(update_payload).eq("id", row["id"]).execute()
                print("[outcomes] %s %s -> %s (%s) pnl=%.1f%%" % (
                    ticker, direction, result,
                    exit_reason or "OPEN", latest_pnl or 0
                ))
            except Exception as _row_err:
                print("[outcomes] row error %s: %s" % (row.get("ticker","?"), str(_row_err)[:120]))
                continue
    except Exception as _ue:
        print("[outcomes] update error: %s" % str(_ue)[:200])

def init_user_watchlist():
    user_id = st.session_state.get("user_id")
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
if st.session_state.get("authenticated") and not st.session_state.get("_paper_trades_loaded"):
    st.session_state._paper_trades_loaded = True


# TELEGRAM ALERT ENGINE

def send_make_webhook(r):
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
    """Send signal alert to Discord with rich embed."""
    if not DISCORD_WEBHOOK_URL:
        return
    import urllib.request, json
    is_bull     = r["direction"] == "bullish"
    opt         = r["opt"]
    action      = "CALL" if is_bull else "PUT"
    color       = 0x00C853 if is_bull else 0xC1121F  # green or red
    bucket_emoji = {"GO NOW": "🚨", "WATCHING": "👀", "ON DECK": "📋"}.get(alert_type, "📡")
    detail      = r.get("detail", {}) or {}
    sig_detail  = r.get("signal_detail", []) or detail.get("signal_detail", [])
    signals_hit = r.get("signals_hit", 0) or detail.get("signals_hit", 0)
    sig_lines   = "\n".join(sig_detail) if sig_detail else "%s/6 signals confirmed" % signals_hit

    embed = {
        "title": "%s %s — %s %s" % (bucket_emoji, alert_type, r["ticker"], action),
        "color": color,
        "fields": [
            {"name": "Pattern", "value": r.get("pattern", "Signal"), "inline": True},
            {"name": "Style",   "value": r.get("style", "swing").upper(), "inline": True},
            {"name": "Confidence", "value": "%s%%" % r["confidence"], "inline": True},
            {"name": "Gates",   "value": "%s/7" % r["gates_passed"], "inline": True},
            {"name": "Entry",   "value": "$%.2f" % r["price"], "inline": True},
            {"name": "Stop",    "value": "$%.2f" % opt["stop"], "inline": True},
            {"name": "Target",  "value": "$%.2f" % opt["target"], "inline": True},
            {"name": "Premium", "value": "$%.2f/sh" % opt["premium"], "inline": True},
            {"name": "R:R",     "value": "%.1fx" % opt.get("rr_option", 0), "inline": True},
            {"name": "Signals", "value": sig_lines or "—", "inline": False},
        ],
        "footer": {"text": "PaidButPressured · Not financial advice · Paper trade first"}
    }
    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass






# Profit-taking thresholds by trade style
PAPER_PROFIT_TARGET = {
    "quick": 30,   # exit quick trades at +30% premium gain
    "swing": 50,   # exit swing trades at +50% premium gain
}
PAPER_STOP_LOSS_PCT = -20  # exit any trade at -20% premium loss

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
    st.iframe("""<script>
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
    # Auto-enter paper trade for new GO NOW — 88%+ confidence, 5/7 gates minimum

# auto_scan_poll removed - background thread runs independently,
# no page refresh needed to trigger it.

if not _blank_state:
    if earnings_days is not None:
        if earnings_days <= 1:   st.error(f"EARNINGS {'TODAY' if earnings_days==0 else 'TOMORROW'} on {selected_ticker} - Avoid new options positions.")
        elif earnings_days <= 7: st.error(f"EARNINGS IN {earnings_days} DAYS on {selected_ticker} - 7-point gate will block.")
        else:                    st.warning(f"Earnings in {earnings_days} days on {selected_ticker} - premiums may be inflated.")

    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1:
        color   = "#D4AF37" if pct_change>=0 else "#C1121F"
        arrow   = "UP" if pct_change>=0 else "DN"
        prepost = "" if mstatus=="open" else (" <span style='color:#F6E27A;font-size:0.72rem'>PRE-MARKET</span>" if mstatus=="pre" else " <span style='color:#F6E27A;font-size:0.72rem'>AFTER-HOURS</span>" if mstatus=="after" else "")
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

_mt = st.session_state.get("macro_triggers", [])
if _mt:
    _mt_chips = ""
    for t in _mt:
        _tc = "#C1121F" if t.startswith("⚠️") else "#D4AF37"
        _mt_chips += (
            "<span style='background:%s22;color:%s;border:1px solid %s44;"
            "padding:2px 8px;border-radius:4px;font-size:0.65rem;margin:2px'>%s</span>"
            % (_tc, _tc, _tc, t)
        )
    st.markdown(
        "<div style='background:#1A1A1D;border:1px solid #2A2A2D;border-radius:8px;"
        "padding:8px 14px;margin-bottom:8px'>"
        "<span style='color:#A1A1A6;font-family:monospace;font-size:0.65rem;"
        "letter-spacing:1px'>📰 MACRO TRIGGERS &nbsp;</span>" + _mt_chips +
        "</div>",
        unsafe_allow_html=True
    )

tab_orb,tab_mom,tab4,tab2,tab7,tab_stats = st.tabs(["🔴 ORB","⚡ MOMENTUM","SCAN","CHART","HOW IT WORKS","📊 STATS"])

try:
    _macro_w = get_macro_event_warning()
    st.session_state['macro_warning'] = _macro_w
    _macro_html = render_macro_event_warning_html(_macro_w)
    if _macro_html:
        st.markdown(_macro_html, unsafe_allow_html=True)
except Exception:
    pass
st.markdown(render_time_of_day_banner_html(), unsafe_allow_html=True)
try:
    _stress = get_market_stress_monitor()
    if _stress.get('available'):
        st.markdown(render_market_stress_html(_stress), unsafe_allow_html=True)
except Exception:
    pass
render_weekly_bias_banner()

with tab2:
    chart_db    = [s for s in detect_double_bottom(df, selected_ticker, rr_min=2.0) if s.confirmed]
    chart_dt    = [s for s in detect_double_top(df, selected_ticker, rr_min=2.0)    if s.confirmed]
    chart_br    = [s for s in detect_break_and_retest(df, selected_ticker, rr_min=2.0) if s.confirmed]
    # Sort all setups by confidence, show only the best one on chart
    chart_setups_all = chart_db + chart_dt + chart_br
    chart_setups = sorted(chart_setups_all,
        key=lambda s: getattr(s, "confidence", 0), reverse=True)[:1]

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

        # If entry flipped AGAINST or EXTENDED since last scan, drop to WATCHING
        _stale_drop = []
        _valid_go   = []
        for _r in go_now:
            try:
                _fresh = check_entry_confirmation(_r["ticker"], _r["direction"], _r.get("entry", 0))
                _fresh_status = _fresh.get("status", "WAITING") if isinstance(_fresh, dict) else str(_fresh)
                if _fresh_status == "AGAINST":
                    _r["entry_status"] = "AGAINST"
                    watching.insert(0, _r)
                    _stale_drop.append(_r["ticker"])
                else:
                    _valid_go.append(_r)
            except Exception:
                _valid_go.append(_r)  # keep if check fails
        go_now = _valid_go
        if _stale_drop:
            st.session_state.auto_scan_go_now   = go_now
            st.session_state.auto_scan_watching = watching

        if st.button("🔄 Scan Now", use_container_width=True):
            with st.spinner("Scanning..."):
                go_now, watching, on_deck, mkt_bias, _macro_triggers = full_scan(
                    scan_list, toggles, account_size, risk_pct,
                    dte_quick, dte_swing, max_premium, scan_style_key
                )
                st.session_state.auto_scan_go_now   = go_now
                st.session_state.auto_scan_watching = watching
                st.session_state.auto_scan_on_deck  = on_deck
                st.session_state.auto_scan_mkt      = mkt_bias
                st.session_state.auto_scan_last_run = datetime.now()
                st.session_state.macro_triggers     = _macro_triggers
                st.rerun()
    else:
        st.caption(f"Scanning {len(scan_list)} tickers through full precision stack")
        # Background thread doesn't work on Railway multi-worker deployments
        # (each worker has its own memory space, results never reach the page)

        go_now   = st.session_state.get("scan_go_now",   [])
        watching = st.session_state.get("scan_watching", [])
        on_deck  = st.session_state.get("scan_on_deck",  [])
        mkt_bias = st.session_state.get("scan_mkt",      "neutral")

        _valid_go2 = []
        for _r in go_now:
            try:
                _fresh2 = check_entry_confirmation(_r["ticker"], _r["direction"], _r.get("entry", 0))
                _fresh2_status = _fresh2.get("status", "WAITING") if isinstance(_fresh2, dict) else str(_fresh2)
                if _fresh2_status == "AGAINST":
                    _r["entry_status"] = "AGAINST"
                    watching.insert(0, _r)
                else:
                    _valid_go2.append(_r)
            except Exception:
                _valid_go2.append(_r)
        go_now = _valid_go2

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

            go_now, watching, on_deck, mkt_bias, _macro_triggers = full_scan(
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
            st.session_state.macro_triggers = _macro_triggers


            # Fire Telegram + paper trades for GO NOW signals
            # Telegram is now manual — admin hits "Send to Telegram" button on each card
            # Auto-firing removed to give full control over what gets alerted
            for r in go_now:
                try: save_signal_history(r)
                except: pass

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

            # Update outcome records for OPEN signals older than 1 day
            try:
                update_signal_outcomes()
            except Exception:
                pass

            # Background thread alerts fail on Railway multi-worker (different worker)
            # This inline call reliably reaches Telegram on the same worker.
            try:
                _news_wl = scan_list[:20]
                _check_watchlist_news_alerts(_news_wl)
            except Exception:
                pass

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

    rejected = [r for r in on_deck if r.get("_rejected")]
    real_on_deck = [r for r in on_deck if not r.get("_rejected")]
    on_deck = real_on_deck

    last_run = st.session_state.get("scan_last_run") or st.session_state.get("auto_scan_last_run")

    # Signal breakdown removed — internal data stays internal

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

                # Clear open positions — admin only
                _is_admin_pt = (
                    st.session_state.get("user_email", "").strip().lower() == ADMIN_EMAIL.strip().lower()
                )
                if _is_admin_pt:
                    if st.button("🗑 Clear Open Positions", key="clear_open_trades", use_container_width=True):
                        closed_only = [t for t in st.session_state.paper_trades if t.get("status") != "OPEN"]
                        st.session_state.paper_trades = closed_only
                        st.success("Open positions cleared. Closed trade history preserved.")
                        st.rerun()

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
                try:
                    st.markdown(render_summary_line_html(r), unsafe_allow_html=True)
                except Exception:
                    pass



                # News sentiment block
                try:
                    _mc_news = r.get("news_data", {})
                    if not _mc_news or not _mc_news.get("article_count"):
                        _, _, _mc_news, _mc_adj, _mc_flip, _mc_flipr = run_news_check(
                            r["ticker"], r.get("direction", "bullish")
                        )
                    else:
                        _mc_adj  = r.get("detail", {}).get("news_conf_adj", 0)
                        _mc_flip = r.get("flip_signal", False)
                        _mc_flipr = r.get("flip_reason", "")
                    st.markdown(render_news_sentiment_html(
                        _mc_news, r["ticker"],
                        signal_direction=r.get("direction"),
                        flip_signal=_mc_flip,
                        flip_reason=_mc_flipr,
                        conf_adj=_mc_adj,
                    ), unsafe_allow_html=True)
                except Exception:
                    st.markdown(render_news_sentiment_html(
                        {}, r["ticker"], signal_direction=r.get("direction")
                    ), unsafe_allow_html=True)

                try:
                    _cfl_card = r.get("confluence", r.get("detail", {}).get("confluence", {}))
                    if _cfl_card and _cfl_card.get("available"):
                        st.markdown(render_confluence_block_html(_cfl_card, r.get("direction","bullish")), unsafe_allow_html=True)
                except Exception:
                    pass

                try:
                    _price_c = r.get("price", 0) or 0
                    _dir_c   = r.get("direction", "bullish")
                    _sty_c   = r.get("style", "swing")
                    _atr_c   = (r.get("atr") or (_price_c * 0.015))
                    _vol_c   = r.get("vol_class", r.get("detail", {}).get("vol_class", {}))
                    _fib_c   = r.get("fib_data",  r.get("detail", {}).get("fib_data",  {}))
                    if not _vol_c.get("available") and r.get("ticker"):
                        try: _vol_c = classify_stock_volatility(r["ticker"], _price_c)
                        except Exception: pass
                    if not _fib_c.get("available") and r.get("ticker"):
                        try: _fib_c = detect_multi_timeframe_fib(r["ticker"], _price_c, _dir_c, _sty_c)
                        except Exception: pass
                    _pred_c  = calc_predicted_move(r.get("ticker",""), _price_c, _dir_c, _atr_c,
                        r.get("sq_state","none"), r.get("sq_compression",0) or 0,
                        r.get("block_detected",False), _sty_c)
                    _strike_c = calc_strike_guidance(_vol_c, _pred_c, _price_c, _dir_c)
                    _block_c  = render_volatility_block_html(_vol_c, _pred_c, _strike_c, _fib_c,
                        _pred_c.get("scan_time",""), style=_sty_c)
                    if _block_c:
                        st.markdown(_block_c, unsafe_allow_html=True)
                except Exception:
                    pass
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
                try:
                    _d    = r.get("detail", {}) or {}
                    _mav  = _d.get("ma200_val")
                    _maa  = _d.get("ma200_above")
                    _mar  = _d.get("ma200_rising")
                    _map  = _d.get("ma200_pct")
                    if _maa is not None and _mav:
                        _is_bull_mc = r.get("direction") == "bullish"
                        _mc = (
                            "#D4AF37" if ((_maa and _mar and _is_bull_mc) or (not _maa and not _mar and not _is_bull_mc))
                            else "#F6E27A" if ((_maa and _is_bull_mc) or (not _maa and not _is_bull_mc))
                            else "#C1121F"
                        )
                        _ms  = "Above Rising" if (_maa and _mar) else "Above Flat" if (_maa and not _mar) else "Below Falling" if (not _maa and not _mar) else "Below Rising"
                        _mp  = " (%+.1f%%)" % _map if _map is not None else ""
                        st.markdown(
                            "<div style='background:#1A1A1D;border:1px solid %s44;border-radius:6px;"
                            "padding:8px 12px;margin:4px 0'>"
                            "<div style='font-size:0.62rem;color:#A1A1A6;letter-spacing:1px;margin-bottom:2px'>200-DAY MA</div>"
                            "<div style='font-size:0.78rem;font-weight:700;color:%s'>%s%s</div>"
                            "<div style='font-size:0.7rem;color:#A1A1A6'>$%.2f</div>"
                            "</div>" % (_mc, _mc, _ms, _mp, _mav),
                            unsafe_allow_html=True
                        )
                except Exception:
                    pass

                try:
                    _d  = r.get("detail", {}) or {}
                    _sr = _d.get("sr_data", {})
                    if _sr and _sr.get("label") not in ("S/R Unavailable", "S/R Error", "No Key S/R Nearby", ""):
                        _sc = (
                            "#D4AF37" if _sr.get("conf_boost", 0) >= 8 else
                            "#22C55E" if _sr.get("conf_boost", 0) > 0 else
                            "#C1121F" if _sr.get("conf_boost", 0) < 0 else
                            "#A1A1A6"
                        )
                        _sup_s = "$%.2f" % _sr["nearest_support"]  if _sr.get("nearest_support")  else "—"
                        _res_s = "$%.2f" % _sr["nearest_resistance"] if _sr.get("nearest_resistance") else "—"
                        st.markdown(
                            "<div style='background:#1A1A1D;border:1px solid %s44;border-radius:6px;"
                            "padding:8px 12px;margin:4px 0'>"
                            "<div style='font-size:0.62rem;color:#A1A1A6;letter-spacing:1px;margin-bottom:2px'>S/R LEVELS</div>"
                            "<div style='font-size:0.78rem;font-weight:700;color:%s'>%s</div>"
                            "<div style='font-size:0.7rem;color:#A1A1A6;margin-top:2px'>%s</div>"
                            "<div style='font-size:0.7rem;margin-top:4px'>"
                            "<span style='color:#22C55E'>Sup %s</span>"
                            " &nbsp;·&nbsp; "
                            "<span style='color:#C1121F'>Res %s</span>"
                            "</div>"
                            "</div>" % (_sc, _sc, _sr.get("label",""), _sr.get("detail",""), _sup_s, _res_s),
                            unsafe_allow_html=True
                        )
                except Exception:
                    pass

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

                # Admin only — Send to Discord button
                _is_admin = (
                    st.session_state.get("is_admin", False) or
                    st.session_state.get("user_email", "").strip().lower() == ADMIN_EMAIL.strip().lower()
                )
                if _is_admin and bucket in ("go_now", "watching"):
                    _tg_key = "tg_sent_%s_%s_%s" % (r["ticker"], r.get("style",""), idx)
                    if st.session_state.get(_tg_key):
                        st.markdown(
                            "<div style='background:#1A1500;border:1px solid #D4AF37;border-radius:6px;"
                            "padding:6px;text-align:center;font-size:0.72rem;color:#D4AF37'>✅ Sent to Discord</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("📣 Send to Discord", key="tg_%s_%s_%s" % (bucket, r["ticker"], idx), use_container_width=True):
                            try:
                                send_telegram_alert(r, alert_type=bucket.replace("_"," ").upper())
                                st.session_state[_tg_key] = True
                                st.success("✅ Sent to Discord!")
                            except Exception as _te:
                                st.error("Discord error: %s" % str(_te)[:60])
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

with tab7:
    st.markdown("""
<style>
.hiw-card { background:#0B0B0C; border-radius:12px; padding:18px 22px;
            margin-bottom:14px; border-left:3px solid #2A2A2D; }
.hiw-t    { font-size:1.05rem; font-weight:700; color:#F5F5F5; margin-bottom:8px; }
.hiw-b    { font-size:0.8rem; color:#A1A1A6; line-height:1.75; }
.hiw-b b  { color:#F5F5F5; }
.hiw-chip { display:inline-block; padding:2px 9px; border-radius:4px;
            font-size:0.65rem; font-weight:700; letter-spacing:0.5px; margin:1px 2px; }
</style>
""", unsafe_allow_html=True)

    _hiw = [
        ("#D4AF37", "🎯 What this tool does",
         "This screener trades ONE strategy: the <b>Opening Range Breakout (ORB)</b>. "
         "Every morning the first 15 minutes (9:30&ndash;9:45 ET) sets a high and a low. "
         "That box is the <b>opening range</b>. When price breaks out of it with real "
         "strength and then comes back to test that level and holds &mdash; that's the trade. "
         "The tool watches all of it for you across your whole list, so you're not marking "
         "levels by hand on 200 charts."),

        ("#22C55E", "🚦 The three buckets",
         "Every setup lands in one of three buckets so you know what to do at a glance:<br><br>"
         "<span class='hiw-chip' style='background:#22C55E22;color:#22C55E;border:1px solid #22C55E'>GO NOW</span> "
         "The retest held. The entry is live right now.<br>"
         "<span class='hiw-chip' style='background:#D4AF3722;color:#D4AF37;border:1px solid #D4AF37'>WATCHING</span> "
         "It broke but hasn't retested yet, or the volume was weak. Wait for confirmation.<br>"
         "<span class='hiw-chip' style='background:#7AA2F722;color:#7AA2F7;border:1px solid #7AA2F7'>ON DECK</span> "
         "The range is set and price is near a boundary, but nothing's triggered. Keep an eye on it."),

        ("#22C55E", "🔑 The retest is the entry — not the break",
         "This is the #1 thing beginners get wrong. When price first breaks out, that's the "
         "<b>worst</b> time to enter &mdash; you're chasing and your stop is far away. The real "
         "entry is the <b>retest</b>: price breaks out, pulls back to 'test' the level it broke, "
         "and if it holds, THAT's your shot with a tight stop. The tool grades each retest "
         "<b>A+ to B</b> based on where it holds (right at the level, or at VWAP / the 9EMA). "
         "A break with no retest yet is never GO NOW &mdash; on purpose."),

        ("#D4AF37", "📊 How setups are scored (70 / 30)",
         "Each setup gets a score out of 100. <b>70 points</b> come from the structure &mdash; "
         "did it break on a clean 5m close, was there volume, did the retest hold, is the "
         "range a healthy size. <b>30 points</b> come from context &mdash; news catalyst, "
         "squeeze, market direction, volatility. The key rule: <b>context can only lower a "
         "score or add a little, never create a signal on its own.</b> Good news can't turn a "
         "weak break into a trade. The structure has to be there first."),

        ("#22C55E", "📈 Volume is the filter",
         "A breakout without volume is a trap. The tool checks the volume on the break against "
         "the stock's own normal pace. Weak volume? It tells you to <b>wait for the next 5m "
         "candle to confirm</b> instead of jumping in. That patience is what separates traders "
         "who last from traders who get chopped up."),

        ("#7AA2F7", "🎯 Entry, stop, target, strike &amp; expiration",
         "Every GO NOW card lays out the whole trade: <b>entry</b> at the level, <b>stop</b> at "
         "the retest invalidation, and <b>target</b> at the nearest realistic level (capped at "
         "~1x the day's average range, so it never targets something price can't reach "
         "intraday). It also suggests a <b>strike and expiration</b> sized to that move, plus a "
         "runner target if it trends. Note: the contract-dollar figures are estimates to help "
         "you size &mdash; not live option prices."),

        ("#C1121F", "⚡ The Momentum tab — high risk, read this",
         "The <b>⚡ MOMENTUM</b> tab catches hard moves in the first 15 minutes, <b>before</b> "
         "the range is even done. This is the most violent, least predictable part of the day. "
         "Every momentum card is stamped <b>EARLY &middot; UNCONFIRMED &middot; CAN REVERSE</b> "
         "for a reason. Use it as a heads-up, not a green light. The card shows a two-leg plan: "
         "a small <b>starter</b> now (as a % of your normal size), then <b>add a second leg</b> "
         "only if the range confirms the same direction after 9:45. Size small here. Always."),

        ("#7AA2F7", "🔍 Scanning and single-ticker lookup",
         "Scan your <b>watchlist</b>, a <b>sector</b>, or the <b>full universe</b> &mdash; the tool "
         "only ever looks for the ORB setup. Want just one name? Type it into the search box at "
         "the top of the ORB tab and you get the full read even <b>before it breaks</b>: where "
         "price sits, how far from the trigger, which side of VWAP, and what to watch for."),

        ("#6B6B6B", "📋 The simple version",
         "1. Scan after 9:45 when the range is set.<br>"
         "2. Look at GO NOW first &mdash; those retests are live.<br>"
         "3. Check the volume and the score before you trust it.<br>"
         "4. Use the entry / stop / target on the card. Respect the stop.<br>"
         "5. Momentum tab = early heads-up only, size small, confirm with the range.<br>"
         "6. An empty board is a real answer. Some mornings there's no trade &mdash; that's fine."),
    ]

    for _color, _title, _body in _hiw:
        st.markdown(
            "<div class='hiw-card' style='border-left-color:" + _color + "'>"
            "<div class='hiw-t'>" + _title + "</div>"
            "<div class='hiw-b'>" + _body + "</div></div>",
            unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#6B6B6B;font-size:0.72rem;text-align:center;padding:8px 0 4px 0'>"
        "This tool finds setups and manages risk &mdash; it doesn't guarantee outcomes. "
        "Trade your own plan. Not financial advice.</div>", unsafe_allow_html=True)


with tab_orb:
    st.markdown(
        "<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-radius:6px;"
        "padding:13px 16px;margin-bottom:14px'>"
        "<div style='color:#D4AF37;font-size:0.72rem;font-weight:700;letter-spacing:1.6px;"
        "margin-bottom:5px'>OPENING RANGE BREAKOUT</div>"
        "<div style='color:#A1A1A6;font-size:0.76rem;line-height:1.5'>"
        "Range is the 9:30&ndash;9:45 ET window, wick to wick. Breaks and retests read on the 5m close. "
        "A break alone is never GO NOW &mdash; the retest is the entry."
        "</div></div>", unsafe_allow_html=True)

    # ================= SINGLE-TICKER LOOKUP =================
    _olk1, _olk2 = st.columns([2, 1])
    with _olk1:
        _orb_lookup_tk = st.text_input(
            "Look up one ticker", value="", key="orb_lookup_input",
            placeholder="e.g. NVDA — full read even before it breaks").strip().upper()
    with _olk2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _orb_lookup_go = st.button("READ TICKER", use_container_width=True, key="orb_lookup_btn")

    if _orb_lookup_go and _orb_lookup_tk:
        with st.spinner("Reading " + _orb_lookup_tk + "..."):
            try:
                _lk = orb_lookup_ticker(_orb_lookup_tk, context_fn=orb_build_context)
                st.session_state["orb_lookup_result"] = _lk
            except Exception as _le:
                st.error("Lookup failed: " + str(_le)[:160])
                st.session_state["orb_lookup_result"] = None

    _lk = st.session_state.get("orb_lookup_result")
    if _lk:
        _mode = _lk.get("mode")
        if _mode == "NO_DATA":
            st.markdown("<div style='color:#C1121F;font-size:0.78rem;padding:6px 0'>"
                        + _lk.get("note", "No data") + "</div>", unsafe_allow_html=True)
        elif _mode == "RANGE_BUILDING":
            st.markdown(render_orb_building_html(_lk["levels"]), unsafe_allow_html=True)
        elif _mode == "WATCHING_PRE":
            st.markdown(render_orb_pre_html(_lk), unsafe_allow_html=True)
        elif _mode == "BROKE":
            for _c in _lk.get("candidates", []):
                try:
                    st.markdown(render_orb_card_html(
                        _c["levels"], _c["events"], _c["structure"],
                        _c["score_data"], _c["geo"], _c["side"]), unsafe_allow_html=True)
                except Exception as _ce:
                    st.markdown("<div style='color:#C1121F;font-size:0.72rem'>Card error: "
                                + str(_ce)[:90] + "</div>", unsafe_allow_html=True)
        if st.button("Clear lookup", key="orb_lookup_clear"):
            st.session_state["orb_lookup_result"] = None
            st.rerun()
        st.markdown("<div style='border-top:1px solid #1A1A1A;margin:14px 0'></div>",
                    unsafe_allow_html=True)

    # ================= UNIVERSE / SECTOR / WATCHLIST SCAN =================
    _osc1, _osc2 = st.columns([1, 1.4])
    with _osc1:
        _orb_scope = st.radio(
            "Scan scope", ["My Watchlist", "Sector", "Full Universe"],
            index=2, horizontal=True, key="orb_scope")
    with _osc2:
        _orb_sector = None
        if _orb_scope == "Sector":
            _sector_names = [k for k in SECTOR_LISTS.keys() if k != "My Watchlist"]
            _orb_sector = st.selectbox("Which sector", _sector_names, key="orb_sector_pick")

    if _orb_scope == "My Watchlist":
        _orb_list = list(st.session_state.get("user_watchlist", []) or [])
        if not _orb_list:
            _orb_list = list(DEFAULT_WATCHLIST)
        _scope_label = "My Watchlist (%d)" % len(_orb_list)
    elif _orb_scope == "Sector" and _orb_sector:
        _orb_list = list(SECTOR_LISTS.get(_orb_sector, []))
        _scope_label = "%s (%d)" % (_orb_sector, len(_orb_list))
    else:
        _orb_list = list(SCAN_UNIVERSE)
        _scope_label = "Full Universe (%d)" % len(_orb_list)
    _orb_list = list(dict.fromkeys([t for t in _orb_list if t]))

    _oc1, _oc2, _oc3 = st.columns([1.1, 1, 1])
    with _oc1:
        _orb_run = st.button("RUN ORB SCAN", type="primary", use_container_width=True, key="orb_run_btn")
    with _oc2:
        _orb_bucket = st.selectbox("Bucket", ["All", "GO NOW", "WATCHING", "ON DECK"], key="orb_bucket_f")
    with _oc3:
        _orb_dir = st.selectbox("Direction", ["Both", "Calls only", "Puts only"], key="orb_dir_f")

    _orb_hide_broken = st.checkbox(
        "Hide setups where structure has broken", value=True, key="orb_hide_broken",
        help="Structure broken = price closed through VWAP or lost the 9EMA. No entry, timeline only.")

    st.markdown(
        "<div style='color:#6B6B6B;font-size:0.71rem;margin:2px 0 8px 0'>Scope: <b style='color:#A1A1A6'>"
        + _scope_label + "</b></div>", unsafe_allow_html=True)

    if _orb_run:
        if not _orb_list:
            st.warning("No tickers in scope. Add names to your watchlist or pick a different scope.")
        else:
            _obar = st.progress(0.0)
            _otxt = st.empty()

            def _orb_prog(done, total, tk):
                try:
                    _obar.progress(min(1.0, done / max(1, total)))
                    _otxt.markdown(
                        "<div style='color:#6B6B6B;font-size:0.72rem'>Scanning %d/%d &middot; %s</div>"
                        % (done, total, tk), unsafe_allow_html=True)
                except Exception:
                    pass

            try:
                _ocands, _obuilding, _ostats = run_orb_scan(
                    _orb_list, max_workers=4, progress_cb=_orb_prog)
                st.session_state["orb_results"] = _ocands
                st.session_state["orb_building"] = _obuilding
                st.session_state["orb_stats"] = _ostats
                st.session_state["orb_scan_time"] = datetime.now().strftime("%-I:%M:%S %p ET")
                st.session_state["orb_scan_scope"] = _scope_label
            except Exception as _oe:
                st.error("ORB scan failed: " + str(_oe)[:180])
            finally:
                _obar.empty()
                _otxt.empty()

    _ocands   = st.session_state.get("orb_results", [])
    _obuilding = st.session_state.get("orb_building", [])
    _ostats   = st.session_state.get("orb_stats", {})

    # Bucket count banner — the "4 go now, 6 watching, 10 on deck" board
    if _ocands:
        _n_go = len([c for c in _ocands if c.get("bucket") == "GO NOW"])
        _n_wa = len([c for c in _ocands if c.get("bucket") == "WATCHING"])
        _n_od = len([c for c in _ocands if c.get("bucket") == "ON DECK"])
        st.markdown(
            "<div style='display:flex;gap:10px;margin:10px 0 6px 0'>"
            "<div style='flex:1;background:#22C55E14;border:1px solid #22C55E55;border-radius:6px;"
            "padding:10px;text-align:center'>"
            "<div style='color:#22C55E;font-size:1.5rem;font-weight:800'>%d</div>"
            "<div style='color:#22C55E;font-size:0.6rem;font-weight:700;letter-spacing:1.2px'>GO NOW</div></div>"
            "<div style='flex:1;background:#D4AF3714;border:1px solid #D4AF3755;border-radius:6px;"
            "padding:10px;text-align:center'>"
            "<div style='color:#D4AF37;font-size:1.5rem;font-weight:800'>%d</div>"
            "<div style='color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:1.2px'>WATCHING</div></div>"
            "<div style='flex:1;background:#7AA2F714;border:1px solid #7AA2F755;border-radius:6px;"
            "padding:10px;text-align:center'>"
            "<div style='color:#7AA2F7;font-size:1.5rem;font-weight:800'>%d</div>"
            "<div style='color:#7AA2F7;font-size:0.6rem;font-weight:700;letter-spacing:1.2px'>ON DECK</div></div>"
            "</div>" % (_n_go, _n_wa, _n_od), unsafe_allow_html=True)

    if _ostats:
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:0.72rem;color:#6B6B6B;"
            "padding:4px 0 12px 0'>"
            "<span>Scanned <b style='color:#F5F5F5'>%d</b></span>"
            "<span>With breaks <b style='color:#F5F5F5'>%d</b></span>"
            "<span>Range building <b style='color:#F5F5F5'>%d</b></span>"
            "<span>Errors <b style='color:#F5F5F5'>%d</b></span>"
            "<span>%s &middot; %s</span></div>"
            % (_ostats.get("scanned", 0), _ostats.get("with_breaks", 0),
               _ostats.get("building", 0), _ostats.get("errors", 0),
               st.session_state.get("orb_scan_scope", ""),
               st.session_state.get("orb_scan_time", "")),
            unsafe_allow_html=True)

    if _obuilding:
        st.markdown(
            "<div style='color:#7AA2F7;font-size:0.74rem;font-weight:700;letter-spacing:1.2px;"
            "margin:6px 0'>RANGE STILL BUILDING &mdash; completes 9:45 ET</div>",
            unsafe_allow_html=True)
        for _b in _obuilding[:12]:
            try:
                st.markdown(render_orb_building_html(_b["levels"]), unsafe_allow_html=True)
            except Exception:
                pass

    _show = []
    for _c in _ocands:
        if _orb_bucket != "All" and _c.get("bucket") != _orb_bucket:
            continue
        if _orb_dir == "Calls only" and _c.get("side") != "high":
            continue
        if _orb_dir == "Puts only" and _c.get("side") != "low":
            continue
        if _orb_hide_broken and _c.get("state") == "STRUCTURE_BROKEN":
            continue
        _show.append(_c)

    if _ocands and not _show:
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:6px;"
            "padding:16px;text-align:center;color:#6B6B6B;font-size:0.78rem'>"
            "Breaks were found, but none match the current filters.</div>",
            unsafe_allow_html=True)
    elif not _ocands and _ostats:
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:6px;"
            "padding:18px;text-align:center'>"
            "<div style='color:#F5F5F5;font-size:0.86rem;font-weight:700;margin-bottom:5px'>"
            "No ORB setups in scope</div>"
            "<div style='color:#6B6B6B;font-size:0.75rem;line-height:1.5'>"
            "Nothing has broken its opening range on a 5m close. That is a normal result &mdash; "
            "on a low-volatility morning the correct read is no trade.</div></div>",
            unsafe_allow_html=True)
    elif not _ocands:
        st.markdown(
            "<div style='color:#6B6B6B;font-size:0.78rem;padding:14px 0'>"
            "Pick a scope and run the scan, or look up a single ticker above.</div>",
            unsafe_allow_html=True)

    for _bk in ["GO NOW", "WATCHING", "ON DECK"]:
        _grp = [c for c in _show if c.get("bucket") == _bk]
        if not _grp:
            continue
        _bcol = ORB_BUCKET_COLOR.get(_bk, "#7AA2F7")
        st.markdown(
            "<div style='color:%s;font-size:0.74rem;font-weight:700;letter-spacing:1.4px;"
            "margin:16px 0 4px 0;padding-bottom:5px;border-bottom:1px solid #1A1A1A'>"
            "%s &nbsp;<span style='color:#6B6B6B;font-weight:400'>%d</span></div>"
            % (_bcol, _bk, len(_grp)), unsafe_allow_html=True)
        for _c in _grp:
            try:
                st.markdown(render_orb_card_html(
                    _c["levels"], _c["events"], _c["structure"],
                    _c["score_data"], _c["geo"], _c["side"]), unsafe_allow_html=True)
            except Exception as _ce:
                st.markdown("<div style='color:#C1121F;font-size:0.72rem'>Card error for "
                            + str(_c.get("ticker", "")) + ": " + str(_ce)[:90] + "</div>",
                            unsafe_allow_html=True)


with tab_mom:
    st.markdown(
        "<div style='background:#0A0A0A;border:1px solid #1F1F1F;border-radius:6px;"
        "padding:13px 16px;margin-bottom:12px'>"
        "<div style='color:#C1121F;font-size:0.72rem;font-weight:700;letter-spacing:1.6px;"
        "margin-bottom:5px'>\u26A1 OPENING MOMENTUM</div>"
        "<div style='color:#A1A1A6;font-size:0.76rem;line-height:1.5'>"
        "Catches hard directional moves in the 9:30&ndash;9:45 window &mdash; before the ORB range "
        "completes. This is the most violent part of the day and these setups are UNCONFIRMED. "
        "Read them as a heads-up, size small, and confirm with the range.</div></div>",
        unsafe_allow_html=True)

    # Time awareness — mode is live only in the opening window
    try:
        _now_et = datetime.now()
        _mins_now = _now_et.hour * 60 + _now_et.minute
        _in_window = (MOM_WINDOW_START <= _mins_now < MOM_WINDOW_END)
    except Exception:
        _in_window = False

    if not _in_window:
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-left:3px solid #6B6B6B;"
            "border-radius:6px;padding:11px 14px;margin-bottom:12px'>"
            "<div style='color:#F6E27A;font-size:0.72rem;line-height:1.5'>"
            "\u23F0 <b>Outside the opening window (9:30&ndash;9:45 ET).</b> This mode is built for "
            "the open. You can still scan to see what fired earlier &mdash; those names hand off to "
            "the ORB tab for the confirmed continuation &mdash; but fresh momentum reads are most "
            "reliable live at the bell.</div></div>", unsafe_allow_html=True)

    _mc1, _mc2 = st.columns([1, 1.4])
    with _mc1:
        _mom_scope = st.radio("Scope", ["My Watchlist", "Sector", "Full Universe"],
                              index=2, horizontal=True, key="mom_scope")
    with _mc2:
        _mom_sector = None
        if _mom_scope == "Sector":
            _mnames = [k for k in SECTOR_LISTS.keys() if k != "My Watchlist"]
            _mom_sector = st.selectbox("Which sector", _mnames, key="mom_sector_pick")

    if _mom_scope == "My Watchlist":
        _mom_list = list(st.session_state.get("user_watchlist", []) or []) or list(DEFAULT_WATCHLIST)
        _mscope_label = "My Watchlist (%d)" % len(_mom_list)
    elif _mom_scope == "Sector" and _mom_sector:
        _mom_list = list(SECTOR_LISTS.get(_mom_sector, []))
        _mscope_label = "%s (%d)" % (_mom_sector, len(_mom_list))
    else:
        _mom_list = list(SCAN_UNIVERSE)
        _mscope_label = "Full Universe (%d)" % len(_mom_list)
    _mom_list = list(dict.fromkeys([t for t in _mom_list if t]))

    _mom_run = st.button("SCAN OPENING MOMENTUM", type="primary",
                         use_container_width=True, key="mom_run_btn")

    if _mom_run and _mom_list:
        _mbar = st.progress(0.0); _mtxt = st.empty()
        def _mom_prog(done, total, tk):
            try:
                _mbar.progress(min(1.0, done / max(1, total)))
                _mtxt.markdown("<div style='color:#6B6B6B;font-size:0.72rem'>Scanning %d/%d &middot; %s</div>"
                               % (done, total, tk), unsafe_allow_html=True)
            except Exception:
                pass
        try:
            _thrust, _delayed, _nulls, _mstats = run_momentum_scan(
                _mom_list, max_workers=4, progress_cb=_mom_prog)
            st.session_state["mom_thrust"] = _thrust
            st.session_state["mom_delayed"] = _delayed
            st.session_state["mom_stats"] = _mstats
            st.session_state["mom_scan_time"] = datetime.now().strftime("%-I:%M:%S %p ET")
            st.session_state["mom_scope_label"] = _mscope_label
        except Exception as _me:
            st.error("Momentum scan failed: " + str(_me)[:170])
        finally:
            _mbar.empty(); _mtxt.empty()

    _thrust = st.session_state.get("mom_thrust", [])
    _delayed = st.session_state.get("mom_delayed", [])
    _mstats = st.session_state.get("mom_stats", {})

    if _mstats:
        _spy_d = _mstats.get("spy_dir", "flat"); _qqq_d = _mstats.get("qqq_dir", "flat")
        _dcol = {"up": "#22C55E", "down": "#C1121F", "flat": "#6B6B6B"}
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:0.72rem;"
            "color:#6B6B6B;padding:8px 0 10px 0'>"
            "<span>SPY open <b style='color:%s'>%s</b></span>"
            "<span>QQQ open <b style='color:%s'>%s</b></span>"
            "<span>Thrust <b style='color:#F5F5F5'>%d</b></span>"
            "<span>Forming <b style='color:#F5F5F5'>%d</b></span>"
            "<span>Errors <b style='color:#F5F5F5'>%d</b></span>"
            "<span>%s &middot; %s</span></div>"
            % (_dcol.get(_spy_d, "#6B6B6B"), _spy_d.upper(),
               _dcol.get(_qqq_d, "#6B6B6B"), _qqq_d.upper(),
               _mstats.get("thrust", 0), _mstats.get("delayed", 0), _mstats.get("errors", 0),
               st.session_state.get("mom_scope_label", ""),
               st.session_state.get("mom_scan_time", "")),
            unsafe_allow_html=True)

    if _thrust:
        st.markdown("<div style='color:#C1121F;font-size:0.74rem;font-weight:700;letter-spacing:1.4px;"
                    "margin:14px 0 4px 0;padding-bottom:5px;border-bottom:1px solid #1A1A1A'>"
                    "\u26A1 IMMEDIATE THRUST <span style='color:#6B6B6B;font-weight:400'>%d</span></div>"
                    % len(_thrust), unsafe_allow_html=True)
        for _m in _thrust:
            try: st.markdown(render_momentum_card_html(_m), unsafe_allow_html=True)
            except Exception as _e:
                st.markdown("<div style='color:#C1121F;font-size:0.72rem'>card error: " + str(_e)[:80] + "</div>",
                            unsafe_allow_html=True)

    if _delayed:
        st.markdown("<div style='color:#F6E27A;font-size:0.74rem;font-weight:700;letter-spacing:1.4px;"
                    "margin:16px 0 4px 0;padding-bottom:5px;border-bottom:1px solid #1A1A1A'>"
                    "\u26A1 DELAYED / FORMING <span style='color:#6B6B6B;font-weight:400'>%d</span></div>"
                    % len(_delayed), unsafe_allow_html=True)
        for _m in _delayed:
            try: st.markdown(render_momentum_card_html(_m), unsafe_allow_html=True)
            except Exception as _e:
                st.markdown("<div style='color:#C1121F;font-size:0.72rem'>card error: " + str(_e)[:80] + "</div>",
                            unsafe_allow_html=True)

    if _mstats and not _thrust and not _delayed:
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1F1F1F;border-radius:6px;"
            "padding:18px;text-align:center'>"
            "<div style='color:#F5F5F5;font-size:0.85rem;font-weight:700;margin-bottom:5px'>"
            "No opening thrust right now</div>"
            "<div style='color:#6B6B6B;font-size:0.75rem;line-height:1.5'>"
            "Nothing is breaking hard enough off the open to flag. On a calm morning that is the "
            "correct read &mdash; no forced trades.</div></div>", unsafe_allow_html=True)
    elif not _mstats:
        st.markdown("<div style='color:#6B6B6B;font-size:0.78rem;padding:12px 0'>"
                    "Scan to read opening momentum across your scope.</div>", unsafe_allow_html=True)


with tab_stats:
    _stats_is_admin = (
        st.session_state.get("is_admin", False) or
        st.session_state.get("user_email", "").strip().lower() == ADMIN_EMAIL.strip().lower()
    )
    if _stats_is_admin:
        overall = get_overall_stats()
        _ov_w  = overall.get("wins", 0) if overall else 0
        _ov_l  = overall.get("losses", 0) if overall else 0
        _ov_o  = overall.get("open", 0) if overall else 0
        _ov_wr = overall.get("win_rate") if overall else None
        _wr_str = (str(_ov_wr) + "% win rate") if _ov_wr else "Building dataset"
        st.markdown(
            "<div style='background:#0d0d0f;border:1px solid #2A2A2D;"
            "border-radius:10px;padding:16px 20px;margin-bottom:16px'>"
            "<div style='color:#D4AF37;font-family:monospace;font-size:0.7rem;"
            "letter-spacing:2px;margin-bottom:10px'>SIGNAL OUTCOMES</div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px'>"
            "<div><div style='font-size:1.3rem;font-weight:700;color:#F5F5F5'>"
            + _wr_str + "</div><div style='color:#A1A1A6;font-size:0.7rem'>Win Rate</div></div>"
            "<div><div style='font-size:1.3rem;font-weight:700;color:#22C55E'>"
            + str(_ov_w) + "</div><div style='color:#A1A1A6;font-size:0.7rem'>Wins</div></div>"
            "<div><div style='font-size:1.3rem;font-weight:700;color:#C1121F'>"
            + str(_ov_l) + "</div><div style='color:#A1A1A6;font-size:0.7rem'>Losses</div></div>"
            "<div><div style='font-size:1.3rem;font-weight:700;color:#A1A1A6'>"
            + str(_ov_o) + "</div><div style='color:#A1A1A6;font-size:0.7rem'>Pending</div></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Full analytics unlock after 30+ resolved trades.")
    else:
        st.markdown(
            "<div style='padding:40px;text-align:center;color:#A1A1A6;"
            "background:#1A1A1D;border-radius:10px;margin-top:20px'>"
            "<div style='color:#D4AF37;font-weight:700;margin-bottom:6px'>"
            "Performance Stats Coming Soon</div>"
            "Analytics are being calibrated. Win rates and edge data "
            "unlock once we have a meaningful dataset.</div>",
            unsafe_allow_html=True,
        )
