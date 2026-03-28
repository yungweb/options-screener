"""
PaidButPressured — Stripe Webhook Handler
Runs alongside dashboard.py on Railway.

Handles:
  1. Ebook purchase ($8.99)
     - Check if buyer already has active subscription in Supabase
     - If NO subscription  → create 7-day trial account + deliver ebook
     - If HAS subscription → deliver ebook only (no double trial)

  2. Subscription purchase (Founder $14.99 / Pressured $29.99)
     - Create or upgrade full account in Supabase
     - Clear any existing trial expiry (full access, no timer)

Start this alongside Streamlit in Railway by adding to your Procfile:
  web: streamlit run dashboard.py --server.port $PORT & python webhook.py

Environment variables needed (already in Railway):
  STRIPE_WEBHOOK_SECRET   — from Stripe Dashboard > Webhooks
  STRIPE_EBOOK_PRICE_ID   — price ID of your $8.99 ebook product
  SUPABASE_URL
  SUPABASE_KEY
  MAKE_WEBHOOK_URL         — fires email delivery via Make.com
  EBOOK_DOWNLOAD_URL       — direct link to your ebook PDF
"""

import os
import json
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pbp_webhook")

# ── Environment ───────────────────────────────────────────────────────────────
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_EBOOK_PRICE_ID = os.environ.get("STRIPE_EBOOK_PRICE_ID", "")
SUPABASE_URL          = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY          = os.environ.get("SUPABASE_KEY", "")
MAKE_WEBHOOK_URL         = os.environ.get("MAKE_WEBHOOK_URL", "")
MAKE_EBOOK_MEMBER_URL    = os.environ.get("MAKE_EBOOK_MEMBER_URL", "")
EBOOK_DOWNLOAD_URL    = os.environ.get("EBOOK_DOWNLOAD_URL", "")

TRIAL_DAYS  = 7
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", 8502))


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        log.error("Supabase connect error: %s", e)
        return None


def get_user_by_email(sb, email):
    """Return user_data row for this email, or None."""
    try:
        res = sb.table("user_data").select("*").eq("email", email).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        log.error("get_user_by_email error: %s", e)
        return None


def has_active_subscription(sb, email):
    """True if this email has a paid subscription (not trial-only)."""
    user = get_user_by_email(sb, email)
    if not user:
        return False
    tier = user.get("tier", "")
    return tier in ("founder", "pressured", "standard")


def create_trial_account(sb, email, name=""):
    """
    Create or update a trial account with 7-day expiry.
    Uses Supabase Auth to create the user, then stores trial metadata.
    Returns (success, temp_password)
    """
    import secrets
    import string

    temp_password = "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(12)
    )

    expiry = (datetime.now(tz=pytz.UTC) + timedelta(days=TRIAL_DAYS)).isoformat()

    try:
        # Create auth user
        resp = sb.auth.admin.create_user({
            "email":            email,
            "password":         temp_password,
            "email_confirm":    True,
            "user_metadata":    {"name": name, "tier": "trial"},
        })

        if not resp.user:
            raise Exception("Auth user creation returned no user")

        user_id = resp.user.id

        # Store trial metadata in user_data table
        sb.table("user_data").upsert({
            "user_id":    user_id,
            "email":      email,
            "tier":       "trial",
            "trial_expiry": expiry,
            "ebook_purchased": True,
            "created_at": datetime.now(tz=pytz.UTC).isoformat(),
            "updated_at": datetime.now(tz=pytz.UTC).isoformat(),
        }).execute()

        log.info("Trial account created for %s — expires %s", email, expiry)
        return True, temp_password

    except Exception as e:
        # User may already exist — update trial fields
        log.warning("create_trial_account fallback for %s: %s", email, e)
        try:
            existing = get_user_by_email(sb, email)
            if existing:
                sb.table("user_data").update({
                    "ebook_purchased":  True,
                    "updated_at":       datetime.now(tz=pytz.UTC).isoformat(),
                }).eq("email", email).execute()
                return True, None  # already has account, just deliver ebook
        except Exception as e2:
            log.error("Fallback update failed: %s", e2)
        return False, None


def upgrade_to_subscription(sb, email, tier, name=""):
    """
    Create or upgrade a full subscription account.
    Clears any trial expiry — full access, no timer.
    """
    try:
        existing = get_user_by_email(sb, email)

        if existing:
            # Upgrade existing account
            sb.table("user_data").update({
                "tier":          tier,
                "trial_expiry":  None,   # clear trial timer
                "updated_at":    datetime.now(tz=pytz.UTC).isoformat(),
            }).eq("email", email).execute()
            log.info("Upgraded %s to %s", email, tier)
        else:
            # New subscriber — create auth user
            import secrets, string
            temp_password = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(12)
            )
            resp = sb.auth.admin.create_user({
                "email":          email,
                "password":       temp_password,
                "email_confirm":  True,
                "user_metadata":  {"name": name, "tier": tier},
            })
            if resp.user:
                sb.table("user_data").upsert({
                    "user_id":       resp.user.id,
                    "email":         email,
                    "tier":          tier,
                    "trial_expiry":  None,
                    "created_at":    datetime.now(tz=pytz.UTC).isoformat(),
                    "updated_at":    datetime.now(tz=pytz.UTC).isoformat(),
                }).execute()
                log.info("New subscriber %s created as %s", email, tier)
                # Fire Make.com to send welcome email with login info
                fire_make_webhook({
                    "event":    "new_subscriber",
                    "email":    email,
                    "name":     name,
                    "tier":     tier,
                    "password": temp_password,
                    "login_url": "https://yungweb.github.io/paidbutpressured",
                })

        return True
    except Exception as e:
        log.error("upgrade_to_subscription error: %s", e)
        return False


# ── Make.com webhook ──────────────────────────────────────────────────────────

def fire_make_webhook(payload: dict, url: str = ""):
    """Send data to Make.com for email automation."""
    target_url = url if url else MAKE_WEBHOOK_URL
    if not target_url:
        log.warning("No Make.com URL set — skipping webhook")
        return
    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            target_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=8)
        log.info("Make.com webhook fired: %s", payload.get("event"))
    except Exception as e:
        log.error("Make.com webhook error: %s", e)


# ── Stripe signature verification ─────────────────────────────────────────────

def verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> bool:
    """Verify Stripe webhook signature to prevent spoofing."""
    try:
        parts     = {k: v for k, v in (p.split("=", 1) for p in sig_header.split(","))}
        timestamp = parts.get("t", "")
        signature = parts.get("v1", "")
        signed    = f"{timestamp}.".encode() + payload
        expected  = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False


# ── Event handlers ────────────────────────────────────────────────────────────

def handle_checkout_completed(session: dict):
    """
    Main routing logic for completed Stripe checkouts.
    Reads product metadata to determine ebook vs subscription.
    """
    sb = get_supabase()
    if not sb:
        log.error("Supabase unavailable — cannot process checkout")
        return

    email    = session.get("customer_details", {}).get("email", "")
    name     = session.get("customer_details", {}).get("name", "") or ""
    metadata = session.get("metadata", {})
    mode     = session.get("mode", "")           # "payment" or "subscription"
    tier     = metadata.get("tier", "")          # "founder", "pressured", "ebook"

    if not email:
        log.warning("No email in session — skipping")
        return

    log.info("Checkout completed: email=%s tier=%s mode=%s", email, tier, mode)

    # ── EBOOK PURCHASE ────────────────────────────────────────────────────
    if tier == "ebook" or mode == "payment":
        already_subscribed = has_active_subscription(sb, email)

        if already_subscribed:
            # Already a member — just deliver the ebook, no trial
            log.info("%s already subscribed — delivering ebook only", email)
            fire_make_webhook({
                "event":         "ebook_member",
                "email":         email,
                "name":          name,
                "ebook_url":     EBOOK_DOWNLOAD_URL,
                "message":       "You already have full app access. Here's your ebook!",
            }, url=MAKE_EBOOK_MEMBER_URL)
        else:
            # New buyer — create 7-day trial + deliver ebook
            success, temp_password = create_trial_account(sb, email, name)
            if success:
                payload = {
                    "event":        "ebook_trial",
                    "email":        email,
                    "name":         name,
                    "ebook_url":    EBOOK_DOWNLOAD_URL,
                    "trial_days":   TRIAL_DAYS,
                    "login_url":    "https://yungweb.github.io/paidbutpressured",
                    "trial_expiry": (
                        datetime.now(tz=pytz.UTC) + timedelta(days=TRIAL_DAYS)
                    ).strftime("%B %d, %Y"),
                }
                if temp_password:
                    payload["temp_password"] = temp_password
                fire_make_webhook(payload)
            else:
                log.error("Failed to create trial for %s", email)

    # ── SUBSCRIPTION PURCHASE ─────────────────────────────────────────────
    elif mode == "subscription" or tier in ("founder", "pressured", "standard"):
        resolved_tier = tier or ("founder" if "14" in str(session.get("amount_total", "")) else "pressured")
        upgrade_to_subscription(sb, email, resolved_tier, name)

    else:
        log.warning("Unknown checkout type: tier=%s mode=%s", tier, mode)


# ── HTTP Server ───────────────────────────────────────────────────────────────

class WebhookHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        log.info(format % args)

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"PBP webhook server running")

    def do_POST(self):
        if self.path != "/webhook":
            self.send_response(404)
            self.end_headers()
            return

        length  = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        sig     = self.headers.get("Stripe-Signature", "")

        # Verify signature
        if STRIPE_WEBHOOK_SECRET and not verify_stripe_signature(payload, sig, STRIPE_WEBHOOK_SECRET):
            log.warning("Invalid Stripe signature — rejecting request")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid signature")
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        event_type = event.get("type", "")
        log.info("Received event: %s", event_type)

        if event_type == "checkout.session.completed":
            handle_checkout_completed(event["data"]["object"])

        # Always return 200 to Stripe — never let retries pile up
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", WEBHOOK_PORT), WebhookHandler)
    log.info("PBP webhook server listening on port %s", WEBHOOK_PORT)
    server.serve_forever()
