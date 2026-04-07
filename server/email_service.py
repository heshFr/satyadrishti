"""
Satya Drishti — Email Service
==============================
Async SMTP email sending with template support.
Falls back gracefully when SMTP is not configured (dev mode).
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from .config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_TLS, FRONTEND_URL

log = logging.getLogger("satyadrishti.email")

_smtp_configured = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def send_email(to: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """Send an email. Returns True on success, False on failure."""
    if not _smtp_configured:
        log.warning("SMTP not configured — email to %s suppressed (subject: %s)", to, subject)
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject

    if text_body:
        msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        if SMTP_TLS:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, to, msg.as_string())
        server.quit()
        log.info("Email sent to %s: %s", to, subject)
        return True
    except Exception as e:
        log.error("Failed to send email to %s: %s", to, e)
        return False


# ─── Email Templates ───────────────────────────────────────────────────

BRAND_STYLE = """
<style>
  body { font-family: 'Inter', system-ui, sans-serif; background: #0c1324; color: #dce1fb; margin: 0; padding: 20px; }
  .container { max-width: 600px; margin: 0 auto; background: #151b2d; border-radius: 16px; padding: 40px; border: 1px solid #2e3447; }
  .logo { color: #a4e6ff; font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 24px; }
  .code { font-size: 32px; font-weight: 800; letter-spacing: 0.3em; color: #00d1ff; background: #191f31; padding: 16px 24px; border-radius: 12px; text-align: center; margin: 24px 0; border: 1px solid #2e3447; }
  .text { color: #bbc9cf; line-height: 1.6; margin-bottom: 16px; }
  .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #2e3447; color: #859399; font-size: 12px; text-align: center; }
  .button { display: inline-block; background: #00d1ff; color: #003543; padding: 12px 32px; border-radius: 8px; font-weight: 700; text-decoration: none; margin: 16px 0; }
  .alert { background: #93000a20; border: 1px solid #93000a40; border-radius: 12px; padding: 16px; margin: 16px 0; }
  .alert-text { color: #ffb4ab; font-weight: 600; }
</style>
"""


def send_verification_email(to: str, name: str, code: str) -> bool:
    html = f"""<!DOCTYPE html><html><head>{BRAND_STYLE}</head><body>
    <div class="container">
        <div class="logo">Satya Drishti</div>
        <p class="text">Hi {name},</p>
        <p class="text">Welcome to Satya Drishti. Verify your email address with the code below:</p>
        <div class="code">{code}</div>
        <p class="text">This code expires in 24 hours. If you didn't create an account, you can safely ignore this email.</p>
        <div class="footer">Satya Drishti — AI-Powered Deepfake Detection</div>
    </div></body></html>"""
    return send_email(to, "Verify your Satya Drishti account", html, f"Your verification code: {code}")


def send_password_reset_email(to: str, name: str, code: str) -> bool:
    html = f"""<!DOCTYPE html><html><head>{BRAND_STYLE}</head><body>
    <div class="container">
        <div class="logo">Satya Drishti</div>
        <p class="text">Hi {name},</p>
        <p class="text">You requested a password reset. Use this code to set a new password:</p>
        <div class="code">{code}</div>
        <p class="text">This code expires in 15 minutes. If you didn't request this, please ignore this email and your password will remain unchanged.</p>
        <div class="footer">Satya Drishti — AI-Powered Deepfake Detection</div>
    </div></body></html>"""
    return send_email(to, "Reset your Satya Drishti password", html, f"Your password reset code: {code}")


def send_threat_alert_email(to: str, name: str, scan_id: str, verdict: str, confidence: float, file_name: str) -> bool:
    threat_color = "#ffb4ab" if verdict == "ai-generated" else "#a4e6ff"
    html = f"""<!DOCTYPE html><html><head>{BRAND_STYLE}</head><body>
    <div class="container">
        <div class="logo">Satya Drishti</div>
        <div class="alert">
            <p class="alert-text">Threat Detected: {verdict.upper()}</p>
        </div>
        <p class="text">Hi {name},</p>
        <p class="text">A scan has detected a potential threat in the file <strong>{file_name}</strong>:</p>
        <p class="text">
            <strong style="color:{threat_color}">Verdict:</strong> {verdict.upper()}<br>
            <strong>Confidence:</strong> {confidence:.1f}%<br>
            <strong>Scan ID:</strong> {scan_id}
        </p>
        <a href="{FRONTEND_URL}/history" class="button">View in Dashboard</a>
        <p class="text" style="font-size:13px; margin-top:24px;">If someone sent you this media, exercise caution. Consider reporting to Cyber Crime Helpline: <strong>1930</strong></p>
        <div class="footer">Satya Drishti — AI-Powered Deepfake Detection</div>
    </div></body></html>"""
    return send_email(to, f"[ALERT] Threat detected: {file_name}", html)


def send_family_alert_email(to: str, user_name: str, alert_level: str, message: str) -> bool:
    html = f"""<!DOCTYPE html><html><head>{BRAND_STYLE}</head><body>
    <div class="container">
        <div class="logo">Satya Drishti</div>
        <div class="alert">
            <p class="alert-text">Family Alert: {alert_level.upper()}</p>
        </div>
        <p class="text">{user_name} has triggered a safety alert on Satya Drishti.</p>
        <p class="text"><strong>Message:</strong> {message}</p>
        <p class="text">If you believe this person may be in danger, contact them immediately or call emergency services at <strong>112</strong>.</p>
        <a href="{FRONTEND_URL}" class="button">Open Satya Drishti</a>
        <div class="footer">This alert was sent automatically by Satya Drishti's protection system.</div>
    </div></body></html>"""
    return send_email(to, f"[SAFETY ALERT] {user_name} needs help", html)
