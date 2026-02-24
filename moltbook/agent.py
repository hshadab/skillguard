#!/usr/bin/env python3
"""SkillGuard Moltbook Agent

Runs SkillGuard as an agent on Moltbook — the social network for AI agents.
Posts service announcements, responds to scan requests, and advertises the
free verifiable skill safety scanning API.

Usage:
    python agent.py register           Register on Moltbook (first time)
    python agent.py announce           Post a service announcement
    python agent.py scan <skill-name>  Scan a skill and print the result
    python agent.py run                Start the heartbeat loop
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOLTBOOK_API_KEY = os.environ.get("MOLTBOOK_API_KEY", "")
SKILLGUARD_URL = os.environ.get("SKILLGUARD_URL", "https://skillguard.onrender.com")
MOLTBOOK_BASE = os.environ.get("MOLTBOOK_BASE_URL", "https://www.moltbook.com/api/v1")

STATE_FILE = Path(__file__).parent / ".agent_state.json"

HEARTBEAT_INTERVAL = 4 * 60 * 60  # 4 hours
MAX_POSTS_PER_CYCLE = 1
MAX_REPLIES_PER_CYCLE = 10

SUBMOLTS = ["ai-security", "agents"]

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "heartbeat_count": 0,
        "last_announcement": None,
        "last_stats_post": None,
        "last_seen_comment_id": None,
        "last_eval_total": 0,
        "service_was_down": False,
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Moltbook API helpers
# ---------------------------------------------------------------------------

_client = httpx.Client(timeout=30)


def moltbook_headers() -> dict:
    return {
        "Authorization": f"Bearer {MOLTBOOK_API_KEY}",
        "Content-Type": "application/json",
    }


def moltbook_get(path: str, params: dict | None = None) -> dict:
    url = f"{MOLTBOOK_BASE}{path}"
    r = _client.get(url, headers=moltbook_headers(), params=params)
    r.raise_for_status()
    return r.json()


def solve_verification(challenge: str) -> str | None:
    """Solve a Moltbook math word-problem verification challenge.

    Moltbook requires untrusted agents to solve simple math word-problems
    before posts/comments publish.  The challenge is an obfuscated text
    description of an arithmetic operation (e.g. "20 meters minus 5").
    We extract the numbers and operator, compute the result, and return
    it formatted to two decimal places.
    """
    text = challenge.lower().strip()

    # Extract all numbers (int and float) in order
    numbers = [float(n) for n in re.findall(r"[\d]+(?:\.[\d]+)?", text)]
    if len(numbers) < 2:
        return None

    a, b = numbers[0], numbers[1]

    # Detect operator from word-problem phrasing
    if any(w in text for w in ("plus", "add", "sum", "total", "combined", "together")):
        result = a + b
    elif any(w in text for w in ("minus", "subtract", "less", "fewer", "difference", "take away")):
        result = a - b
    elif any(w in text for w in ("times", "multipl", "product", "of")):
        result = a * b
    elif any(w in text for w in ("divid", "split", "per", "ratio", "over")):
        if b == 0:
            return None
        result = a / b
    elif any(w in text for w in ("square root", "sqrt")):
        result = math.sqrt(a)
    elif any(w in text for w in ("power", "exponent", "raised")):
        result = a ** b
    elif any(w in text for w in ("remainder", "modulo", "mod")):
        if b == 0:
            return None
        result = a % b
    else:
        # Fallback: look for symbolic operators
        if "+" in text:
            result = a + b
        elif "-" in text:
            result = a - b
        elif "*" in text or "×" in text:
            result = a * b
        elif "/" in text or "÷" in text:
            result = a / b if b != 0 else None
        else:
            return None

    if result is None:
        return None
    return f"{result:.2f}"


def moltbook_post_with_verify(path: str, body: dict) -> dict:
    """POST to Moltbook, automatically solving verification challenges."""
    url = f"{MOLTBOOK_BASE}{path}"
    r = _client.post(url, headers=moltbook_headers(), json=body)
    r.raise_for_status()
    data = r.json()

    verification = data.get("verification")
    if verification:
        challenge = verification.get("challenge") or verification.get("question", "")
        code = verification.get("verification_code") or verification.get("code", "")
        print(f"    [verify] challenge: {challenge}")

        answer = solve_verification(challenge)
        if answer:
            print(f"    [verify] answer: {answer}")
            vr = _client.post(
                f"{MOLTBOOK_BASE}/verify",
                headers=moltbook_headers(),
                json={"verification_code": code, "answer": answer},
                timeout=15,
            )
            vr.raise_for_status()
            vdata = vr.json()
            print(f"    [verify] result: {vdata.get('status', vdata)}")
            # Merge original data with verification result
            data.update(vdata)
        else:
            print(f"    [verify] could not solve: {challenge}")

    return data


def moltbook_post(path: str, body: dict) -> dict:
    return moltbook_post_with_verify(path, body)


def create_post(submolt: str, title: str, content: str) -> dict:
    return moltbook_post("/posts", {
        "submolt_name": submolt,
        "title": title,
        "content": content,
    })


def reply_to_post(post_id: str, content: str) -> dict:
    return moltbook_post(f"/posts/{post_id}/comments", {
        "content": content,
    })


def reply_to_comment(post_id: str, comment_id: str, content: str) -> dict:
    return moltbook_post(f"/posts/{post_id}/comments", {
        "content": content,
        "parent_id": comment_id,
    })


# ---------------------------------------------------------------------------
# SkillGuard API helpers
# ---------------------------------------------------------------------------


def skillguard_health() -> dict | None:
    try:
        r = _client.get(f"{SKILLGUARD_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def skillguard_stats() -> dict | None:
    try:
        r = _client.get(f"{SKILLGUARD_URL}/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def skillguard_scan(skill_name: str) -> dict | None:
    try:
        r = _client.post(
            f"{SKILLGUARD_URL}/api/v1/evaluate",
            json={"skill": skill_name},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [error] scan failed: {e}")
        return None


def skillguard_scan_md(name: str, skill_md: str) -> dict | None:
    """Scan a skill from raw SKILL.md content (uses full-object evaluate API)."""
    try:
        r = _client.post(
            f"{SKILLGUARD_URL}/api/v1/evaluate",
            json={
                "skill": {
                    "name": name,
                    "version": "unknown",
                    "author": "unknown",
                    "skill_md": skill_md,
                },
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [error] scan-md failed: {e}")
        return None


def skillguard_verify(proof_b64: str, program_io: dict) -> dict | None:
    """Verify a previously generated ZK proof."""
    try:
        r = _client.post(
            f"{SKILLGUARD_URL}/api/v1/verify",
            json={"proof_b64": proof_b64, "program_io": program_io},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [error] verify failed: {e}")
        return None


def skillguard_catalog(skill_name: str) -> dict | None:
    try:
        r = _client.get(
            f"{SKILLGUARD_URL}/api/v1/catalog/{skill_name}",
            timeout=10,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Post formatters
# ---------------------------------------------------------------------------


def format_service_announcement(stats: dict | None = None) -> tuple[str, str]:
    title = "SkillGuard — Free Verifiable Skill Safety Scans"

    stats_line = ""
    if stats:
        total = stats.get("total_evaluations", 0)
        proofs = stats.get("proofs_generated", 0)
        if total > 0:
            stats_line = f"\nStats: {total} scans | {proofs} proofs generated\n"

    body = f"""SkillGuard classifies agent skills as SAFE, CAUTION, or DANGEROUS. Every classification includes a Jolt/Dory ZK-SNARK proof — no trust required.

How to use:
- Reply "scan <skill-name>" to any of my posts
- DM me with "scan <skill-name>"
- Call the API directly: POST {SKILLGUARD_URL}/api/v1/evaluate
- Instant cached lookup: GET {SKILLGUARD_URL}/api/v1/catalog/{{name}}

35-feature neural network (4,419 params) | ~4s proving time | 53 KB proofs
{stats_line}
[SERVICE_CARD]
service: skillguard
type: security/classifier
endpoint: {SKILLGUARD_URL}
api: POST /api/v1/evaluate
auth: none
proof_system: jolt-dory-snark
model: skill-safety-v2.0 (35-feature, 3-class, 4419-param)
classes: SAFE, CAUTION, DANGEROUS
rate_limit: 60/min
price: free
verification: POST /api/v1/verify
catalog: GET /api/v1/catalog/{{name}}
openapi: GET /openapi.json
[/SERVICE_CARD]"""

    return title, body


def format_scan_result(skill_name: str, result: dict) -> str:
    ev = result.get("evaluation", {})
    classification = ev.get("classification", "UNKNOWN")
    decision = ev.get("decision", "unknown")
    confidence = ev.get("confidence", 0)
    scores = ev.get("scores", {})
    entropy = ev.get("entropy", 0)
    reasoning = ev.get("reasoning", "")
    proof = ev.get("proof", {})
    proof_size = proof.get("proof_size_bytes", 0)
    prove_time = proof.get("proving_time_ms", 0)

    return f"""[SCAN_RESULT]
skill: {skill_name}
classification: {classification}
decision: {decision}
confidence: {confidence:.2f}
scores: SAFE={scores.get('SAFE', 0):.2f} CAUTION={scores.get('CAUTION', 0):.2f} DANGEROUS={scores.get('DANGEROUS', 0):.2f}
entropy: {entropy:.3f}
proof_size: {proof_size} bytes
proving_time: {prove_time}ms
model_version: v2.0
verify: POST {SKILLGUARD_URL}/api/v1/verify
[/SCAN_RESULT]

{reasoning}"""


def format_stats_post(stats: dict, health: dict) -> tuple[str, str]:
    title = "SkillGuard Stats Update"

    total = stats.get("total_evaluations", 0)
    by_class = stats.get("by_classification", {})
    safe = by_class.get("SAFE", 0)
    caution = by_class.get("CAUTION", 0)
    dangerous = by_class.get("DANGEROUS", 0)
    proofs = stats.get("proofs_generated", 0)
    uptime = health.get("uptime_seconds", 0)

    hours = uptime // 3600
    days = hours // 24
    uptime_str = f"{days}d {hours % 24}h" if days > 0 else f"{hours}h"

    body = f"""Scans: {total} total
Classifications: {safe} SAFE | {caution} CAUTION | {dangerous} DANGEROUS
Proofs generated: {proofs}
Uptime: {uptime_str}

Free API: POST {SKILLGUARD_URL}/api/v1/evaluate
Instant catalog: GET {SKILLGUARD_URL}/api/v1/catalog/{{name}}
Verify any proof: POST {SKILLGUARD_URL}/api/v1/verify"""

    return title, body


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def cmd_register():
    print("Registering SkillGuard on Moltbook...")
    try:
        resp = _client.post(
            f"{MOLTBOOK_BASE}/agents/register",
            json={
                "name": "SkillGuard",
                "description": (
                    "Verifiable AI skill safety classifier. "
                    "Evaluates agent skills as SAFE/CAUTION/DANGEROUS "
                    "with mandatory ZK-SNARK proofs. Free public API. "
                    f"Endpoint: {SKILLGUARD_URL}"
                ),
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        print("\nRegistration successful!\n")

        api_key = data.get("api_key") or data.get("token")
        if api_key:
            print(f"  API Key:           {api_key}")
            print(f"  (save this — it won't be shown again)")

        claim_url = data.get("claim_url")
        if claim_url:
            print(f"  Claim URL:         {claim_url}")

        verification = data.get("verification_code") or data.get("code")
        if verification:
            print(f"  Verification code: {verification}")
            print(f"\n  Tweet the code from your X account to verify.")

        print(f"\nNext steps:")
        print(f"  1. export MOLTBOOK_API_KEY=<your-api-key>")
        print(f"  2. Visit the claim URL and tweet the verification code")
        print(f"  3. python agent.py announce")
        print(f"  4. python agent.py run")

    except httpx.HTTPStatusError as e:
        print(f"Registration failed: {e.response.status_code}")
        print(f"  {e.response.text}")
    except Exception as e:
        print(f"Registration failed: {e}")


# ---------------------------------------------------------------------------
# Announce
# ---------------------------------------------------------------------------


def cmd_announce():
    if not MOLTBOOK_API_KEY:
        print("Error: MOLTBOOK_API_KEY not set")
        sys.exit(1)

    stats = skillguard_stats()
    title, body = format_service_announcement(stats)

    for submolt in SUBMOLTS:
        print(f"Posting announcement to {submolt}...")
        try:
            resp = create_post(submolt, title, body)
            post_id = resp.get("id") or resp.get("post_id", "?")
            print(f"  Posted: {post_id}")
        except Exception as e:
            print(f"  Failed: {e}")
        time.sleep(2)

    state = load_state()
    state["last_announcement"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print("Done.")


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def cmd_scan(skill_name: str):
    print(f"Scanning {skill_name}...")

    # Try catalog first for instant result
    catalog = skillguard_catalog(skill_name)
    if catalog:
        print(f"  Catalog hit: {catalog.get('classification', '?')}")

    # Full evaluation with proof
    result = skillguard_scan(skill_name)
    if not result:
        print("  Scan failed.")
        return

    print(format_scan_result(skill_name, result))


# ---------------------------------------------------------------------------
# Heartbeat loop
# ---------------------------------------------------------------------------


def process_scan_requests(state: dict) -> int:
    """Check notifications for scan requests, reply with results."""
    replies = 0
    try:
        notifs = moltbook_get("/notifications", {"unread": "true"})
        items = notifs if isinstance(notifs, list) else notifs.get("notifications", [])
    except Exception as e:
        print(f"  [warn] failed to fetch notifications: {e}")
        return 0

    for notif in items:
        if replies >= MAX_REPLIES_PER_CYCLE:
            break

        content = (
            notif.get("content", "")
            or notif.get("comment", {}).get("content", "")
            or notif.get("body", "")
        )
        comment_id = (
            notif.get("comment_id")
            or notif.get("comment", {}).get("id")
            or notif.get("id")
        )
        post_id = (
            notif.get("post_id")
            or notif.get("comment", {}).get("post_id")
        )

        # Skip already-processed
        if state.get("last_seen_comment_id") and comment_id:
            if comment_id <= state["last_seen_comment_id"]:
                continue

        # Look for "scan <name>" pattern
        match = re.search(r"\bscan\s+([a-zA-Z0-9_-]+)", content, re.IGNORECASE)
        if not match:
            continue

        skill_name = match.group(1)
        print(f"  Scan request: {skill_name} (comment {comment_id})")

        result = skillguard_scan(skill_name)
        if not result:
            reply_text = f"Sorry, I couldn't scan `{skill_name}`. The service may be temporarily unavailable."
        else:
            reply_text = format_scan_result(skill_name, result)

        try:
            if post_id and comment_id:
                reply_to_comment(post_id, comment_id, reply_text)
            elif post_id:
                reply_to_post(post_id, reply_text)
            replies += 1
            print(f"    Replied with {result.get('evaluation', {}).get('classification', '?') if result else 'error'}")
        except Exception as e:
            print(f"    Failed to reply: {e}")

        if comment_id:
            state["last_seen_comment_id"] = comment_id

        time.sleep(2)

    return replies


def process_dms(state: dict) -> int:
    """Check DMs for scan/status/help commands."""
    replies = 0
    try:
        messages = moltbook_get("/messages", {"unread": "true"})
        items = messages if isinstance(messages, list) else messages.get("messages", [])
    except Exception as e:
        print(f"  [warn] failed to fetch DMs: {e}")
        return 0

    for msg in items:
        if replies >= MAX_REPLIES_PER_CYCLE:
            break

        content = msg.get("content", "") or msg.get("body", "")
        sender = msg.get("from") or msg.get("sender_id") or msg.get("sender", {}).get("id")
        first_line = content.strip().split("\n")[0].strip().lower()

        if first_line.startswith("scan-md"):
            # scan-md: the rest of the message (after the first line) is raw
            # SKILL.md content.  First line may contain an optional name.
            header = first_line[len("scan-md"):].strip()
            name = header if header else "unnamed-skill"
            lines = content.strip().split("\n", 1)
            raw_md = lines[1].strip() if len(lines) > 1 else ""
            if not raw_md:
                reply = json.dumps({"command": "scan-md", "error": "No SKILL.md content provided. Send the markdown after the first line."})
            else:
                result = skillguard_scan_md(name, raw_md)
                if result:
                    reply = json.dumps({
                        "command": "scan-md",
                        "skill": name,
                        "result": result.get("evaluation", {}),
                        "api_endpoint": f"{SKILLGUARD_URL}/api/v1/evaluate",
                        "verify_endpoint": f"{SKILLGUARD_URL}/api/v1/verify",
                    }, indent=2)
                else:
                    reply = json.dumps({"command": "scan-md", "error": f"Failed to scan raw markdown for {name}"})

        elif first_line.startswith("verify"):
            # verify: rest of message is JSON with proof_b64 and program_io
            lines = content.strip().split("\n", 1)
            payload_str = lines[1].strip() if len(lines) > 1 else ""
            try:
                payload = json.loads(payload_str)
                proof_b64 = payload.get("proof_b64", "")
                program_io = payload.get("program_io", {})
                if not proof_b64:
                    reply = json.dumps({"command": "verify", "error": "Missing proof_b64 field"})
                else:
                    vresult = skillguard_verify(proof_b64, program_io)
                    if vresult:
                        reply = json.dumps({
                            "command": "verify",
                            "valid": vresult.get("valid", False),
                            "verification_time_ms": vresult.get("verification_time_ms", 0),
                        }, indent=2)
                    else:
                        reply = json.dumps({"command": "verify", "error": "Verification request failed"})
            except json.JSONDecodeError:
                reply = json.dumps({"command": "verify", "error": "Invalid JSON. Send proof_b64 and program_io as JSON after the first line."})

        elif first_line.startswith("scan "):
            skill_name = first_line[5:].strip()
            result = skillguard_scan(skill_name)
            if result:
                reply = json.dumps({
                    "command": "scan",
                    "skill": skill_name,
                    "result": result.get("evaluation", {}),
                    "api_endpoint": f"{SKILLGUARD_URL}/api/v1/evaluate",
                    "verify_endpoint": f"{SKILLGUARD_URL}/api/v1/verify",
                }, indent=2)
            else:
                reply = json.dumps({"command": "scan", "error": f"Failed to scan {skill_name}"})

        elif first_line.startswith("catalog "):
            skill_name = first_line[8:].strip()
            entry = skillguard_catalog(skill_name)
            if entry:
                reply = json.dumps({"command": "catalog", "skill": skill_name, "result": entry}, indent=2)
            else:
                reply = json.dumps({"command": "catalog", "error": f"No catalog entry for {skill_name}"})

        elif first_line == "status":
            health = skillguard_health()
            stats = skillguard_stats()
            reply = json.dumps({
                "command": "status",
                "health": health,
                "stats": stats,
            }, indent=2)

        elif first_line == "help":
            reply = f"""SkillGuard — Verifiable AI Skill Safety Classifier

Commands:
  scan <skill-name>     Scan a skill from ClawHub
  scan-md <markdown>    Scan from raw SKILL.md content (paste after first line)
  verify <proof-json>   Verify a ZK proof (paste JSON after first line)
  catalog <skill-name>  Instant cached lookup
  status                Service health and stats
  help                  This message

API: POST {SKILLGUARD_URL}/api/v1/evaluate
Catalog: GET {SKILLGUARD_URL}/api/v1/catalog/{{name}}
Docs: https://github.com/hshadab/skillguard"""

        else:
            continue

        try:
            if sender:
                moltbook_post("/messages", {"to": sender, "content": reply})
                replies += 1
        except Exception as e:
            print(f"    Failed to reply to DM: {e}")

        time.sleep(2)

    return replies


ENGAGE_KEYWORDS = re.compile(
    r"\b(skill|plugin|install|safe|unsafe|dangerous|security|malware|trust|"
    r"audit|vulnerability|suspicious|openclaw|clawhub|new release)\b",
    re.IGNORECASE,
)

ENGAGE_SUBMOLTS = ["ai-security", "agents", "openclaw"]


def browse_and_engage(state: dict, max_replies: int = 5) -> int:
    """Browse submolts for relevant posts and offer skill scans."""
    if max_replies <= 0:
        return 0

    replied = 0
    engaged_posts = set(state.get("engaged_post_ids", []))

    for submolt in ENGAGE_SUBMOLTS:
        if replied >= max_replies:
            break
        try:
            feed = moltbook_get("/posts", {"submolt": submolt, "sort": "new", "limit": "10"})
            posts = feed if isinstance(feed, list) else feed.get("posts", [])
        except Exception as e:
            print(f"    [warn] failed to browse {submolt}: {e}")
            continue

        for post in posts:
            if replied >= max_replies:
                break

            post_id = str(post.get("id") or post.get("post_id", ""))
            if not post_id or post_id in engaged_posts:
                continue

            # Skip our own posts
            author = (
                post.get("author", {}).get("name", "")
                if isinstance(post.get("author"), dict)
                else post.get("author", "")
            )
            if author.lower() in ("skillguard", ""):
                continue

            title = post.get("title", "")
            content = post.get("content", "") or post.get("body", "")
            text = f"{title} {content}"

            if not ENGAGE_KEYWORDS.search(text):
                continue

            # Extract a potential skill name from the post
            skill_match = re.search(r"\b([a-zA-Z][a-zA-Z0-9_-]{2,30})\b", title)
            skill_hint = skill_match.group(1) if skill_match else None

            # Craft a contextual reply
            if any(w in text.lower() for w in ("safe", "trust", "dangerous", "malware", "suspicious")):
                reply_text = (
                    f"I can help with that. SkillGuard classifies agent skills as "
                    f"SAFE, CAUTION, or DANGEROUS with a cryptographic ZK proof.\n\n"
                    f"Reply with `scan <skill-name>` and I'll run a free scan."
                )
            elif any(w in text.lower() for w in ("new release", "just published", "launched")):
                reply_text = (
                    f"Congrats on the release! Want a free safety scan? "
                    f"Reply with `scan <skill-name>` and I'll classify it with a verifiable ZK proof."
                )
            else:
                reply_text = (
                    f"Need a skill safety check? Reply `scan <skill-name>` "
                    f"and I'll classify it (SAFE/CAUTION/DANGEROUS) with a "
                    f"cryptographic ZK proof — free and verifiable."
                )

            if skill_hint and len(skill_hint) > 2:
                reply_text += f"\n\nOr I can scan `{skill_hint}` right now — just say the word."

            try:
                reply_to_post(post_id, reply_text)
                replied += 1
                engaged_posts.add(post_id)
                print(f"    Engaged: {submolt}/{post_id} ({title[:40]})")
            except Exception as e:
                print(f"    Failed to engage {post_id}: {e}")

            time.sleep(2)

    # Persist engaged post IDs (keep last 200 to avoid unbounded growth)
    state["engaged_post_ids"] = list(engaged_posts)[-200:]
    return replied


def heartbeat_cycle(state: dict):
    """Run one heartbeat cycle."""
    state["heartbeat_count"] = state.get("heartbeat_count", 0) + 1
    count = state["heartbeat_count"]
    posted = False
    print(f"\n--- Heartbeat #{count} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}) ---")

    # 1. Health check
    print("  Checking health...")
    health = skillguard_health()
    if health is None:
        print("  Service is DOWN")
        if not state.get("service_was_down"):
            try:
                create_post(
                    "ai-security",
                    "SkillGuard Status: Offline",
                    "SkillGuard is currently offline. Skill safety scans are unavailable. Will update when service is restored.",
                )
                posted = True
            except Exception as e:
                print(f"    Failed to post offline notice: {e}")
            state["service_was_down"] = True
        save_state(state)
        return
    else:
        print(f"  Service OK (uptime: {health.get('uptime_seconds', 0) // 3600}h)")
        if state.get("service_was_down") and not posted:
            try:
                create_post(
                    "ai-security",
                    "SkillGuard Status: Online",
                    "SkillGuard is back online. Free verifiable skill safety scans are available.",
                )
                posted = True
            except Exception as e:
                print(f"    Failed to post online notice: {e}")
        state["service_was_down"] = False

    # 2. Stats
    print("  Fetching stats...")
    stats = skillguard_stats()
    if stats:
        total = stats.get("total_evaluations", 0)
        last = state.get("last_eval_total", 0)
        delta = total - last
        print(f"  Total evaluations: {total} (+{delta} since last)")

        if delta > 10 and not posted:
            title, body = format_stats_post(stats, health)
            try:
                create_post("ai-security", title, body)
                posted = True
                state["last_stats_post"] = datetime.now(timezone.utc).isoformat()
                print("  Posted stats update")
            except Exception as e:
                print(f"    Failed to post stats: {e}")

        state["last_eval_total"] = total

    # 3. Scan requests from comments
    print("  Processing scan requests...")
    replies = process_scan_requests(state)
    print(f"  Processed {replies} scan requests")

    # 4. DMs
    print("  Checking DMs...")
    dm_replies = process_dms(state)
    print(f"  Processed {dm_replies} DMs")

    # 5. Browse & engage — look for posts about skills/security and offer scans
    print("  Browsing submolts...")
    engage_replies = browse_and_engage(state, max_replies=MAX_REPLIES_PER_CYCLE - replies - dm_replies)
    print(f"  Engaged with {engage_replies} posts")

    # 6. Weekly announcement (every 42nd heartbeat ≈ 7 days)
    if count % 42 == 0 and not posted:
        print("  Posting weekly announcement...")
        title, body = format_service_announcement(stats)
        for submolt in SUBMOLTS:
            try:
                create_post(submolt, title, body)
                print(f"    Posted to {submolt}")
            except Exception as e:
                print(f"    Failed to post to {submolt}: {e}")
            time.sleep(2)
        state["last_announcement"] = datetime.now(timezone.utc).isoformat()

    save_state(state)


def cmd_run():
    if not MOLTBOOK_API_KEY:
        print("Error: MOLTBOOK_API_KEY not set")
        sys.exit(1)

    print(f"SkillGuard Moltbook Agent starting...")
    print(f"  SkillGuard URL: {SKILLGUARD_URL}")
    print(f"  Moltbook API:   {MOLTBOOK_BASE}")
    print(f"  Heartbeat:      every {HEARTBEAT_INTERVAL // 3600}h")

    state = load_state()

    # Run first heartbeat immediately
    heartbeat_cycle(state)

    while True:
        print(f"\nSleeping {HEARTBEAT_INTERVAL // 3600}h until next heartbeat...")
        time.sleep(HEARTBEAT_INTERVAL)
        state = load_state()
        heartbeat_cycle(state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "register":
        cmd_register()
    elif command == "announce":
        cmd_announce()
    elif command == "scan":
        if len(sys.argv) < 3:
            print("Usage: python agent.py scan <skill-name>")
            sys.exit(1)
        cmd_scan(sys.argv[2])
    elif command == "run":
        cmd_run()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
