#!/usr/bin/env bash
#
# SkillGuard Pre-Install Hook
#
# Usage: skillguard-install-hook.sh <skill-name> [version]
#
# This script checks a skill for safety issues before installation.
# Intended for use as a pre-install hook in the OpenClaw CLI.
#
# Manual testing:
#   chmod +x examples/skillguard-install-hook.sh
#   ./examples/skillguard-install-hook.sh weather-helper 1.0.0
#
# Modes:
#   1. Hosted API mode: set SKILLGUARD_API_URL to a running SkillGuard server
#      SKILLGUARD_API_URL=http://localhost:8080 ./examples/skillguard-install-hook.sh weather-helper
#
#   2. Local binary mode (default): requires skillguard + clawhub in PATH
#      Requirements:
#        - skillguard binary in PATH (or set SKILLGUARD_BIN)
#        - clawhub binary in PATH (or set CLAWHUB_BIN) for live skill export

set -euo pipefail

SKILL_NAME="${1:?Usage: $0 <skill-name> [version]}"
SKILL_VERSION="${2:-}"

SKILLGUARD_API_URL="${SKILLGUARD_API_URL:-}"

# ---- Hosted API mode ----
if [ -n "$SKILLGUARD_API_URL" ]; then
    echo "Using hosted SkillGuard API: ${SKILLGUARD_API_URL}"

    # Build request JSON
    if [ -n "$SKILL_VERSION" ]; then
        REQUEST_BODY="{\"skill\":\"${SKILL_NAME}\",\"version\":\"${SKILL_VERSION}\"}"
    else
        REQUEST_BODY="{\"skill\":\"${SKILL_NAME}\"}"
    fi

    SCAN_OUTPUT=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$REQUEST_BODY" \
        "${SKILLGUARD_API_URL}/api/v1/evaluate/name") || {
        echo "WARNING: Failed to reach SkillGuard API. Proceeding with caution."
        exit 0
    }

    # Check for API-level errors
    SUCCESS=$(echo "$SCAN_OUTPUT" | grep -o '"success":true' || true)
    if [ -z "$SUCCESS" ]; then
        ERROR=$(echo "$SCAN_OUTPUT" | grep -o '"error":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo "WARNING: SkillGuard API error: ${ERROR:-unknown}. Proceeding with caution."
        exit 0
    fi

    # Parse the decision from the evaluation
    DECISION=$(echo "$SCAN_OUTPUT" | grep -o '"decision":"[^"]*"' | head -1 | cut -d'"' -f4)
    CLASSIFICATION=$(echo "$SCAN_OUTPUT" | grep -o '"classification":"[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$DECISION" ]; then
        echo "WARNING: Could not determine safety decision. Proceeding with caution."
        exit 0
    fi

    echo "Classification: $CLASSIFICATION"
    echo "Decision: $DECISION"

    case "$DECISION" in
        allow)
            echo "Skill ${SKILL_NAME} is safe. Proceeding with installation."
            exit 0
            ;;
        flag)
            echo "WARNING: Skill ${SKILL_NAME} has been flagged for review."
            REASONING=$(echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4)
            [ -n "$REASONING" ] && echo "Reason: $REASONING"
            read -rp "Do you want to proceed with installation? [y/N] " CONFIRM
            case "$CONFIRM" in
                [yY]|[yY][eE][sS])
                    echo "Proceeding despite flag..."
                    exit 0
                    ;;
                *)
                    echo "Installation cancelled."
                    exit 1
                    ;;
            esac
            ;;
        deny)
            echo "BLOCKED: Skill ${SKILL_NAME} has been denied by SkillGuard."
            REASONING=$(echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4)
            [ -n "$REASONING" ] && echo "Reason: $REASONING"
            exit 1
            ;;
        *)
            echo "Unknown decision: $DECISION"
            exit 1
            ;;
    esac
fi

# ---- Local binary mode ----
SKILLGUARD_BIN="${SKILLGUARD_BIN:-skillguard}"
CLAWHUB_BIN="${CLAWHUB_BIN:-clawhub}"

TMP_DIR=$(mktemp -d)
TMP_JSON="${TMP_DIR}/${SKILL_NAME}.json"
trap 'rm -rf "$TMP_DIR"' EXIT

# Step 1: Export the skill to a temporary JSON file
echo "Exporting skill ${SKILL_NAME}..."
if [ -n "$SKILL_VERSION" ]; then
    "$CLAWHUB_BIN" export "$SKILL_NAME" --version "$SKILL_VERSION" --output "$TMP_JSON"
else
    "$CLAWHUB_BIN" export "$SKILL_NAME" --output "$TMP_JSON"
fi

# Step 2: Check the skill with SkillGuard
echo "Checking ${SKILL_NAME} for safety issues..."
SCAN_OUTPUT=$("$SKILLGUARD_BIN" check --input "$TMP_JSON" --format json 2>/dev/null) || true

# Step 3: Parse the decision from JSON output
DECISION=$(echo "$SCAN_OUTPUT" | grep -o '"decision":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$DECISION" ]; then
    echo "WARNING: Could not determine safety decision. Proceeding with caution."
    exit 0
fi

echo "Decision: $DECISION"

# Step 4: Act on the decision
case "$DECISION" in
    allow)
        echo "Skill ${SKILL_NAME} is safe. Proceeding with installation."
        if [ -n "$SKILL_VERSION" ]; then
            "$CLAWHUB_BIN" install "$SKILL_NAME" --version "$SKILL_VERSION"
        else
            "$CLAWHUB_BIN" install "$SKILL_NAME"
        fi
        ;;
    flag)
        echo "WARNING: Skill ${SKILL_NAME} has been flagged for review."
        echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4
        read -rp "Do you want to proceed with installation? [y/N] " CONFIRM
        case "$CONFIRM" in
            [yY]|[yY][eE][sS])
                echo "Installing ${SKILL_NAME} despite flag..."
                if [ -n "$SKILL_VERSION" ]; then
                    "$CLAWHUB_BIN" install "$SKILL_NAME" --version "$SKILL_VERSION"
                else
                    "$CLAWHUB_BIN" install "$SKILL_NAME"
                fi
                ;;
            *)
                echo "Installation cancelled."
                exit 1
                ;;
        esac
        ;;
    deny)
        echo "BLOCKED: Skill ${SKILL_NAME} has been denied by SkillGuard."
        echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4
        exit 1
        ;;
    *)
        echo "Unknown decision: $DECISION"
        exit 1
        ;;
esac
