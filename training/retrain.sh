#!/usr/bin/env bash
# End-to-end retrain pipeline: ingest feedback -> train -> export -> test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== SkillGuard Retrain Pipeline ==="
echo

# Step 1: Ingest feedback (if feedback.jsonl exists)
FEEDBACK_PATH="${FEEDBACK_PATH:-/var/data/skillguard-cache/feedback.jsonl}"
if [ -f "$FEEDBACK_PATH" ]; then
    echo "Step 1: Ingesting feedback from $FEEDBACK_PATH"
    python "$SCRIPT_DIR/ingest_feedback.py" --feedback-path "$FEEDBACK_PATH"
    echo "  Review candidates in training/feedback-candidates.json"
    echo "  Apply approved labels to training/real-labels.json before continuing."
    echo
else
    echo "Step 1: No feedback file found at $FEEDBACK_PATH, skipping"
fi

# Step 2: Build the Rust binary (needed for feature extraction)
echo "Step 2: Building skillguard binary"
cd "$PROJECT_DIR"
cargo build --release
echo

# Step 3: Train
echo "Step 3: Training model"
cd "$PROJECT_DIR"
python training/train.py --dataset real --num-classes 3 --export --augment-dangerous 20 --holdout-fraction 0.15
echo

# Step 4: Run tests
echo "Step 4: Running tests"
cargo test
echo

# Step 5: Re-run batch scan to refresh catalog (if crawler feature available)
# echo "Step 5: Refreshing catalog"
# cargo run --features crawler -- scan --from-awesome --format json --output scan-report.json

echo "=== Retrain complete ==="
echo "Next steps:"
echo "  1. Review training results in data/training_summary.json"
echo "  2. Commit updated weights in src/model.rs"
echo "  3. Update data/model_versions.json with new version"
echo "  4. Deploy"
