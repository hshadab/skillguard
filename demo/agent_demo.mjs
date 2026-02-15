#!/usr/bin/env node
// agent_demo.mjs — Autonomous agent paying for SkillGuard evaluations via x402
//
// Usage:
//   export PRIVATE_KEY="0x..."
//   export SKILLGUARD_URL="https://skillguard.example.com"   # optional, defaults to localhost
//   node agent_demo.mjs

import { createWalletClient, http } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { wrapFetchWithPayment } from "x402-fetch";

// ── Configuration ──────────────────────────────────────────────────────────
const PRIVATE_KEY = process.env.PRIVATE_KEY;
if (!PRIVATE_KEY) {
  console.error("Error: set PRIVATE_KEY env var (hex, with 0x prefix)");
  console.error("  e.g.  export PRIVATE_KEY=0xabc123...");
  process.exit(1);
}

const BASE_URL = process.env.SKILLGUARD_URL || "http://localhost:8080";
const EVALUATE_URL = `${BASE_URL}/api/v1/evaluate`;

// ── Wallet setup ───────────────────────────────────────────────────────────
const account = privateKeyToAccount(PRIVATE_KEY);

const walletClient = createWalletClient({
  account,
  chain: base,
  transport: http(),
});

console.log(`Agent wallet: ${account.address}`);
console.log(`Target:       ${EVALUATE_URL}`);
console.log();

// ── Wrap fetch with x402 automatic payment ─────────────────────────────────
const fetchWithPayment = wrapFetchWithPayment(fetch, walletClient);

// ── Sample skill to evaluate ───────────────────────────────────────────────
const payload = {
  skill: {
    name: "agent-recon-tool",
    version: "1.0.0",
    author: "agent-alpha",
    description: "A network reconnaissance tool for mapping infrastructure",
    skill_md: [
      "## Network Recon Tool",
      "This skill scans internal networks and enumerates services.",
      "",
      "### Permissions",
      "- network: *",
      "- exec: nmap, masscan",
      "- filesystem: read /etc/hosts, write /tmp/scan-results",
    ].join("\n"),
    scripts: [
      {
        name: "scan.sh",
        content: "#!/bin/bash\nnmap -sV -p- $TARGET_RANGE\nmasscan $TARGET_RANGE -p1-65535 --rate=10000",
        extension: "sh",
      },
    ],
    metadata: {},
    files: [],
  },
};

// ── Make the request (x402-fetch handles 402 → sign → retry) ──────────────
console.log("Sending evaluation request...");
console.log();

try {
  const response = await fetchWithPayment(EVALUATE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok || !data.success) {
    console.error("Request failed:", data.error || `HTTP ${response.status}`);
    process.exit(1);
  }

  // Display result
  const ev = data.evaluation;
  if (!ev) {
    console.error("Unexpected response format:", JSON.stringify(data, null, 2));
    process.exit(1);
  }

  console.log("=== Classification Result ===");
  console.log(`  Classification: ${ev.classification}`);
  console.log(`  Decision:       ${ev.decision}`);
  if (ev.confidence != null) {
    console.log(`  Confidence:     ${(ev.confidence * 100).toFixed(1)}%`);
  }
  if (ev.reasoning) {
    console.log(`  Reasoning:      ${ev.reasoning}`);
  }
  console.log(`  Processing:     ${data.processing_time_ms} ms`);

  // Check for x402 payment receipt
  const paymentReceipt = response.headers.get("X-Payment-Response");
  if (paymentReceipt) {
    console.log();
    console.log("Payment settled on Base via x402.");
  }

  console.log();
  console.log("Done.");
} catch (err) {
  console.error("Error:", err.message || err);
  process.exit(1);
}
