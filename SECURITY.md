# Security

## Reporting Vulnerabilities

If you discover a security vulnerability in SkillGuard, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainer or use GitHub's private vulnerability reporting feature.
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

## Known Limitations

SkillGuard is a defense-in-depth layer. It is not a guarantee of safety.

### Evasion Techniques

The following techniques may bypass SkillGuard's detection:

- **Novel obfuscation**: Multi-stage payloads that decode at runtime through encrypted blobs or custom encoding schemes not covered by regex patterns.
- **Polyglot files**: Files that are valid in multiple formats and execute differently depending on the interpreter.
- **Unicode tricks**: Homoglyph attacks using characters that look identical to ASCII but aren't matched by patterns (e.g., Cyrillic `а` vs Latin `a`). SkillGuard detects zero-width characters but not all homoglyph substitutions.
- **Indirect execution**: Skills that install a benign dependency which itself contains malicious code (supply chain attacks via transitive dependencies).
- **Time-delayed payloads**: Skills that appear safe at scan time but download and execute malicious code after a delay or on a trigger condition.
- **Steganography**: Malicious code hidden in image or data files bundled with the skill.

### Detection Boundaries

- **Regex-based**: Feature extraction uses regex patterns, which cannot understand program semantics. Obfuscation that breaks regex matching will evade detection.
- **Static analysis only**: SkillGuard does not execute skills. Behavior that only manifests at runtime is not detected.
- **Fixed feature set**: The classifier uses 22 features. Attack vectors outside these dimensions produce no signal.
- **Model accuracy**: ~90% decision accuracy on the validation set. Approximately 1 in 10 skills may receive an incorrect decision.

### What SkillGuard Does Detect

- Reverse shells (`nc -e`, `/dev/tcp/`, `socat`, etc.)
- Data exfiltration (HTTP POST to external URLs, `curl --data`)
- Obfuscated code (`eval()`, `atob()`, `base64.b64decode()`, `new Function()`)
- Privilege escalation (`sudo`, `chmod 777`, `chown root`)
- Persistence mechanisms (`crontab`, `systemctl enable`, `.bashrc` modification)
- LLM-targeted social engineering (instructions telling the AI to leak secrets)
- PowerShell execution (`Invoke-Expression`, `-EncodedCommand`, `Invoke-WebRequest`)
- Rust/Go shell execution (`std::process::Command`, `exec.Command`)
- DOM manipulation payloads (`document.write`, `innerHTML`)
- Password-protected archives bundled with skills
- Low-reputation author signals (new accounts, zero downloads)

## Deployment Recommendations

1. **Run behind a reverse proxy** (nginx, Caddy) with TLS termination.
2. **Enable API key authentication** via `SKILLGUARD_API_KEY` for production deployments.
3. **Set rate limits** to prevent abuse (default: 60 req/min per IP).
4. **Monitor access logs** (`skillguard-access.jsonl`) for unusual patterns.
5. **Re-scan skills on updates** — a skill that passes today could be modified later.
6. **Use SkillGuard as one layer** in a defense-in-depth strategy, not as a sole gate.
