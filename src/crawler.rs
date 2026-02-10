//! Crawler for the awesome-openclaw-skills list.
//!
//! Parses the awesome-openclaw-skills README to extract skill entries,
//! converts GitHub tree URLs to raw content URLs, and fetches SKILL.md files
//! with concurrency control and optional authentication.

use eyre::{Result, WrapErr};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default URL for the awesome-openclaw-skills README.
const DEFAULT_AWESOME_LIST_URL: &str =
    "https://raw.githubusercontent.com/VoltAgent/awesome-openclaw-skills/main/README.md";

/// Configuration for the crawler.
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    /// URL of the awesome-openclaw-skills README (raw markdown).
    pub awesome_list_url: String,
    /// Optional GitHub personal access token for higher rate limits.
    pub github_token: Option<String>,
    /// Maximum number of concurrent fetches.
    pub concurrency: usize,
    /// Delay between fetches in milliseconds.
    pub delay_ms: u64,
    /// Maximum number of entries to process (0 = no limit).
    pub limit: usize,
    /// Output directory for crawled SKILL.md files.
    pub output_dir: PathBuf,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            awesome_list_url: DEFAULT_AWESOME_LIST_URL.to_string(),
            github_token: None,
            concurrency: 5,
            delay_ms: 200,
            limit: 0,
            output_dir: PathBuf::from("crawled-skills"),
        }
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single entry parsed from the awesome list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwesomeListEntry {
    /// Skill name from the markdown link text.
    pub name: String,
    /// Author extracted from the GitHub URL path.
    pub author: String,
    /// Description text after the link.
    pub description: String,
    /// Original `github.com/.../tree/...` URL.
    pub github_tree_url: String,
    /// Converted `raw.githubusercontent.com/...` URL for fetching content.
    pub raw_content_url: String,
    /// Category from the section header.
    pub category: String,
}

/// Result of a crawl operation.
#[derive(Debug, Serialize)]
pub struct CrawlResult {
    /// Successfully fetched entries.
    pub successes: Vec<CrawledSkill>,
    /// Entries that failed to fetch.
    pub failures: Vec<CrawlFailure>,
}

/// A successfully crawled skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawledSkill {
    /// Parsed entry from the awesome list.
    pub entry: AwesomeListEntry,
    /// Path where the SKILL.md was saved.
    pub saved_path: String,
}

/// A failed crawl attempt.
#[derive(Debug, Clone, Serialize)]
pub struct CrawlFailure {
    /// The entry that failed.
    pub entry: AwesomeListEntry,
    /// Error message.
    pub error: String,
}

/// Manifest written alongside crawled files.
#[derive(Debug, Serialize, Deserialize)]
pub struct CrawlManifest {
    /// When the crawl was performed.
    pub crawled_at: String,
    /// Source URL of the awesome list.
    pub source_url: String,
    /// Total entries found in the list.
    pub total_entries: usize,
    /// Entries successfully fetched.
    pub fetched: usize,
    /// Entries that failed.
    pub failed: usize,
    /// Individual entry details.
    pub entries: Vec<CrawledSkill>,
}

// ---------------------------------------------------------------------------
// Awesome list parser
// ---------------------------------------------------------------------------

/// Regex for matching markdown link entries in the awesome list.
/// Format: `- [skill-name](https://github.com/...) - Description`
static ENTRY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[\s*-]*\[([^\]]+)\]\((https://github\.com/[^)]+)\)\s*[-–—:]?\s*(.*?)$")
        .expect("valid entry regex")
});

/// Regex for matching category headers.
/// Matches both `## Category` and `<summary>Category</summary>` patterns.
static CATEGORY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)(?:^#{2,3}\s+(.+?)$|<summary>\s*(.+?)\s*</summary>)")
        .expect("valid category regex")
});

/// Parse the awesome list markdown and extract skill entries.
pub fn parse_awesome_list(markdown: &str) -> Vec<AwesomeListEntry> {
    let mut entries = Vec::new();
    let mut current_category = String::from("Uncategorized");

    // Process line by line to track categories
    for line in markdown.lines() {
        // Check for category headers
        if let Some(caps) = CATEGORY_RE.captures(line) {
            let cat = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            if !cat.is_empty() {
                current_category = cat;
            }
            continue;
        }

        // Check for entry links
        if let Some(caps) = ENTRY_RE.captures(line) {
            let name = caps.get(1).map(|m| m.as_str().trim()).unwrap_or_default();
            let url = caps.get(2).map(|m| m.as_str().trim()).unwrap_or_default();
            let desc = caps.get(3).map(|m| m.as_str().trim()).unwrap_or_default();

            if name.is_empty() || url.is_empty() {
                continue;
            }

            let author = extract_author_from_url(url);
            let raw_url = github_tree_to_raw(url);

            entries.push(AwesomeListEntry {
                name: name.to_string(),
                author,
                description: desc.to_string(),
                github_tree_url: url.to_string(),
                raw_content_url: raw_url,
                category: current_category.clone(),
            });
        }
    }

    entries
}

/// Extract the author/username from a GitHub URL.
///
/// URL format: `https://github.com/OWNER/REPO/tree/BRANCH/path/to/SKILL.md`
/// Returns the OWNER segment.
pub fn extract_author_from_url(url: &str) -> String {
    let stripped = url
        .trim_start_matches("https://github.com/")
        .trim_start_matches("http://github.com/");
    stripped.split('/').next().unwrap_or("unknown").to_string()
}

/// Convert a GitHub tree URL to a raw.githubusercontent.com URL.
///
/// Input:  `https://github.com/OWNER/REPO/tree/BRANCH/path/to/SKILL.md`
/// Output: `https://raw.githubusercontent.com/OWNER/REPO/BRANCH/path/to/SKILL.md`
pub fn github_tree_to_raw(url: &str) -> String {
    let stripped = url
        .trim_start_matches("https://github.com/")
        .trim_start_matches("http://github.com/");

    // Split into parts: OWNER/REPO/tree/BRANCH/path...
    let parts: Vec<&str> = stripped.splitn(5, '/').collect();
    if parts.len() >= 5 && parts[2] == "tree" {
        // parts: [OWNER, REPO, "tree", BRANCH, PATH]
        format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}",
            parts[0], parts[1], parts[3], parts[4]
        )
    } else if parts.len() >= 4 && parts[2] == "blob" {
        // Handle blob URLs too
        format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}",
            parts[0], parts[1], parts[3], parts[4]
        )
    } else {
        // Fallback: just swap the domain
        format!("https://raw.githubusercontent.com/{}", stripped)
    }
}

// ---------------------------------------------------------------------------
// GitHub fetcher
// ---------------------------------------------------------------------------

/// Fetch a single SKILL.md from a raw GitHub URL.
async fn fetch_skill_md(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<String> {
    let mut request = client.get(url);

    if let Some(t) = token {
        request = request.header("Authorization", format!("Bearer {}", t));
    }
    request = request.header("User-Agent", "skillguard-crawler/0.1");

    let response = request
        .send()
        .await
        .wrap_err_with(|| format!("Failed to fetch {}", url))?;

    let status = response.status();
    if !status.is_success() {
        eyre::bail!("HTTP {} fetching {}", status, url);
    }

    response
        .text()
        .await
        .wrap_err_with(|| format!("Failed to read response body from {}", url))
}

/// Run the full crawl pipeline: parse the awesome list, fetch SKILL.md files,
/// save them to disk, and write a manifest.
pub async fn run_crawl(config: &CrawlConfig) -> Result<CrawlResult> {
    info!(url = %config.awesome_list_url, "Fetching awesome list");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Fetch the awesome list
    let list_md = fetch_skill_md(
        &client,
        &config.awesome_list_url,
        config.github_token.as_deref(),
    )
    .await
    .wrap_err("Failed to fetch awesome list")?;

    // Parse entries
    let mut entries = parse_awesome_list(&list_md);
    info!(total = entries.len(), "Parsed entries from awesome list");

    if config.limit > 0 && entries.len() > config.limit {
        entries.truncate(config.limit);
        info!(limit = config.limit, "Truncated to limit");
    }

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)
        .wrap_err_with(|| format!("Failed to create output dir: {:?}", config.output_dir))?;

    // Fetch SKILL.md files with concurrency control
    let semaphore = std::sync::Arc::new(Semaphore::new(config.concurrency));
    let mut handles = Vec::new();
    let token = config.github_token.clone();

    for entry in entries {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let token = token.clone();
        let output_dir = config.output_dir.clone();
        let delay_ms = config.delay_ms;

        let handle = tokio::spawn(async move {
            // Delay to avoid hammering GitHub
            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }

            let result = fetch_and_save(&client, &entry, token.as_deref(), &output_dir).await;
            drop(permit);
            (entry, result)
        });

        handles.push(handle);
    }

    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for handle in handles {
        match handle.await {
            Ok((entry, Ok(saved_path))) => {
                debug!(name = %entry.name, path = %saved_path, "Fetched skill");
                successes.push(CrawledSkill { entry, saved_path });
            }
            Ok((entry, Err(e))) => {
                warn!(name = %entry.name, error = %e, "Failed to fetch skill");
                failures.push(CrawlFailure {
                    entry,
                    error: e.to_string(),
                });
            }
            Err(e) => {
                warn!(error = %e, "Task join error");
            }
        }
    }

    // Write manifest
    let manifest = CrawlManifest {
        crawled_at: chrono::Utc::now().to_rfc3339(),
        source_url: config.awesome_list_url.clone(),
        total_entries: successes.len() + failures.len(),
        fetched: successes.len(),
        failed: failures.len(),
        entries: successes.clone(),
    };

    let manifest_path = config.output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)
        .wrap_err_with(|| format!("Failed to write manifest to {:?}", manifest_path))?;

    info!(
        fetched = successes.len(),
        failed = failures.len(),
        manifest = %manifest_path.display(),
        "Crawl complete"
    );

    Ok(CrawlResult {
        successes,
        failures,
    })
}

/// Fetch a single skill's SKILL.md and save it to disk.
async fn fetch_and_save(
    client: &reqwest::Client,
    entry: &AwesomeListEntry,
    token: Option<&str>,
    output_dir: &Path,
) -> Result<String> {
    let content = fetch_skill_md(client, &entry.raw_content_url, token).await?;

    // Save to {output_dir}/{author}/{skill_name}/SKILL.md
    let skill_dir = output_dir.join(&entry.author).join(&entry.name);
    std::fs::create_dir_all(&skill_dir)?;

    let skill_path = skill_dir.join("SKILL.md");
    std::fs::write(&skill_path, &content)?;

    Ok(skill_path.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_awesome_list_basic() {
        let md = r#"
# Awesome OpenClaw Skills

## Productivity

- [todo-manager](https://github.com/openclaw/skills/tree/main/skills/alice/todo-manager/SKILL.md) - Manage your TODO lists
- [calendar-sync](https://github.com/openclaw/skills/tree/main/skills/bob/calendar-sync/SKILL.md) - Sync calendars across platforms

## Security

- [vault-reader](https://github.com/openclaw/skills/tree/main/skills/charlie/vault-reader/SKILL.md) - Read secrets from Vault
"#;

        let entries = parse_awesome_list(md);
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].name, "todo-manager");
        assert_eq!(entries[0].author, "openclaw");
        assert_eq!(entries[0].description, "Manage your TODO lists");
        assert_eq!(entries[0].category, "Productivity");

        assert_eq!(entries[1].name, "calendar-sync");
        assert_eq!(entries[1].author, "openclaw");
        assert_eq!(entries[1].category, "Productivity");

        assert_eq!(entries[2].name, "vault-reader");
        assert_eq!(entries[2].category, "Security");
    }

    #[test]
    fn test_parse_awesome_list_with_summary_tags() {
        let md = r#"
<details>
<summary>AI & Machine Learning</summary>

- [ml-trainer](https://github.com/user1/ml-skills/tree/main/trainer/SKILL.md) - Train ML models

</details>

<details>
<summary>DevOps</summary>

- [deploy-bot](https://github.com/user2/deploy/tree/main/bot/SKILL.md) - Automated deployments

</details>
"#;

        let entries = parse_awesome_list(md);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "ml-trainer");
        assert_eq!(entries[0].category, "AI & Machine Learning");
        assert_eq!(entries[1].name, "deploy-bot");
        assert_eq!(entries[1].category, "DevOps");
    }

    #[test]
    fn test_parse_awesome_list_edge_cases() {
        let md = r#"
## Tools

- [simple](https://github.com/org/repo/tree/main/SKILL.md) - Simple tool
- [no-desc](https://github.com/org/repo/tree/main/other/SKILL.md)
- Not a link, just text
- [broken-link](not-a-url) - This should be skipped
"#;

        let entries = parse_awesome_list(md);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "simple");
        assert_eq!(entries[1].name, "no-desc");
        assert_eq!(entries[1].description, "");
    }

    #[test]
    fn test_extract_author_from_url() {
        assert_eq!(
            extract_author_from_url("https://github.com/alice/my-skill/tree/main/skills/SKILL.md"),
            "alice"
        );
        assert_eq!(
            extract_author_from_url(
                "https://github.com/openclaw/skills/tree/main/skills/bob/skill/SKILL.md"
            ),
            "openclaw"
        );
        assert_eq!(extract_author_from_url("https://github.com/solo"), "solo");
    }

    #[test]
    fn test_github_tree_to_raw() {
        assert_eq!(
            github_tree_to_raw(
                "https://github.com/openclaw/skills/tree/main/skills/alice/SKILL.md"
            ),
            "https://raw.githubusercontent.com/openclaw/skills/main/skills/alice/SKILL.md"
        );

        assert_eq!(
            github_tree_to_raw("https://github.com/user/repo/tree/dev/path/to/SKILL.md"),
            "https://raw.githubusercontent.com/user/repo/dev/path/to/SKILL.md"
        );
    }

    #[test]
    fn test_github_blob_to_raw() {
        assert_eq!(
            github_tree_to_raw("https://github.com/user/repo/blob/main/SKILL.md"),
            "https://raw.githubusercontent.com/user/repo/main/SKILL.md"
        );
    }

    #[test]
    fn test_parse_multi_category_markdown() {
        let md = r#"
# Awesome OpenClaw Skills

A curated list of amazing skills.

## Getting Started

Some introductory text, no links here.

## Communication

- [email-sender](https://github.com/comms/tools/tree/main/email/SKILL.md) - Send emails
- [slack-notifier](https://github.com/comms/tools/tree/main/slack/SKILL.md) - Slack notifications

## Data Processing

- [csv-parser](https://github.com/data/tools/tree/main/csv/SKILL.md) - Parse CSV files
- [json-transformer](https://github.com/data/tools/tree/main/json/SKILL.md) - Transform JSON data

## Communication

- [sms-gateway](https://github.com/comms/sms/tree/main/gateway/SKILL.md) - Send SMS messages
"#;

        let entries = parse_awesome_list(md);
        assert_eq!(entries.len(), 5);

        // First two under Communication
        assert_eq!(entries[0].category, "Communication");
        assert_eq!(entries[1].category, "Communication");

        // Next two under Data Processing
        assert_eq!(entries[2].category, "Data Processing");
        assert_eq!(entries[3].category, "Data Processing");

        // Last one back under Communication
        assert_eq!(entries[4].category, "Communication");
    }
}
