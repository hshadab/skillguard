use clap::{Parser, Subcommand};
use eyre::{Result, WrapErr};
use std::fs;
use std::path::PathBuf;
use tracing::{error, info};

use skillguard::scores::ClassScores;
use skillguard::skill::{derive_decision, skill_from_skill_md, Skill, SkillFeatures, VTReport};

#[derive(Parser)]
#[command(
    name = "skillguard",
    about = "Verifiable AI safety classifier powered by Jolt Atlas ZKML with x402 agentic commerce."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP classifier service
    Serve {
        /// Address to bind to (use 0.0.0.0:8080 to expose externally)
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Rate limit in requests per minute per IP (0 = no limit)
        #[arg(long, default_value_t = 60)]
        rate_limit: u32,

        /// Path for JSONL access log
        #[arg(long, default_value = "skillguard-access.jsonl")]
        access_log: String,

        /// Maximum access log file size in bytes before rotation (0 = no limit)
        #[arg(long, default_value_t = 50_000_000)]
        max_log_bytes: u64,

        /// Cache directory for proofs and metrics persistence
        #[arg(long, default_value = "/var/data/skillguard-cache")]
        cache_dir: String,
    },

    /// Check a skill for safety issues (local CLI)
    Check {
        /// Path to a SKILL.md file or JSON skill definition
        #[arg(long)]
        input: PathBuf,

        /// Path to optional VirusTotal report JSON
        #[arg(long)]
        vt_report: Option<PathBuf>,

        /// Output format: json or summary
        #[arg(long, default_value = "summary")]
        format: String,

        /// Generate a ZKML proof for the classification
        #[arg(long)]
        prove: bool,
    },

    /// Crawl the awesome-openclaw-skills list and fetch SKILL.md files
    #[cfg(feature = "crawler")]
    Crawl {
        /// URL of the awesome-openclaw-skills README (raw markdown)
        #[arg(long)]
        awesome_url: Option<String>,

        /// Maximum number of skills to crawl (0 = no limit)
        #[arg(long, default_value_t = 0)]
        limit: usize,

        /// Maximum concurrent fetches
        #[arg(long, default_value_t = 5)]
        concurrency: usize,

        /// Delay between fetches in milliseconds
        #[arg(long, default_value_t = 200)]
        delay_ms: u64,

        /// Output directory for crawled skills
        #[arg(long, default_value = "crawled-skills")]
        output_dir: PathBuf,
    },

    /// Extract the 35-dim feature vector from a SKILL.md (read from stdin), output as JSON
    ExtractFeatures,

    /// Batch scan crawled skills and produce reports
    #[cfg(feature = "crawler")]
    Scan {
        /// Input directory containing crawled SKILL.md files
        #[arg(long)]
        input_dir: Option<PathBuf>,

        /// Fetch and scan directly from the awesome list (live mode)
        #[arg(long)]
        from_awesome: bool,

        /// Output format: json, csv, or summary
        #[arg(long, default_value = "summary")]
        format: String,

        /// Output file (defaults to stdout)
        #[arg(long)]
        output: Option<PathBuf>,

        /// Filter results by classification (comma-separated, e.g. DANGEROUS,CAUTION)
        #[arg(long)]
        filter: Option<String>,

        /// Maximum concurrent classifications
        #[arg(long, default_value_t = 5)]
        concurrency: usize,

        /// Maximum skills to scan from awesome list (live mode only, 0 = no limit)
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
}

/// Environment-variable-derived configuration for the server.
struct EnvConfig {
    api_key: Option<String>,
    pay_to: Option<String>,
    facilitator_url: String,
    external_url: Option<String>,
    price_usdc_micro: u64,
}

fn cmd_serve(
    bind: String,
    rate_limit: u32,
    access_log: String,
    max_log_bytes: u64,
    cache_dir: String,
    env: EnvConfig,
) -> Result<()> {
    use skillguard::server::{run_server, ServerConfig};

    let bind_addr = bind
        .parse()
        .wrap_err_with(|| format!("Invalid bind address: {}", bind))?;

    let config = ServerConfig {
        bind_addr,
        rate_limit_rpm: rate_limit,
        access_log_path: access_log,
        max_access_log_bytes: max_log_bytes,
        cache_dir,
        api_key: env.api_key.clone(),
        pay_to: env.pay_to.clone(),
        facilitator_url: env.facilitator_url,
        external_url: env.external_url,
        price_usdc_micro: env.price_usdc_micro,
    };

    info!("Starting SkillGuard ZKML classifier service");
    info!(model = "skill-safety", params = 4460, proving = "Jolt/Dory");
    info!(model_hash = %skillguard::model_hash());
    if env.api_key.is_some() {
        info!("API key authentication enabled on /api/v1/* endpoints");
    }
    if let Some(ref addr) = env.pay_to {
        info!(pay_to = %addr, "x402 payment enabled ($0.001 USDC per request on Base, proofs included)");
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server(config))?;

    Ok(())
}

#[cfg(feature = "crawler")]
fn cmd_crawl(
    awesome_url: Option<String>,
    limit: usize,
    concurrency: usize,
    delay_ms: u64,
    output_dir: PathBuf,
) -> Result<()> {
    use skillguard::crawler::{run_crawl, CrawlConfig};

    let config = CrawlConfig {
        awesome_list_url: awesome_url.unwrap_or_else(|| CrawlConfig::default().awesome_list_url),
        github_token: std::env::var("GITHUB_TOKEN").ok(),
        concurrency,
        delay_ms,
        limit,
        output_dir,
    };

    info!("Starting crawl of awesome-openclaw-skills list");

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(run_crawl(&config))?;

    println!("Crawl complete:");
    println!("  Fetched: {}", result.successes.len());
    println!("  Failed:  {}", result.failures.len());

    if !result.failures.is_empty() {
        println!("\nFailed entries:");
        for f in &result.failures {
            println!("  {} ({}): {}", f.entry.name, f.entry.author, f.error);
        }
    }

    Ok(())
}

#[cfg(feature = "crawler")]
fn cmd_scan(
    input_dir: Option<PathBuf>,
    from_awesome: bool,
    format: String,
    output: Option<PathBuf>,
    filter: Option<String>,
    concurrency: usize,
    limit: usize,
) -> Result<()> {
    use skillguard::batch::{run_batch_scan, BatchConfig, ScanMode};

    let mode = if from_awesome {
        ScanMode::Live {
            github_token: std::env::var("GITHUB_TOKEN").ok(),
            limit,
        }
    } else {
        ScanMode::Directory {
            path: input_dir.unwrap_or_else(|| PathBuf::from("crawled-skills")),
        }
    };

    let filter_classes: Vec<String> = filter
        .map(|f| f.split(',').map(|s| s.trim().to_uppercase()).collect())
        .unwrap_or_default();

    let config = BatchConfig {
        mode,
        format,
        output,
        filter: filter_classes,
        concurrency,
    };

    info!("Starting batch scan");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_batch_scan(&config))?;

    Ok(())
}

fn cmd_extract_features() -> Result<()> {
    use std::io::Read as _;

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .wrap_err("Failed to read SKILL.md from stdin")?;

    // Build a minimal Skill from the raw markdown
    let skill = Skill {
        name: "stdin".into(),
        version: "1.0.0".into(),
        author: "unknown".into(),
        description: String::new(),
        skill_md: input,
        scripts: Vec::new(),
        metadata: skillguard::skill::SkillMetadata::default(),
        files: Vec::new(),
    };

    let features = SkillFeatures::extract(&skill, None);
    let vec = features.to_normalized_vec();
    println!("{}", serde_json::to_string(&vec)?);

    Ok(())
}

fn cmd_check(
    input: PathBuf,
    vt_report_path: Option<PathBuf>,
    format: String,
    prove: bool,
) -> Result<i32> {
    // Load skill from input
    let skill: Skill = if input.extension().map(|e| e == "json").unwrap_or(false) {
        let content = fs::read_to_string(&input)?;
        serde_json::from_str(&content)?
    } else {
        skill_from_skill_md(&input)?
    };

    // Load optional VT report
    let vt_report: Option<VTReport> = if let Some(vt_path) = vt_report_path {
        let content = fs::read_to_string(&vt_path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Extract features
    let features = SkillFeatures::extract(&skill, vt_report.as_ref());
    let feature_vec = features.to_normalized_vec();
    tracing::debug!(?features, "extracted skill features");
    tracing::debug!(?feature_vec, "normalized feature vector");
    let model_hash = skillguard::model_hash();

    let (classification, raw_scores, confidence, proof_bundle) = if prove {
        let prover = skillguard::prover::ProverState::initialize()?;
        let (cls, scores, conf, bundle) = skillguard::classify_with_proof(&prover, &feature_vec)?;
        (cls, scores, conf, Some(bundle))
    } else {
        let (cls, scores, conf) = skillguard::classify(&feature_vec)?;
        (cls, scores, conf, None)
    };

    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    match format.as_str() {
        "json" => {
            let mut result = serde_json::json!({
                "success": true,
                "evaluation": {
                    "skill_name": skill.name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "scores": scores,
                    "reasoning": reasoning,
                    "raw_logits": raw_scores,
                    "entropy": scores.entropy(),
                },
                "model_hash": model_hash,
            });
            if let Some(ref bundle) = proof_bundle {
                result["proof"] = serde_json::to_value(bundle)?;
            }
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            println!("Skill Safety Scan Results");
            println!("========================");
            println!("Skill: {} v{}", skill.name, skill.version);
            println!();
            println!("Classification: {}", classification.as_str());
            println!("Decision:       {}", decision.as_str());
            println!("Confidence:     {:.1}%", confidence * 100.0);
            println!("Reasoning:      {}", reasoning);
            println!();
            println!("Scores:");
            println!("  SAFE:       {:.1}%", scores.safe * 100.0);
            println!("  CAUTION:    {:.1}%", scores.caution * 100.0);
            println!("  DANGEROUS:  {:.1}%", scores.dangerous * 100.0);
            println!();
            println!(
                "Raw logits:  [{}, {}, {}]",
                raw_scores[0], raw_scores[1], raw_scores[2]
            );
            println!("Entropy:     {:.4}", scores.entropy());
            println!();
            println!("Model Hash: {}", model_hash);
            if let Some(ref bundle) = proof_bundle {
                println!();
                println!("ZK Proof:");
                println!("  Size:        {} bytes", bundle.proof_size_bytes);
                println!("  Proving time: {} ms", bundle.proving_time_ms);
                println!(
                    "  Proof (b64): {}...{}",
                    &bundle.proof_b64[..40.min(bundle.proof_b64.len())],
                    if bundle.proof_b64.len() > 40 {
                        &bundle.proof_b64[bundle.proof_b64.len().saturating_sub(20)..]
                    } else {
                        ""
                    }
                );
            }
        }
    }

    if classification.is_deny() {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn main() {
    // Initialize structured logging. Reads RUST_LOG env for filtering
    // (e.g., RUST_LOG=skillguard=debug). Defaults to info level.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let api_key = std::env::var("SKILLGUARD_API_KEY").ok();
    let pay_to = std::env::var("SKILLGUARD_PAY_TO").ok();
    let facilitator_url = std::env::var("SKILLGUARD_FACILITATOR_URL")
        .unwrap_or_else(|_| "https://pay.openfacilitator.io".to_string());
    let external_url = std::env::var("SKILLGUARD_EXTERNAL_URL").ok();
    let price_usdc_micro: u64 = match std::env::var("SKILLGUARD_PRICE_USDC_MICRO") {
        Ok(v) => v.parse().unwrap_or_else(|e| {
            tracing::warn!(
                var = "SKILLGUARD_PRICE_USDC_MICRO",
                value = %v,
                error = %e,
                "failed to parse, using default 1000"
            );
            1000
        }),
        Err(_) => 1000,
    };

    let env_config = EnvConfig {
        api_key,
        pay_to,
        facilitator_url,
        external_url,
        price_usdc_micro,
    };

    let result = match cli.command {
        Commands::Serve {
            bind,
            rate_limit,
            access_log,
            max_log_bytes,
            cache_dir,
        } => cmd_serve(
            bind,
            rate_limit,
            access_log,
            max_log_bytes,
            cache_dir,
            env_config,
        ),
        Commands::ExtractFeatures => cmd_extract_features(),
        Commands::Check {
            input,
            vt_report,
            format,
            prove,
        } => match cmd_check(input, vt_report, format, prove) {
            Ok(code) => {
                if code != 0 {
                    std::process::exit(code);
                }
                Ok(())
            }
            Err(e) => Err(e),
        },
        #[cfg(feature = "crawler")]
        Commands::Crawl {
            awesome_url,
            limit,
            concurrency,
            delay_ms,
            output_dir,
        } => cmd_crawl(awesome_url, limit, concurrency, delay_ms, output_dir),
        #[cfg(feature = "crawler")]
        Commands::Scan {
            input_dir,
            from_awesome,
            format,
            output,
            filter,
            concurrency,
            limit,
        } => cmd_scan(
            input_dir,
            from_awesome,
            format,
            output,
            filter,
            concurrency,
            limit,
        ),
    };

    if let Err(e) = result {
        error!("{e:?}");
        std::process::exit(1);
    }
}
