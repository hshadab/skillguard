use clap::{Parser, Subcommand};
use eyre::{Result, WrapErr};
use std::fs;
use std::path::PathBuf;

use skillguard::scores::ClassScores;
use skillguard::skill::{derive_decision, Skill, SkillFeatures, VTReport, skill_from_skill_md};

#[derive(Parser)]
#[command(
    name = "skillguard",
    about = "Standalone skill safety classifier for OpenClaw/ClawHub skills."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP classifier service
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "0.0.0.0:8080")]
        bind: String,

        /// Rate limit in requests per minute per IP (0 = no limit)
        #[arg(long, default_value_t = 60)]
        rate_limit: u32,

        /// Path for JSONL access log
        #[arg(long, default_value = "skillguard-access.jsonl")]
        access_log: String,
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
    },
}

fn cmd_serve(bind: String, rate_limit: u32, access_log: String) -> Result<()> {
    use skillguard::server::{run_server, ServerConfig};

    let bind_addr = bind
        .parse()
        .wrap_err_with(|| format!("Invalid bind address: {}", bind))?;

    let config = ServerConfig {
        bind_addr,
        rate_limit_rpm: rate_limit,
        access_log_path: access_log,
    };

    eprintln!("Starting SkillGuard classifier service...");
    eprintln!("Model: skill-safety (1,924 params)");
    eprintln!("Model hash: {}", skillguard::model_hash());

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server(config))?;

    Ok(())
}

fn cmd_check(input: PathBuf, vt_report_path: Option<PathBuf>, format: String) -> Result<i32> {
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

    // Classify
    let (classification, raw_scores, confidence) = skillguard::classify(&feature_vec)?;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());
    let model_hash = skillguard::model_hash();

    match format.as_str() {
        "json" => {
            let result = serde_json::json!({
                "success": true,
                "evaluation": {
                    "skill_name": skill.name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "scores": scores,
                    "reasoning": reasoning,
                },
                "model_hash": model_hash,
            });
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
            println!("  MALICIOUS:  {:.1}%", scores.malicious * 100.0);
            println!();
            println!("Model Hash: {}", model_hash);
        }
    }

    if classification.is_deny() {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Serve {
            bind,
            rate_limit,
            access_log,
        } => cmd_serve(bind, rate_limit, access_log),
        Commands::Check {
            input,
            vt_report,
            format,
        } => match cmd_check(input, vt_report, format) {
            Ok(code) => {
                if code != 0 {
                    std::process::exit(code);
                }
                Ok(())
            }
            Err(e) => Err(e),
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
