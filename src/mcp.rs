//! MCP (Model Context Protocol) server for SkillGuard.
//!
//! Exposes the `skillguard_evaluate` tool over JSON-RPC via stdio.
//! Reads JSON-RPC requests from stdin (one per line), writes responses to stdout.
//!
//! Protocol flow:
//! 1. Client sends `initialize` -> server responds with capabilities
//! 2. Client sends `initialized` notification (no response)
//! 3. Client sends `tools/list` -> server responds with tool definitions
//! 4. Client sends `tools/call` -> server classifies and responds with result

use std::io::{BufRead, BufReader, Write};

use eyre::Result;
use serde_json::{json, Value};

use crate::scores::ClassScores;
use crate::skill::{derive_decision, Skill, SkillFeatures, SkillMetadata};

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

/// Build a JSON-RPC 2.0 success response.
fn jsonrpc_ok(id: &Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

/// Build a JSON-RPC 2.0 error response.
fn jsonrpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        },
    })
}

// Standard JSON-RPC error codes
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// ---------------------------------------------------------------------------
// MCP protocol constants
// ---------------------------------------------------------------------------

const SERVER_NAME: &str = "skillguard";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: &str = "2024-11-05";

// ---------------------------------------------------------------------------
// Tool schema
// ---------------------------------------------------------------------------

/// Return the JSON Schema for the `skillguard_evaluate` tool's input.
fn tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to evaluate. A minimal Skill struct is built from this name."
            },
            "skill_md": {
                "type": "string",
                "description": "Full SKILL.md markdown content to evaluate. Features are extracted from this content."
            }
        },
        "additionalProperties": false
    })
}

/// Return the tool definition for `tools/list`.
fn tool_definition() -> Value {
    json!({
        "name": "skillguard_evaluate",
        "description": "Evaluate a skill for safety. Provide either `skill_name` (minimal evaluation) or `skill_md` (full markdown content for richer feature extraction). Returns classification, decision, confidence, scores, and reasoning.",
        "inputSchema": tool_input_schema()
    })
}

// ---------------------------------------------------------------------------
// Classification logic
// ---------------------------------------------------------------------------

/// Build a `Skill` from a name string (minimal, no content analysis).
fn skill_from_name(name: &str) -> Skill {
    // Use neutral metadata defaults so the model relies on the (empty) content
    // features rather than penalising missing metadata.
    let neutral_author_date = (chrono::Utc::now() - chrono::Duration::days(180))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    Skill {
        name: name.to_string(),
        version: "1.0.0".into(),
        author: "unknown".into(),
        description: String::new(),
        skill_md: String::new(),
        scripts: Vec::new(),
        metadata: SkillMetadata {
            stars: 50,
            downloads: 500,
            author_account_created: neutral_author_date,
            author_total_skills: 5,
            ..Default::default()
        },
        files: Vec::new(),
    }
}

/// Build a `Skill` from raw SKILL.md content.
fn skill_from_markdown(skill_md: &str) -> Skill {
    // Parse optional YAML frontmatter for name/description/author,
    // mirroring the approach in `skill::skill_from_skill_md` but
    // operating on a string rather than a file path.
    let fm = parse_frontmatter(skill_md);

    let name = fm.name.unwrap_or_else(|| "mcp-input".into());
    let description = fm.description.unwrap_or_default();
    let author = fm.author.unwrap_or_else(|| "unknown".into());

    // Detect file references in markdown for extension-diversity features.
    let file_ref_re =
        regex::Regex::new(r"\b[\w./-]+\.(sh|py|js|ts|rb|lua|php|ps1|bat|exe|zip|tar|gz)\b")
            .expect("valid regex");
    let files: Vec<String> = file_ref_re
        .find_iter(skill_md)
        .map(|m| m.as_str().to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let neutral_author_date = (chrono::Utc::now() - chrono::Duration::days(180))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    Skill {
        name,
        version: "1.0.0".into(),
        author,
        description,
        skill_md: skill_md.to_string(),
        scripts: Vec::new(),
        metadata: SkillMetadata {
            stars: 50,
            downloads: 500,
            author_account_created: neutral_author_date,
            author_total_skills: 5,
            ..Default::default()
        },
        files,
    }
}

/// Minimal frontmatter parser (mirrors `skill::parse_yaml_frontmatter`).
struct FrontmatterFields {
    name: Option<String>,
    description: Option<String>,
    author: Option<String>,
}

fn parse_frontmatter(content: &str) -> FrontmatterFields {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return FrontmatterFields {
            name: None,
            description: None,
            author: None,
        };
    }

    let after_open = &trimmed[3..];
    let close_pos = after_open.find("\n---");
    let frontmatter = match close_pos {
        Some(pos) => &after_open[..pos],
        None => {
            return FrontmatterFields {
                name: None,
                description: None,
                author: None,
            }
        }
    };

    let mut name = None;
    let mut description = None;
    let mut author = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        let extract =
            |val: &str| -> String { val.trim().trim_matches('"').trim_matches('\'').to_string() };
        if let Some(val) = line.strip_prefix("name:") {
            name = Some(extract(val));
        } else if let Some(val) = line.strip_prefix("description:") {
            description = Some(extract(val));
        } else if let Some(val) = line.strip_prefix("author:") {
            author = Some(extract(val));
        }
    }

    FrontmatterFields {
        name,
        description,
        author,
    }
}

/// Run classification on a `Skill` and return a JSON result object.
fn evaluate_skill(skill: &Skill) -> Result<Value> {
    let features = SkillFeatures::extract(skill, None);
    let feature_vec = features.to_normalized_vec();

    let (classification, raw_scores, confidence) = crate::classify(&feature_vec)?;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    Ok(json!({
        "classification": classification.as_str(),
        "decision": decision.as_str(),
        "confidence": confidence,
        "scores": {
            "SAFE": scores.safe,
            "CAUTION": scores.caution,
            "DANGEROUS": scores.dangerous,
        },
        "reasoning": reasoning,
    }))
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

/// Handle a single parsed JSON-RPC request and return a response (or None for notifications).
fn handle_request(request: &Value) -> Option<Value> {
    let id = request.get("id");
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");

    match method {
        // -----------------------------------------------------------------
        // initialize
        // -----------------------------------------------------------------
        "initialize" => {
            let id = id?;
            Some(jsonrpc_ok(
                id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    }
                }),
            ))
        }

        // -----------------------------------------------------------------
        // initialized (notification — no response)
        // -----------------------------------------------------------------
        "notifications/initialized" => None,

        // -----------------------------------------------------------------
        // tools/list
        // -----------------------------------------------------------------
        "tools/list" => {
            let id = id?;
            Some(jsonrpc_ok(
                id,
                json!({
                    "tools": [tool_definition()]
                }),
            ))
        }

        // -----------------------------------------------------------------
        // tools/call
        // -----------------------------------------------------------------
        "tools/call" => {
            let id = id?;
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let tool_name = params.get("name").and_then(Value::as_str).unwrap_or("");

            if tool_name != "skillguard_evaluate" {
                return Some(jsonrpc_error(
                    id,
                    METHOD_NOT_FOUND,
                    &format!("Unknown tool: {}", tool_name),
                ));
            }

            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            let skill_name = arguments.get("skill_name").and_then(Value::as_str);
            let skill_md = arguments.get("skill_md").and_then(Value::as_str);

            let skill = match (skill_name, skill_md) {
                (_, Some(md)) => skill_from_markdown(md), // prefer skill_md if both provided
                (Some(name), None) => skill_from_name(name),
                (None, None) => {
                    return Some(jsonrpc_error(
                        id,
                        INVALID_PARAMS,
                        "Either `skill_name` or `skill_md` must be provided",
                    ));
                }
            };

            match evaluate_skill(&skill) {
                Ok(result) => Some(jsonrpc_ok(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&result)
                                .unwrap_or_else(|_| result.to_string()),
                        }]
                    }),
                )),
                Err(e) => Some(jsonrpc_error(
                    id,
                    INTERNAL_ERROR,
                    &format!("Classification failed: {}", e),
                )),
            }
        }

        // -----------------------------------------------------------------
        // Unknown method
        // -----------------------------------------------------------------
        _ => {
            // Notifications (no id) get no response.
            let id = id?;
            Some(jsonrpc_error(
                id,
                METHOD_NOT_FOUND,
                &format!("Unknown method: {}", method),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the MCP server, reading JSON-RPC messages from stdin and writing
/// responses to stdout. One message per line (newline-delimited JSON).
///
/// The server runs until stdin is closed (EOF).
pub fn run_mcp_server() -> Result<()> {
    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout().lock();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                let err = jsonrpc_error(&Value::Null, PARSE_ERROR, "Parse error");
                serde_json::to_writer(&mut stdout, &err)?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                continue;
            }
        };

        // Validate basic JSON-RPC structure
        if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            let id = request.get("id").cloned().unwrap_or(Value::Null);
            let err = jsonrpc_error(&id, INVALID_REQUEST, "Invalid JSON-RPC version");
            serde_json::to_writer(&mut stdout, &err)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
            continue;
        }

        if let Some(response) = handle_request(&request) {
            serde_json::to_writer(&mut stdout, &response)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_response() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0.1" }
            }
        });

        let response = handle_request(&request).expect("initialize should return a response");
        assert_eq!(response["id"], 1);
        assert!(response.get("error").is_none());
        assert_eq!(
            response["result"]["protocolVersion"],
            PROTOCOL_VERSION
        );
        assert_eq!(response["result"]["serverInfo"]["name"], SERVER_NAME);
        assert!(response["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn test_initialized_notification_no_response() {
        let request = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });

        let response = handle_request(&request);
        assert!(response.is_none(), "notifications should not produce a response");
    }

    #[test]
    fn test_tools_list() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });

        let response = handle_request(&request).expect("tools/list should return a response");
        assert_eq!(response["id"], 2);
        assert!(response.get("error").is_none());

        let tools = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "skillguard_evaluate");
        assert!(tools[0]["inputSchema"].is_object());
    }

    #[test]
    fn test_tools_call_missing_params() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "skillguard_evaluate",
                "arguments": {}
            }
        });

        let response = handle_request(&request).expect("should return error response");
        assert!(response.get("error").is_some());
        assert_eq!(response["error"]["code"], INVALID_PARAMS);
    }

    #[test]
    fn test_tools_call_unknown_tool() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        });

        let response = handle_request(&request).expect("should return error response");
        assert!(response.get("error").is_some());
        assert_eq!(response["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn test_unknown_method() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method"
        });

        let response = handle_request(&request).expect("should return error response");
        assert!(response.get("error").is_some());
        assert_eq!(response["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn test_tools_call_with_skill_name() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "skillguard_evaluate",
                "arguments": {
                    "skill_name": "hello-world"
                }
            }
        });

        let response = handle_request(&request).expect("should return a response");
        assert!(response.get("error").is_none(), "unexpected error: {:?}", response.get("error"));
        let content = response["result"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");

        // Parse the text content back as JSON and verify fields
        let text = content[0]["text"].as_str().unwrap();
        let result: Value = serde_json::from_str(text).unwrap();
        assert!(result.get("classification").is_some());
        assert!(result.get("decision").is_some());
        assert!(result.get("confidence").is_some());
        assert!(result.get("scores").is_some());
        assert!(result.get("reasoning").is_some());
    }

    #[test]
    fn test_tools_call_with_skill_md() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "skillguard_evaluate",
                "arguments": {
                    "skill_md": "---\nname: test-skill\n---\n# Test Skill\n\nA simple test skill that greets users."
                }
            }
        });

        let response = handle_request(&request).expect("should return a response");
        assert!(response.get("error").is_none(), "unexpected error: {:?}", response.get("error"));
        let content = response["result"]["content"].as_array().unwrap();
        let text = content[0]["text"].as_str().unwrap();
        let result: Value = serde_json::from_str(text).unwrap();
        assert!(result.get("classification").is_some());
    }

    #[test]
    fn test_frontmatter_parsing() {
        let md = "---\nname: my-skill\nauthor: alice\ndescription: Does things\n---\n# Content";
        let fm = parse_frontmatter(md);
        assert_eq!(fm.name.as_deref(), Some("my-skill"));
        assert_eq!(fm.author.as_deref(), Some("alice"));
        assert_eq!(fm.description.as_deref(), Some("Does things"));
    }

    #[test]
    fn test_frontmatter_missing() {
        let md = "# No Frontmatter\n\nJust markdown.";
        let fm = parse_frontmatter(md);
        assert!(fm.name.is_none());
        assert!(fm.author.is_none());
    }

    #[test]
    fn test_skill_from_name_defaults() {
        let skill = skill_from_name("test-skill");
        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.metadata.stars, 50);
        assert_eq!(skill.metadata.downloads, 500);
        assert!(skill.skill_md.is_empty());
    }

    #[test]
    fn test_skill_from_markdown_extracts_files() {
        let md = "# Skill\n\nRun `payload.sh` and `helper.py` to get started.";
        let skill = skill_from_markdown(md);
        assert!(skill.files.contains(&"payload.sh".to_string()));
        assert!(skill.files.contains(&"helper.py".to_string()));
    }
}
