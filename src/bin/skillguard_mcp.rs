use std::sync::Arc;

fn main() {
    // Initialise tracing so prover init logs are visible on stderr
    // (MCP uses stdout exclusively for JSON-RPC).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let prover = Arc::new(
        skillguard::prover::ProverState::initialize().expect("Failed to initialise ZK prover"),
    );

    if let Err(e) = skillguard::mcp::run_mcp_server(prover) {
        eprintln!("MCP server error: {}", e);
        std::process::exit(1);
    }
}
