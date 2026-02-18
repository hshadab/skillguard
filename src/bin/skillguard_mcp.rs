fn main() {
    if let Err(e) = skillguard::mcp::run_mcp_server() {
        eprintln!("MCP server error: {}", e);
        std::process::exit(1);
    }
}
