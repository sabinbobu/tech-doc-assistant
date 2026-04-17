# MCP Server — Technical Documentation Assistant

This MCP server exposes your RAG pipeline as three tools any MCP-compatible
client (Claude Desktop, VS Code, other agents) can call.

## Tools

| Tool | Purpose |
|---|---|
| `docs_status` | Check if documents are indexed and ready |
| `docs_search` | Retrieve raw relevant passages (no LLM call) |
| `docs_ask` | Full RAG Q&A with cited answer |

## Connecting to Claude Desktop

### Step 1: Find your Claude Desktop config file

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Step 2: Add the MCP server entry

Open the config file and add this (replace the path with your actual project path):

```json
{
  "mcpServers": {
    "tech-docs": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/ABSOLUTE/PATH/TO/tech-doc-assistant",
        "python",
        "src/mcp_server/server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

**Important:** Use the absolute path to your project directory.
On macOS: `/Users/YOUR_USERNAME/projects/tech-doc-assistant`

### Step 3: Restart Claude Desktop

Fully quit and reopen Claude Desktop.
You should see a 🔧 icon in the chat interface indicating MCP tools are available.

### Step 4: Test it

Type in Claude Desktop:
```
Check if my documentation assistant is ready, then ask it what a deviation is in MISRA compliance.
```

Claude will automatically call `docs_status` then `docs_ask` and return
a cited answer from your local MISRA document.

## Adding the MCP dependency

Add to your `pyproject.toml` dependencies:
```
"mcp[cli]>=1.0.0",
```

Then:
```bash
uv pip install -e ".[dev]"
```

## Testing the server manually

```bash
# Verify it starts without errors
uv run python src/mcp_server/server.py
```

You should see:
```
MCP server starting — connecting to ChromaDB...
ChromaDB ready — XXXX chunks indexed
```

Press Ctrl+C to stop.
