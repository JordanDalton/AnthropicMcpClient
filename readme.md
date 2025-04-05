# Anthropic MCP Client

A command-line interface (CLI) tool that allows users to interact with the Anthropic MCP (Model Context Protocol) by connecting to multiple servers and executing tools.

## Installation

```bash
npm install
```

```bash
echo "ANTHROPIC_API_KEY=<your key here>" > .env
```

Create a servers.json file:

```json
{
  "mcpServers": {
    "filesystem-mcp": {
      "command": "npx",
      "args": [
        "@shtse8/filesystem-mcp"
      ],
      "name": "Filesystem (npx)"
    },
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
  }
}
```

## Usage

```
npm run build
npm start
```