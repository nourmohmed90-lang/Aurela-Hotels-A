"""
Temporary MCP server used for local integration testing.
Not part of the production server.
"""

from mcp.server.fastmcp import FastMCP


app = FastMCP("Test Server")

@app.add_tool
def hello(name: str) -> str:
    return f"Hello {name}"

if __name__ == "__main__":
    app.run()