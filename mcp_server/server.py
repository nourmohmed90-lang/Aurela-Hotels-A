import os
import sqlite3
from . import mcp

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB = os.path.join(BASE_DIR, "database", "hotel.db")

# DATABASE
          
def get_db():
    return sqlite3.connect(DB)

if __name__ == "__main__":
    import os
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        print("Starting Aurelia Hotel MCP Server...")
        mcp.run(transport="streamable-http")