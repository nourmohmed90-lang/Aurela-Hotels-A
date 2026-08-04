import os
import sqlite3
from . import mcp

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB = os.path.join(BASE_DIR, "db", "hotel.db")

# DATABASE
          
def get_db():
    return sqlite3.connect(DB)

if __name__ == "__main__":
    print("Starting Aurelia Hotel MCP Server...")
    mcp.run(transport="streamable-http")