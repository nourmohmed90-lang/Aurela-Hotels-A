from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Aurelia Hotel Recovery Server",
    json_response=True,
)


from . import tools
from . import resources
from . import prompts
from . import notifications