from typing import Dict
from fastmcp import FastMCP

mcp = FastMCP("Status Server")

@mcp.tool
def get_status() -> Dict[str, str | None]:
    """Get server status"""
    try:
        return {
            "result": "The server is runnning.",
            "log": "Successfully executed",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Error: {e}",
        }

@mcp.tool()
def get_uptime() -> Dict[str, str | None]:
    """Get server uptime"""
    try:
        return {
            "result": "24h 30m 15s",
            "log": "Successfully executed",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Error: {e}",
        }

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8182,
    )
