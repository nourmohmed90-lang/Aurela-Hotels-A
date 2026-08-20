import os
import sys
import sqlite3
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Link project path to access state_graph and mcp_server
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from mcp_server.server import list_tool_statuses, set_tool_status
from state_graph.graph import vip_booking_graph

app = FastAPI(title="Aurelia Hotels - Admin Platform")

DB_PATH = os.path.join(BASE_DIR, "database", "hotel.db")
DOCS_DIR = os.path.join(BASE_DIR, "documents")
os.makedirs(DOCS_DIR, exist_ok=True)

# Helper DB connection
def get_db():
    return sqlite3.connect(DB_PATH)

# ==========================================
# 1. MCP TOOL MANAGEMENT ENDPOINTS
# ==========================================
@app.get("/api/admin/tools")
def get_mcp_tools():
    """Returns all available MCP tools and their active/disabled status."""
    return list_tool_statuses()

class ToolToggleRequest(BaseModel):
    tool_name: str
    enabled: bool

@app.post("/api/admin/tools/toggle")
def toggle_mcp_tool(req: ToolToggleRequest):
    """Enables or disables an MCP tool in real-time."""
    result = set_tool_status(req.tool_name, req.enabled)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

# ==========================================
# 2. HITL & FAILURE TICKET DASHBOARD
# ==========================================
@app.get("/api/admin/pending-tasks")
def get_pending_tasks():
    """Fetches all pending HITL approvals and open failure tickets from checkpoints."""
    conn = get_db()
    cursor = conn.cursor()
    # Query checkpoints stored by LangGraph SqliteSaver
    cursor.execute("SELECT thread_id, checkpoint FROM checkpoints")
    rows = cursor.fetchall()
    conn.close()

    tasks = []
    # Parse checkpoints for active HITL pauses or tickets
    for thread_id, _ in set(rows):
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = vip_booking_graph.get_state(config)
            if state and state.values:
                val = state.values
                status = val.get("status")
                if status in ["PAUSED_HITL", "FAILED_TICKET"]:
                    tasks.append({
                        "thread_id": thread_id,
                        "booking_id": val.get("booking_id"),
                        "status": status,
                        "total_price": val.get("total_price", 0.0),
                        "items": val.get("itinerary_items", []),
                        "error_message": val.get("error_message")
                    })
        except Exception:
            continue
    return tasks

class TaskActionRequest(BaseModel):
    thread_id: str
    action: str  # "approve" or "resolve"

@app.post("/api/admin/tasks/action")
def resolve_task(req: TaskActionRequest):
    """Approves a HITL task or resolves a ticket, then resumes state graph execution."""
    config = {"configurable": {"thread_id": req.thread_id}}
    state_obj = vip_booking_graph.get_state(config)
    
    if not state_obj or not state_obj.values:
        raise HTTPException(status_code=404, detail="Task thread not found.")

    current_state = dict(state_obj.values)

    if req.action == "approve":
        current_state["is_approved"] = True
        current_state["status"] = "APPROVED"
    elif req.action == "resolve":
        current_state["status"] = "IN_PROGRESS"
        current_state["error_message"] = None

    # Resume state graph from its exact checkpoint
    resumed_state = vip_booking_graph.invoke(current_state, config=config)
    return {"success": True, "new_status": resumed_state.get("status")}

# ==========================================
# 3. RAG DOCUMENT MANAGEMENT
# ==========================================
@app.get("/api/admin/documents")
def list_documents():
    """Lists all documents stored in the RAG corpus."""
    files = os.listdir(DOCS_DIR)
    return {"documents": files}

@app.post("/api/admin/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a new document to the RAG repository."""
    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"success": True, "filename": file.filename}

@app.delete("/api/admin/documents/{filename}")
def delete_document(filename: str):
    """Deletes a document from the RAG store."""
    file_path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"success": True}
    raise HTTPException(status_code=404, detail="Document not found.")

# Single-page Admin Dashboard HTML Interface
@app.get("/", response_class=HTMLResponse)
def admin_dashboard():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        return f.read()