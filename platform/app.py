import os
import sys
import sqlite3
import uuid
from pathlib import Path
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
from state_graph.graph_complaint import complaint_resolution_graph, run_complaint_graph
from state_graph.graph_incident import incident_escalation_graph, run_incident_graph
from state_graph import tickets as ticket_store
from state_graph.tickets import resume_graph_from_ticket
from memory.stores.builder import sync_documents, validate_document
from memory.stores.config import SUPPORTED_EXTENSIONS
from langgraph.types import Command

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
# 2A. HITL APPROVAL QUEUE
# ==========================================
# HITL pauses are expected, in-flow decisions -- they live entirely in the
# LangGraph checkpoint (no separate table), so this reads checkpoint state
# directly. Contrast with 2B below, which reads a real Tickets table.

@app.get("/api/admin/hitl-tasks")
def get_hitl_tasks():
    """Fetches all pending HITL approvals by scanning active checkpoints."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, checkpoint FROM checkpoints")
    rows = cursor.fetchall()
    conn.close()

    tasks = []
    for thread_id, _ in set(rows):
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = vip_booking_graph.get_state(config)
            if state and state.values and state.values.get("status") == "PAUSED_HITL":
                val = state.values
                tasks.append({
                    "thread_id": thread_id,
                    "booking_id": val.get("booking_id"),
                    "total_price": val.get("total_price", 0.0),
                    "items": val.get("itinerary_items", []),
                })
        except Exception:
            continue
    return tasks

class HitlApproveRequest(BaseModel):
    thread_id: str

@app.post("/api/admin/hitl-tasks/approve")
def approve_hitl_task(req: HitlApproveRequest):
    """Approves a HITL-paused run and resumes it from its checkpoint via interrupt()."""
    config = {"configurable": {"thread_id": req.thread_id}}
    state_obj = vip_booking_graph.get_state(config)

    if not state_obj or not state_obj.values:
        raise HTTPException(status_code=404, detail="Run thread not found.")

    # Resume from the interrupt() inside shared_hitl_approval_node, not from the entry point.
    # This prevents decompose/execute_react from silently re-running.
    result = vip_booking_graph.invoke(
        Command(resume={"approved": True, "feedback": ""}),
        config=config
    )
    return {"success": True, "new_status": result.get("status")}

# ==========================================
# 2B. FAILURE TICKET QUEUE
# ==========================================
# Tickets are unplanned failures with their own persisted record and status
# lifecycle (open/investigating/resolved) in the Tickets table -- see
# state_graph/tickets.py for why this is intentionally not the same
# mechanism as HITL above.

@app.get("/api/admin/tickets")
def get_tickets(status: str = None):
    """Lists failure tickets, optionally filtered by status."""
    return ticket_store.list_tickets(status=status)

class TicketStatusRequest(BaseModel):
    status: str  # "open" | "investigating" | "resolved"

@app.post("/api/admin/tickets/{ticket_id}/status")
def set_ticket_status(ticket_id: str, req: TicketStatusRequest):
    """Moves a ticket through its lifecycle without touching the underlying
    run. Lets an admin mark something 'investigating' before it's actually
    fixed, without prematurely resuming a broken run.
    """
    try:
        return ticket_store.update_ticket_status(ticket_id, req.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

class ResolveTicketRequest(BaseModel):
    approved: bool = True
    feedback: str = ""

@app.post("/api/admin/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str, req: ResolveTicketRequest = None):
    """Marks a ticket resolved AND resumes the underlying run from its exact
    checkpoint via the ticket's stored thread_id -- not restarted from
    scratch.

    Delegates to resume_graph_from_ticket() in state_graph/tickets.py which
    branches on graph_name and handles all three graphs:
    - incident_escalation_graph: Command(resume=...) from interrupt() node.
    - complaint_resolution_graph: Command(resume=...) or re-invoke.
    - vip_booking_graph: re-invoke from entry point with updated state dict.
    """
    ticket = ticket_store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    approved = req.approved if req else True
    feedback = req.feedback if req else ""

    result = resume_graph_from_ticket(ticket_id, approved, feedback)
    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Resume failed: {result.get('error', 'unknown error')}"
        )
    return result

# ==========================================
# 2C. COMPLAINT RESOLUTION GRAPH (Graph #2 -- Person B)
# ==========================================
# Uses interrupt()/Command(resume=...) rather than the vip_booking_graph's
# re-invoke-from-entry pattern -- see state_graph/nodes_complaint.py for why.

import uuid as _uuid

class StartComplaintRequest(BaseModel):
    reservation_id: int
    guest_id: int
    issue_type: str
    priority: str = "Medium"
    created_by: int  # staff_id logging the complaint
    max_rounds: int = 3

@app.post("/api/complaints/start")
def start_complaint(req: StartComplaintRequest):
    """Kicks off a new guest complaint resolution run."""
    thread_id = f"complaint-{_uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "request_id": None, "reservation_id": req.reservation_id, "guest_id": req.guest_id,
        "issue_type": req.issue_type, "priority": req.priority, "created_by": req.created_by,
        "round_number": 0, "max_rounds": req.max_rounds,
        "proposed_type": None, "proposed_amount": None, "policy_context": None,
        "candidate_offers": [], "guest_decision": None, "requires_manager_approval": False,
        "hitl_reason": None, "is_approved": None, "compensation_id": None,
        "status": "DRAFTING", "error_message": None, "ticket_id": None,
    }

    result, ticket_id = run_complaint_graph(initial_state, config)
    return {"thread_id": thread_id, "state": result, "ticket_id": ticket_id}

def _list_complaint_threads_paused_on(node_name: str):
    """Scans checkpoints for complaint_resolution_graph threads currently
    interrupted at `node_name`. Thread IDs are prefixed 'complaint-' so they
    can be distinguished from vip_booking_graph threads sharing the same
    physical checkpoints table in hotel.db.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'complaint-%'")
    thread_ids = [r[0] for r in cursor.fetchall()]
    conn.close()

    out = []
    for tid in thread_ids:
        config = {"configurable": {"thread_id": tid}}
        try:
            snap = complaint_resolution_graph.get_state(config)
            if snap and snap.next == (node_name,):
                out.append({"thread_id": tid, **snap.values})
        except Exception:
            continue
    return out

@app.get("/api/admin/complaints/hitl-tasks")
def get_complaint_hitl_tasks():
    """Pending manager-approval pauses in the complaint resolution graph."""
    return _list_complaint_threads_paused_on("hitl_manager_approval")

class ComplaintHitlApproveRequest(BaseModel):
    thread_id: str
    approved: bool

@app.post("/api/admin/complaints/hitl-tasks/approve")
def approve_complaint_hitl(req: ComplaintHitlApproveRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result, ticket_id = run_complaint_graph(Command(resume={"approved": req.approved}), config)
    return {"success": True, "state": result, "ticket_id": ticket_id}

@app.get("/api/admin/complaints/awaiting-guest")
def get_complaint_awaiting_guest():
    """Runs currently paused waiting on the guest's reply."""
    return _list_complaint_threads_paused_on("await_guest_response")

class ComplaintGuestResponseRequest(BaseModel):
    thread_id: str
    guest_decision: str  # "accept" | "reject" | "counter"

@app.post("/api/admin/complaints/guest-response")
def record_complaint_guest_response(req: ComplaintGuestResponseRequest):
    """Staff record the guest's actual reply (received via whatever real
    channel -- email/SMS -- in a full deployment), resuming the run exactly
    where it paused, possibly hours or days after it started.
    """
    if req.guest_decision not in ("accept", "reject", "counter"):
        raise HTTPException(status_code=400, detail="guest_decision must be accept, reject, or counter.")
    config = {"configurable": {"thread_id": req.thread_id}}
    result, ticket_id = run_complaint_graph(Command(resume={"guest_decision": req.guest_decision}), config)
    return {"success": True, "state": result, "ticket_id": ticket_id}

# ==========================================
# 2D. INCIDENT & MAINTENANCE ESCALATION GRAPH (Graph #3 -- Person C)
# ==========================================
# Uses interrupt()/Command(resume=...) — same pattern as Graph 2.
# Thread IDs are prefixed 'incident-' to distinguish from other graphs.

class StartIncidentRequest(BaseModel):
    guest_id: str
    room_number: str
    issue_description: str

@app.post("/api/incidents/start")
def start_incident(req: StartIncidentRequest):
    """Kicks off a new incident escalation run."""
    thread_id = f"incident-{_uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "guest_id": req.guest_id,
        "room_number": req.room_number,
        "issue_description": req.issue_description,
        "severity_level": None,
        "policy_analysis": None,
        "requires_human_approval": False,
        "resolution_plan": None,
        "human_approved": None,
        "ticket_id": None,
        "status": "IN_PROGRESS",
        "error_message": None,
    }

    result, ticket_id = run_incident_graph(initial_state, config)
    return {"thread_id": thread_id, "state": result, "ticket_id": ticket_id}


def _list_incident_threads_paused_on(node_name: str):
    """Scans checkpoints for incident_escalation_graph threads currently
    interrupted at `node_name`. Thread IDs are prefixed 'incident-'.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'incident-%'")
    thread_ids = [r[0] for r in cursor.fetchall()]
    conn.close()

    out = []
    for tid in thread_ids:
        config = {"configurable": {"thread_id": tid}}
        try:
            snap = incident_escalation_graph.get_state(config)
            if snap and snap.next == (node_name,):
                out.append({"thread_id": tid, **snap.values})
        except Exception:
            continue
    return out


@app.get("/api/admin/incidents/hitl-tasks")
def get_incident_hitl_tasks():
    """Pending manager-approval pauses in the incident escalation graph."""
    return _list_incident_threads_paused_on("hitl_approval")


class IncidentHitlApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


@app.post("/api/admin/incidents/hitl-tasks/approve")
def approve_incident_hitl(req: IncidentHitlApproveRequest):
    """Manager approves or declines a high-severity incident escalation."""
    config = {"configurable": {"thread_id": req.thread_id}}
    result, ticket_id = run_incident_graph(
        Command(resume={"approved": req.approved, "feedback": req.feedback}), config
    )
    return {"success": True, "state": result, "ticket_id": ticket_id}


# ==========================================
# 3. RAG DOCUMENT MANAGEMENT
# ==========================================
# Every endpoint here calls sync_documents() after touching DOCS_DIR, so the
# vector + BM25 indexes are rebuilt from whatever's on disk and the RAG
# agent's *next query* reflects the change -- not just the filesystem. See
# memory/stores/builder.py for the sync logic.

@app.get("/api/admin/documents")
def list_documents():
    """Lists all documents stored in the RAG corpus."""
    files = os.listdir(DOCS_DIR)
    return {"documents": files}

@app.post("/api/admin/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a new document to the RAG repository and reindexes it into
    the vector + BM25 stores so it's retrievable on the agent's next query.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Validate the file parses before it's allowed to join the shared corpus.
    # A single malformed upload must not be able to degrade retrieval for
    # every other document -- fail loudly here instead.
    try:
        validate_document(Path(file_path))
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process '{file.filename}': {e}. File was not added."
        )

    try:
        sync_result = sync_documents()
    except Exception as e:
        # Reindex failed after the file was already validated -- remove it
        # rather than leave the corpus in a mismatched state (file on disk,
        # not in the index).
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    return {"success": True, "filename": file.filename, "reindex": sync_result}

@app.delete("/api/admin/documents/{filename}")
def delete_document(filename: str):
    """Deletes a document from the RAG store and reindexes so the agent
    stops retrieving it on the next query.
    """
    file_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found.")

    os.remove(file_path)
    try:
        sync_result = sync_documents()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File was deleted but reindex failed: {e}. Retrieval may be stale until the index is rebuilt."
        )

    return {"success": True, "reindex": sync_result}

# Single-page Admin Dashboard HTML Interface
@app.get("/", response_class=HTMLResponse)
def admin_dashboard():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        return f.read()
