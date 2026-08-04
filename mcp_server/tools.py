from __future__ import annotations
import time
from jsonschema import validate
from mcp.server.fastmcp import Context
from . import mcp
from .schema import ROOM_SEARCH_SCHEMA
from .server import get_db
from pydantic import BaseModel

@mcp.tool()
def get_reservation(reservation_id: int):
    """
    Return reservation details matching the database schema.
    """
    conn = get_db()
    row = conn.execute(
        """
        SELECT reservation_id, guest_id, room_id, check_in, check_out, reservation_status, total_price
        FROM Reservations
        WHERE reservation_id = ?
        """,
        (reservation_id,)
    ).fetchone()
    conn.close()

    if not row:
        return {
            "error": "Reservation not found."
        }

    return {
        "reservation_id": row[0],
        "guest_id": row[1],
        "room_id": row[2],
        "check_in": row[3],
        "check_out": row[4],
        "reservation_status": row[5],
        "total_price": row[6],
    }


@mcp.tool()
def search_available_rooms(room_type: str):
    """Search available rooms."""

    room_type = room_type.strip().lower()

    mapping = {
        "standard": "Standard",
        "standard room": "Standard",
        "deluxe": "Deluxe",
        "deluxe room": "Deluxe",
        "suite": "Suite",
        "suite room": "Suite",
        "all": "ALL",
    }

    room_type = mapping.get(room_type, room_type)

    conn = get_db()
    cursor = conn.cursor()

    if room_type == "ALL":
        cursor.execute("""
            SELECT
                h.hotel_name,
                r.room_number,
                r.room_type,
                r.capacity,
                r.price_per_night
            FROM Rooms r
            JOIN Hotels h
                ON r.hotel_id = h.hotel_id
            WHERE LOWER(r.room_status)='available'
        """)
    else:
        cursor.execute("""
            SELECT
                h.hotel_name,
                r.room_number,
                r.room_type,
                r.capacity,
                r.price_per_night
            FROM Rooms r
            JOIN Hotels h
                ON r.hotel_id = h.hotel_id
            WHERE
                LOWER(r.room_status)='available'
                AND LOWER(r.room_type)=LOWER(?)
        """, (room_type,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "success": False,
            "message": f"No {room_type} rooms available."
        }

    return {
        "success": True,
        "rooms": [
            {
                "hotel": r[0],
                "room": r[1],
                "type": r[2],
                "capacity": r[3],
                "price": float(r[4]),
            }
            for r in rows
        ]
    }

@mcp.tool()
def find_alternative_branch(room_type: str):

    room_type = room_type.strip().lower()

    mapping = {
        "standard": "Standard",
        "standard room": "Standard",
        "deluxe": "Deluxe",
        "deluxe room": "Deluxe",
        "suite": "Suite",
        "suite room": "Suite",
    }

    room_type = mapping.get(room_type, room_type)

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            h.hotel_name,
            r.room_number,
            r.room_type
        FROM Rooms r
        JOIN Hotels h
            ON r.hotel_id=h.hotel_id
        WHERE
            LOWER(r.room_type)=LOWER(?)
            AND LOWER(r.room_status)='available'
    """, (room_type,))

    rows = cursor.fetchall()
    conn.close()

    return {
        "alternatives": [
            {
                "hotel": r[0],
                "room": r[1],
                "type": r[2],
            }
            for r in rows
        ]
    }

@mcp.tool()
async def search_all_branches(room_type: str, ctx: Context):
    """
    Search across all hotel branches with real MCP progress tracking.
    """
    conn = get_db()
    hotels = conn.execute(
        """
        SELECT hotel_id, hotel_name, city
        FROM Hotels
        """
    ).fetchall()
    
    total = len(hotels)
    progress_updates = []

    for index, hotel in enumerate(hotels):
        hotel_id, hotel_name, city = hotel
        
        # Report progress through MCP context
        await ctx.info(f"Checking branch: {hotel_name} ({city})")
        await ctx.report_progress(index + 1, total)
        time.sleep(0.5)

        rooms = conn.execute(
            """
            SELECT room_id, room_number, price_per_night
            FROM Rooms
            WHERE hotel_id = ? AND room_type = ? AND room_status = 'Available'
            """,
            (hotel_id, room_type)
        ).fetchall()

        progress_updates.append(
            {
                "hotel_name": hotel_name,
                "city": city,
                "available_rooms_count": len(rooms),
                "progress": int(((index + 1) / total) * 100),
                "message": f"Searched {hotel_name}"
            }
        )

    conn.close()
    return {
        "status": "completed",
        "searched_branches": len(progress_updates)
    }


@mcp.tool()
def analyze_reservation(reservation_id: int):
    """
    Analyze reservation status, guest loyalty, and potential risks.
    """
    conn = get_db()
    row = conn.execute(
        """
        SELECT g.full_name, r.room_type, h.hotel_name, res.reservation_status, g.loyalty_level
        FROM Reservations res
        JOIN Guests g ON res.guest_id = g.guest_id
        JOIN Rooms r ON res.room_id = r.room_id
        JOIN Hotels h ON r.hotel_id = h.hotel_id
        WHERE res.reservation_id = ?
        """,
        (reservation_id,)
    ).fetchone()
    conn.close()

    if not row:
        return {
            "error": "Reservation not found."
        }

    guest, room_type, hotel, status, loyalty = row

    if status == "Overbooked":
        risk = "High"
        recommendation = "Transfer guest to an alternative branch immediately."
    elif status == "Maintenance":
        risk = "Medium"
        recommendation = "Upgrade room or offer maintenance compensation."
    elif loyalty == "VIP":
        risk = "Medium"
        recommendation = "Ensure VIP service protocols and manager greeting."
    else:
        risk = "Low"
        recommendation = "Normal check-in process."

    return {
        "guest_name": guest,
        "room_type": room_type,
        "hotel_name": hotel,
        "reservation_status": status,
        "loyalty_level": loyalty,
        "risk_level": risk,
        "recommendation": recommendation
    }


@mcp.tool()
def approve_guest_transfer(
    reservation_id: int,
    target_hotel_id: int,
    employee_role: str
):
    """
    Approve guest transfer to another hotel branch (Manager only).
    """
    if employee_role != "manager":
        return {
            "error": "Only managers can approve guest transfers."
        }

    conn = get_db()
    reservation = conn.execute(
        """
        SELECT res.reservation_status, g.loyalty_level, res.room_id
        FROM Reservations res
        JOIN Guests g ON res.guest_id = g.guest_id
        WHERE res.reservation_id = ?
        """,
        (reservation_id,)
    ).fetchone()

    if not reservation:
        conn.close()
        return {
            "error": "Reservation not found."
        }

    status, loyalty, old_room_id = reservation

    if loyalty == "VIP":
        conn.close()
        return {
            "requires_human_confirmation": True,
            "message": "VIP guest transfer requires direct manager confirmation."
        }

    # Get room type of the old room
    old_room = conn.execute(
        "SELECT room_type FROM Rooms WHERE room_id = ?",
        (old_room_id,)
    ).fetchone()

    if not old_room:
        conn.close()
        return {"error": "Original room record not found."}

    # Find an available room of the same type in the target hotel
    new_room = conn.execute(
        """
        SELECT room_id FROM Rooms
        WHERE hotel_id = ? AND room_type = ? AND room_status = 'Available'
        LIMIT 1
        """,
        (target_hotel_id, old_room[0])
    ).fetchone()

    if not new_room:
        conn.close()
        return {
            "error": "No available rooms of matching type found in the target hotel."
        }

    new_room_id = new_room[0]

    # Execute transfer update
    conn.execute(
        """
        UPDATE Reservations
        SET room_id = ?, reservation_status = 'Transferred'
        WHERE reservation_id = ?
        """,
        (new_room_id, reservation_id)
    )
    
    # Update room statuses
    conn.execute("UPDATE Rooms SET room_status = 'Occupied' WHERE room_id = ?", (new_room_id,))
    conn.execute("UPDATE Rooms SET room_status = 'Available' WHERE room_id = ?", (old_room_id,))
    
    conn.commit()
    conn.close()

    return {
        "status": "Guest transferred successfully.",
        "new_room_id": new_room_id
    }


@mcp.tool()
def resolve_overbooking(
    reservation_id: int,
    target_hotel_id: int,
    employee_role: str
):
    """
    Resolve overbooking issues by transferring the reservation to a new hotel branch.
    """
    if employee_role != "manager":
        return {
            "error": "Only managers can resolve overbookings."
        }

    conn = get_db()
    reservation = conn.execute(
        """
        SELECT reservation_status, room_id
        FROM Reservations
        WHERE reservation_id = ?
        """,
        (reservation_id,)
    ).fetchone()

    if reservation is None:
        conn.close()
        return {
            "error": "Reservation not found."
        }

    if reservation[0] != "Overbooked":
        conn.close()
        return {
            "error": "Reservation is not marked as overbooked."
        }

    old_room_id = reservation[1]
    room_info = conn.execute(
        "SELECT room_type FROM Rooms WHERE room_id = ?",
        (old_room_id,)
    ).fetchone()

    new_room = conn.execute(
        """
        SELECT room_id FROM Rooms
        WHERE hotel_id = ? AND room_type = ? AND room_status = 'Available'
        LIMIT 1
        """,
        (target_hotel_id, room_info[0])
    ).fetchone()

    if not new_room:
        conn.close()
        return {
            "error": "No available rooms found in the target hotel."
        }

    new_room_id = new_room[0]

    conn.execute(
        """
        UPDATE Reservations
        SET room_id = ?,
            reservation_status = 'Transferred'
        WHERE reservation_id = ?
        """,
        (new_room_id, reservation_id)
    )
    conn.execute("UPDATE Rooms SET room_status = 'Occupied' WHERE room_id = ?", (new_room_id,))
    conn.commit()
    conn.close()

    return {
        "status": "Guest successfully transferred and overbooking resolved.",
        "new_room_id": new_room_id
    }


@mcp.tool()
def recommend_compensation(issue: str):
    """
    Recommend appropriate recovery compensation based on the issue type.
    """
    recommendations = {
        "Overbooking": "Free room upgrade and complimentary breakfast",
        "Maintenance": "20% room discount and lounge access",
        "Room Not Ready": "Free welcome drink and lounge access",
        "Double Booking": "Upgrade to Deluxe Room",
        "VIP": "Luxury Suite Upgrade and transport service"
    }

    return {
        "issue": issue,
        "recommendation": recommendations.get(
            issue,
            "Manager review required for custom compensation"
        )
    }

class ManagerApproval(BaseModel):
    manager_note: str

@mcp.tool()
async def approve_compensation(
    ctx: Context,
    request_id: int,
    approved_by: int,
    amount: float,
):
    """
    Approve compensation request with manager confirmation.
    """

    if amount > 4000.0:
        return {
            "error": "Compensation amount exceeds allowed limit."
        }

    approval = await ctx.elicit(
        "Manager approval required before approving this compensation.",
        ManagerApproval,
    )

    if approval.action != "accept":
        return {
            "status": "cancelled",
            "message": "Manager rejected the request."
        }

    return {
        "status": "approved",
        "request_id": request_id,
        "amount": amount,
        "approved_by": approved_by,
        "manager_note": approval.data.manager_note,
    }