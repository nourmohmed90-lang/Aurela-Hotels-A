import sqlite3
from pathlib import Path
from ..models import EnvironmentFeedback

class Environment:
    """A grounded evaluator that performs real checks using the Aurelia Hotels SQLite database."""

    def __init__(self, db_path: str | None = None):
        # Default to database/hotel.db under the workspace root
        if db_path is None:
            project_root = Path(__file__).resolve().parents[2]
            db_path = str(project_root / "database" / "hotel.db")
        self.db_path = db_path

    def get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def evaluate(self, state: str) -> EnvironmentFeedback:
        """
        Evaluates the proposed solution or action state by validating business rules,
        reservation checks, availability checks, or refund/compensation limits.
        """
        state_lower = state.lower()
        details = []
        success = True
        score = 1.0

        # Rule 1: Financial Limit Check (Maximum refund or compensation approval is $4000.0)
        # Scan state for monetary numbers (e.g. $5000, 4500, etc.)
        import re
        amounts = re.findall(r"\$?\b(\d+(?:\.\d{1,2})?)\b", state)
        for amt in amounts:
            try:
                val = float(amt)
                # If they mention a large value, check if it looks like a refund or compensation amount
                if val > 4000.0:
                    # Let's verify context around the number
                    context_match = re.search(rf"(?:refund|compensat|pay|limit|cost|price|voucher|amount).*?{re.escape(amt)}|{re.escape(amt)}.*?(?:refund|compensat|pay|limit|cost|price|voucher|amount)", state_lower)
                    if context_match or "refund" in state_lower or "compensation" in state_lower:
                        success = False
                        score = min(score, 0.3)
                        details.append(f"Compensation or refund amount ${val} exceeds the permitted limit of $4000.0.")
            except ValueError:
                pass

        # Rule 2: Database validation of guest / reservations / rooms if IDs or names are mentioned
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # Check if reservation ID is mentioned
        reservation_ids = re.findall(r"\breservation\s*(?:id\s*)?#?(\d+)\b", state_lower)
        if reservation_ids:
            for res_id in reservation_ids:
                cursor.execute("SELECT reservation_id, guest_id, room_id, reservation_status, total_price FROM Reservations WHERE reservation_id = ?", (res_id,))
                res = cursor.fetchone()
                if not res:
                    success = False
                    score = min(score, 0.4)
                    details.append(f"Reservation ID {res_id} not found in the database.")
                else:
                    details.append(f"Validated Reservation ID {res_id} (Status: {res[3]}, Price: {res[4]}).")
                    # If status is Overbooked or VIP Conflict, check if the solution resolves it or mentions transfer
                    if res[3] == "Overbooked" and "transfer" not in state_lower and "rebook" not in state_lower:
                        score = min(score, 0.5)
                        details.append(f"Reservation {res_id} is Overbooked, but no transfer or rebooking action was proposed.")

        # Check for guest names
        cursor.execute("SELECT full_name, loyalty_level FROM Guests")
        guests = cursor.fetchall()
        guest_mentioned = False
        for guest_name, loyalty in guests:
            # Match first name or full name
            first_name = guest_name.split()[0].lower()
            if first_name in state_lower or guest_name.lower() in state_lower:
                guest_mentioned = True
                details.append(f"Found guest profile for {guest_name} ({loyalty} tier).")
                if loyalty == "VIP" and "vip" not in state_lower:
                    # VIP guest needs VIP protocol attention
                    score = min(score, 0.7)
                    details.append(f"Guest {guest_name} is VIP, but VIP protocols were not explicitly mentioned.")

        # Check if room categories are available if mentioned
        room_types = ["standard", "deluxe", "suite"]
        for rt in room_types:
            if rt in state_lower:
                # Query room availability
                cursor.execute("SELECT COUNT(*) FROM Rooms WHERE LOWER(room_type) = ? AND LOWER(room_status) = 'available'", (rt,))
                count = cursor.fetchone()[0]
                if count == 0:
                    # Let's see if they propose booking it anyway
                    if "book" in state_lower or "reserve" in state_lower or "transfer to" in state_lower:
                        score = min(score, 0.5)
                        success = False
                        details.append(f"Proposed booking a {rt.capitalize()} room, but none are available.")
                else:
                    details.append(f"Verified available {rt.capitalize()} rooms: {count} available.")

        conn.close()

        # Fallback if no specific failure details but details contains general confirmations
        if success and score > 0.6:
            success = True
        else:
            success = False

        if not details:
            details.append("No specific grounded database elements (reservations, guests, rooms) could be validated from the draft.")

        return EnvironmentFeedback(success=success, score=round(score, 4), details=details)
