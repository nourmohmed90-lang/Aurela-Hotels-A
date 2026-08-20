import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Connect to database/hotel.db
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))
conn = sqlite3.connect(db_path)

# Initialize LangGraph checkpoint tables in hotel.db
memory = SqliteSaver(conn)
memory.setup()

print("Successfully created checkpoint tables in hotel.db!")
conn.close()