import time
from . import mcp
@mcp.prompt()
def draft_guest_apology(
    guest_name: str,
    issue: str
):

    return f"""
Write a professional apology email.

Guest:
{guest_name}

Issue:
{issue}

The email should:

- apologize sincerely
- explain the situation
- reassure the guest
- remain professional
"""
@mcp.prompt()
def draft_transfer_message(
    guest_name: str,
    old_branch: str,
    new_branch: str
):

    return f"""
Write a message informing

Guest:
{guest_name}

that their reservation has been transferred

From:
{old_branch}

To:
{new_branch}

Use a polite and reassuring tone.
"""
@mcp.prompt()
def draft_compensation_email(

    guest_name: str,

    compensation: str

):

    return f"""
Write a professional email.

Guest:
{guest_name}

Compensation:

{compensation}

Explain why the hotel is offering
this compensation.

Remain polite and empathetic.
"""