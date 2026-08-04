from . import mcp

# RESOURCES
from . import mcp

@mcp.resource("policy://guest-compensation")
def compensation_policy():

    return """
Guest Compensation Policy

1. Overbooking:
   - Free room upgrade if available.
   - Complimentary breakfast.
   - Late checkout.

2. Room Under Maintenance:
   - Upgrade if possible.
   - 20% room discount.

3. Room Not Ready:
   - Free lounge access.
   - Welcome drink.

4. Guest Transfer:
   - Free transportation.
   - Discount voucher.

5. VIP Guest:
   - Manager approval required.
"""
@mcp.resource(
    "policy://manager-approval"
)
def manager_policy():

    return """
Manager approval is required when:

- VIP guest reservation
- Compensation exceeds $200
- Guest transfer to another branch
- Reservation cancellation after check-in
"""
@mcp.resource(
    "policy://vip-guidelines"
)
def vip_guidelines():

    return """
VIP Guest Policy

- Never downgrade room.

- Manager approval required
  before transferring.

- Offer complimentary transport.

- Offer premium compensation
  when necessary.
"""