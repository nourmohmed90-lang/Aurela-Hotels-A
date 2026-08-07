# Aurelia Hotels Compensation Policy

## Purpose
This document defines the rules for approving guest compensation requests during recovery cases.

## Compensation Types
- Room Upgrade
- Discount
- Free Night
- Meal Voucher
- Refund

## Approval Rules
- Compensation amount must not be negative.
- Requests under $100 can be approved by a Front Desk Supervisor.
- Requests of $100 or more require Manager approval.
- VIP guests receive higher priority during compensation decisions.
- Every approved compensation must be linked to a valid recovery request.

## Defensive Design Rules
- Reject requests with invalid request IDs.
- Reject requests that have already been approved.
- Reject negative compensation amounts.
- Only users with the Manager role can approve high-value compensations.
