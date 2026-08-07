# Aurelia Hotels Cancellation Policy

## Purpose
This document defines the rules for cancelling hotel reservations.

## Cancellation Rules
- Reservations may be cancelled free of charge up to 24 hours before check-in.
- Late cancellations may incur a one-night charge.
- Non-refundable reservations cannot be cancelled for a refund.
- Refunds are processed using the original payment method.

## Special Cases
- VIP guests may receive flexible cancellation privileges.
- Cancellations caused by hotel operational issues are fully refundable.

## Defensive Design Rules
- Reject cancellation requests for non-existent reservations.
- Reject duplicate cancellation requests.
- Record every cancellation with a timestamp and reason.
- Notify the guest after successful cancellation.