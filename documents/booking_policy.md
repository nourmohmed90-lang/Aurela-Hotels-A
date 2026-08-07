# Aurelia Hotels Booking Policy

## Purpose
This document defines the rules for creating, modifying, and confirming hotel reservations.

## Booking Rules
- Guests must provide valid identification when booking.
- Reservations are confirmed only after payment or a valid payment guarantee.
- Room availability is determined in real time.
- Guests may request special accommodations during booking.
- Booking modifications are subject to room availability.

## Check-in Requirements
- Standard check-in begins at 3:00 PM.
- Early check-in is subject to availability.
- Guests must present a valid government-issued ID.

## Defensive Design Rules
- Reject bookings with invalid guest information.
- Reject duplicate reservation requests.
- Do not confirm bookings without payment authorization.
- Prevent overbooking by verifying room availability before confirmation.