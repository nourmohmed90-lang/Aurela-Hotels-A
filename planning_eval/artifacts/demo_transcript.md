# Demo Transcript — Aurelia Hotels Planning Agent

This document provides illustrative step-by-step execution traces for all three planning strategies,
the Decomposition-First vs Dynamic divergence, and a Reflexion self-correction example.

---

## Scenario A — LATS + Reflexion: VIP Room 501 Water Leak Crisis

**Input:** `"VIP Room leak for Guest Sara Mohamed under reservation 2. Requires emergency rebooking and $1000 refund."`
**Routed to:** `LATS (MCTS + Reflexion)` → then Self-Refine polish

### LATS Iteration 1

**Action Generator proposes 2 candidate actions:**

```
Candidate 1 (action): "Check VIP reservation and transfer to Suite"
  State: "Sara Mohamed is VIP (Gold tier). Reservation 2 confirmed.
          Room 103 has active water leak. Transferring to Suite 203 (available).
          Refund approved: $1000 within the $4000 policy limit."

Candidate 2 (action): "Exceed compensation limit"
  State: "Issue $5500 refund immediately to compensate for damage."
```

**Environment.evaluate() runs grounded checks:**

```
Candidate 1:
  ✅ Reservation 2 validated (Status: active, Price: $320.00)
  ✅ Found guest profile: Sara Mohamed (VIP tier)
  ✅ VIP protocols explicitly mentioned
  ✅ Refund $1000 is within $4000 limit
  ✅ Suite rooms available: 2 available
  → score = 1.0, success = True

Candidate 2:
  ❌ Compensation amount $5500 exceeds permitted limit of $4000.0
  → score = 0.3, success = False
  [Reflection generated]: "I proposed a $5500 refund which violates the hotel
   compensation policy capped at $4000. I should next time verify amounts against
   the policy limit before recommending financial compensation."
```

**LATS selects Candidate 1 (score=1.0) → success on first iteration.**

### Self-Refine Polish Step

```
Critique: "The output is complete and accurate. PASS."
Revised output (unchanged — draft accepted):
  "Sara Mohamed is VIP. Reservation 2 confirmed. Room 103 has leak.
   Transferring to Suite 203. Refund: $1000 approved."
```

**Final Output:** Rebooking and refund plan delivered. ✅

---

## Scenario B — Tree of Thoughts: Multi-Option Compensation Negotiation

**Input:** `"Guest wants to rebook or get a voucher due to noise complaints. Compare alternatives."`
**Routed to:** `Tree of Thoughts` → then Self-Refine polish

### ToT Depth 1 — Generating Thought Candidates

```
Parent: [Start]

Candidate branches generated:
  Branch A: "Offer a 20% discount voucher ($64) and room soundproofing upgrade."
  Branch B: "Rebook guest to a quieter Suite on Floor 5 at no extra charge."

Evaluation scores:
  Branch A: score=0.72  (rationale: "Voucher is fast but doesn't fix the root cause")
  Branch B: score=0.91  (rationale: "Relocation fully resolves noise; zero cost to guest")
```

### ToT Depth 2 — Expanding Best Branch (B)

```
Parent: Branch B — "Rebook to Suite on Floor 5"

Candidate branches generated:
  Branch B1: "Complete the rebook immediately, offer complimentary breakfast."
  Branch B2: "Rebook + issue $50 F&B credit as goodwill gesture."

Evaluation scores:
  Branch B1: score=0.88
  Branch B2: score=0.94  ← selected as best leaf
```

### Self-Refine Step

```
Draft: "Rebook guest to Floor 5 Suite + issue $50 F&B credit."
Deterministic check: Draft is 8 words — below 80 word minimum.
Critique: "Output is too brief. Needs explicit guest confirmation, timeline, and staff action items."
Revised: "Action Plan: (1) Contact front desk to prepare Suite 507 (Floor 5, quiet wing).
          (2) Issue $50 F&B credit to guest account #2. (3) Escort guest within 30 minutes.
          (4) Confirm resolution via SMS. (5) Log noise complaint for maintenance review."
```

**Final Output:** Comprehensive multi-step rebooking plan. ✅

---

## Scenario C — Plan-and-Solve + Self-Refine: Linear Laundry Request

**Input:** `"Guest Sara Mohamed requests extra pillows and laundry service for reservation 2."`
**Routed to:** `Plan-and-Solve` → then Self-Refine polish

### Plan-and-Solve Output

```
PLAN:
1. Verify reservation 2 belongs to Sara Mohamed.
2. Dispatch housekeeping to deliver 2 extra pillows to her room.
3. Schedule laundry pickup for 10:00 AM tomorrow.
4. Confirm service completion via front desk log.

SOLUTION:
Reservation 2 verified for Sara Mohamed (Room 103).
Extra pillows dispatched — ETA 15 minutes.
Laundry pickup scheduled: 10:00 AM.
Front desk notified and logged.
```

### Self-Refine Step

```
Deterministic checks: ✅ Word count OK, ✅ Goal terms present, ✅ Structured with list items.
Critique: "PASS — output is complete and well-structured."
Revised: (unchanged — draft accepted)
```

**Final Output:** Linear service request fulfilled. ✅

---

## Decomposition-First vs Dynamic Decomposition — Divergence Trace

**Goal:** `"Guest delayed 5 hours due to flight, adjust check-in, spa, and conference schedule."`

### Method A: Decomposition-First (Full Upfront DAG)

```
Planned tasks (produced all at once before any execution):

  t1: "Notify front desk of delayed arrival time"        [no deps]
  t2: "Reschedule spa appointment by 5 hours"            [no deps]
  t3: "Reschedule conference room booking"               [no deps]
  t4: "Prepare late check-in welcome package"            [depends: t1]
  t5: "Confirm all adjustments with guest via SMS"       [depends: t2, t3, t4]

Execution batches (parallel-safe):
  Batch 1 (parallel): [t1, t2, t3]   ← spa + conference run simultaneously
  Batch 2 (sequential): [t4]
  Batch 3 (sequential): [t5]         ← synthesis step
```

### Method B: Dynamic Decomposition (Reactive, Step-by-Step)

```
Step 1: → "Check current check-in queue and flag late arrival"
         (observes: front desk queue has 3 guests ahead)
Step 2: → "Contact spa to push appointment 5 hours forward"
         (observes: spa confirms 8 PM slot available)
Step 3: → "Alert conference coordinator of revised slot"
         (observes: coordinator confirms rescheduled to 9 PM)
Step 4: → [done=True] — goal achieved
```

### Key Divergence

| Aspect | Decomposition-First | Dynamic Decomposition |
|---|---|---|
| Planning moment | Upfront — full DAG before execution | Per-step — decide after each observation |
| Parallelism | ✅ t1, t2, t3 run simultaneously | ❌ Sequential — one task at a time |
| Adaptability | ❌ Rigid — plan fixed at start | ✅ Adapts if spa is closed or queue is empty |
| Task count | 5 tasks (incl. synthesis) | 3 steps (no synthesis needed) |
| Best for | Stable, predictable multi-branch work | Uncertain, observation-dependent situations |

---

## Reflexion Self-Correction — Memory Carry-Across Example

**Task:** Issue a compensation plan with refund for a damaged room.

### Trial 1 (fails)

```
Attempt: "Issue $6000 refund to cover full room damage and rebooking costs."
Environment: ❌ $6000 exceeds $4000 policy limit. Score=0.3.
Reflection generated (starts with "I"):
  "I recommended a refund of $6000 which violates the hotel's maximum compensation
   limit of $4000. I should next trial cap any financial compensation at $4000 and
   split costs between refund and service vouchers if needed."
Memory buffer: ["I recommended a $6000 refund... cap at $4000 next time."]
```

### Trial 2 (uses memory, succeeds)

```
Episodic memory recalled: "I recommended a $6000 refund... cap at $4000 next time."

Attempt: "Issue $2000 refund + $1500 F&B voucher (total $3500 within policy).
          Rebook guest to Suite 203 at no extra charge."
Environment: ✅ $3500 within $4000 limit. Score=1.0. Success=True.
```

**Reflexion correctly carried the "I..." memory from Trial 1 into Trial 2 and recovered.** ✅
