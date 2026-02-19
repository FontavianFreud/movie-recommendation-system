# NEXT (Only do these 3 tickets)

## Ticket 1 — Add docs + guardrails
Acceptance:
- `docs/PRD_MVP.md`, `MILESTONES.md`, `NEXT.md` exist in repo
- Commit pushed to GitHub

## Ticket 2 — Choose rating scheme + document it
Pick one and update PRD:
- Option A: like/dislike (-1, +1)
- Option B: stars (1–5)
Acceptance:
- PRD specifies the scheme
- Existing code (if any) aligns with it

## Ticket 3 — Build the smallest “baseline recommender” function
Acceptance:
- Given a list of rated movie IDs, return top 20 unrated movies
- Deterministic ordering (same input → same output)
- Quick test script exists (can run in <5s)

