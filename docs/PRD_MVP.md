MVP PRD — Movie Recommender
1) Goal

Ship a minimal movie recommender product that demonstrates real-world software engineering (data pipeline + recommendation logic + user-facing experience) and is stable enough to deploy and share.

2) Success Criteria (MVP)

A user should be able to:

Rate 5–15 movies quickly (under ~2 minutes)

Get a ranked list of recommendations

See posters + basic metadata (title/year/genres)

See simple “why” explanations for recommendations

Leave and come back (session persistence) without losing ratings

Engineering success criteria:

Runs locally with clear setup steps

Deterministic recommendation output for the same inputs

Deployed demo URL (goal)

3) Non-goals (Out of Scope for MVP)

Letterboxd scraping

Full accounts/password auth

Social features (friends, follows, feeds)

Heavy ML training per user in real time

Perfect coverage of all movies ever (bounded catalog is fine)

4) Target User + Primary Flow

Target user: someone who wants quick movie recommendations and is willing to rate a handful of movies.

Primary flow:

User visits the app

Searches for movies they’ve seen and rates them

Clicks “Get Recommendations”

Receives recommendations + “why” chips

Optionally shares/returns via a session link

5) Product Scope (Pages)
/rate

Search bar

Search results (poster, title, year)

Rate action (like/dislike; see section 6)

“Rated list” panel

/recs

Ranked recommendations (poster, title, year, genres)

“Why” chips

Button: “Back to rate more”

Optional:

/about (simple explanation)

6) Ratings Design Decision (MVP choice)

MVP rating scheme: Like/Dislike

rating = +1 (like)

rating = -1 (dislike)

Reason:

simplest UX

easiest debugging

clean cold-start behavior

7) Data + Catalog Definition (Bounded for speed)

To keep recommendation generation fast and debuggable, define a bounded catalog.

Catalog rule (MVP):

Recommendations come from a local catalog of the top N movies by popularity (suggest N = 10,000).

Catalog is refreshed periodically (nightly or manual for MVP).

Stored fields per movie (minimum):

tmdb_id

title

year

genres[]

poster_path

popularity

vote_average

vote_count

overview (optional)

Why:

avoids “all of TMDB” being unbounded

reduces rate-limit pain

makes results reproducible

8) Identity + Persistence

Default: session-based

Create a session_id UUID and store it in localStorage (frontend).

Store ratings server-side keyed by session_id (backend).

Optional (MVP-friendly):

Shareable link: /recs?s=<session_id>

No passwords in MVP.

9) Recommendation Strategy (Ship baseline first)
Stage 0 — Baseline recommender (first shipping milestone)

Return popular movies not rated by the user:

exclude rated tmdb_ids

rank remaining by popularity (and optionally vote_count)

Purpose:

proves the full pipeline end-to-end with low algorithm risk

Stage 1 — Content-based recommender (MVP “real” logic)

Compute similarity from metadata:

genre overlap (e.g., Jaccard)

year proximity bonus/penalty

small popularity prior to break ties

Candidate generation:

for each liked movie, consider catalog movies sharing >= 1 genre

apply a year window (start with ±15 years)

aggregate scores across liked movies (sum or max)

Score sketch:

score = alphagenre_sim + betayear_sim + gamma*popularity_norm

10) Cold Start Rules + Gates

Gate to enable personalized recs:

enable at 10 total ratings, OR

enable at 5 likes if candidate pool >= 20

Before gate is met:

show progress (“Rate 2 more to unlock personalized recs”)

optionally show baseline popular list as a preview

Sparse candidate fallbacks (debug saver):
If personalized rec list < 20:

expand year window

reduce genre constraint

mix in baseline popular recs

11) “Why” Explanations (Deterministic)

Each recommendation returns up to 3 “why” chips:

Because you liked {Movie}

Shares genres: Action, Thriller

Similar era: 2010s

No LLM needed.

12) API Contract (Minimal Examples)

These are example HTTP endpoints. Implement via FastAPI, Next.js API routes, etc.

12.1 Create Session

Method: POST
Path: /api/session

Response JSON:
{
"session_id": "uuid-string"
}

12.2 Search Movies

Method: GET
Path: /api/search
Query params: q=<string>

Response JSON:
{
"results": [
{
"tmdb_id": 603,
"title": "The Matrix",
"year": 1999,
"poster_path": "...",
"genres": ["Action", "Sci-Fi"]
}
]
}

12.3 Save Rating

Method: POST
Path: /api/session/{session_id}/ratings

Request JSON:
{
"tmdb_id": 603,
"rating": 1
}

Response JSON:
{
"ok": true
}

12.4 Get Recommendations

Method: GET
Path: /api/session/{session_id}/recs

Response JSON:
{
"recs": [
{
"tmdb_id": 155,
"title": "The Dark Knight",
"year": 2008,
"poster_path": "...",
"genres": ["Action", "Crime"],
"score": 0.82,
"why": [
"Because you liked The Matrix",
"Shares genres: Action",
"Similar era: 2000s"
]
}
]
}

13) Risks + Mitigations

Rate limits -> cache search results + bounded catalog

Sparse/empty recs -> fallback rules in section 10

Debug fear -> tiny milestones + commit after each + basic tests on recommender function