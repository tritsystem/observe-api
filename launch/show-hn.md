# Show HN draft

Post this yourself at news.ycombinator.com/submit under your own account --
I'm not going to post to HN on your behalf. Edit freely; this is a
starting draft, not a final copy.

**Rewritten from the original draft** to lead with the commerce/
reputation layer instead of code search. Reason: this session found real,
active competitors for "semantic code search for agents" (Semble --
MinishLab/semble, 98% fewer tokens than grep, got real HN traction in
2026 -- plus Claude Context, WarpGrep, `cs`, colGREP), and three attempts
to close that specific gap (AST chunking, RRF fusion, a definition-boost
reranker) all measured net negative on OBSERVE's own corpus -- see
README.md. Leading with an unverified "we're as good at code search"
claim into a crowd that already has a benchmark-backed incumbent is a bad
bet. The ACP/UCP cross-merchant discovery + two-sided reputation layer
has no equivalent found anywhere in that research. It's also what the
live landing page itself now leads with -- this draft should match what
a visitor actually sees when they click through.

## Title

```
Show HN: A cross-merchant discovery + reputation layer for AI agent commerce
```

(Not "OBSERVE" up front -- HN cares what it does before what it's called.)

## Body (as a text post, or link + first comment)

The Agentic Commerce Protocol (OpenAI + Stripe) and Google's UCP both
define checkout between a buyer-agent and one already-known merchant.
I read both specs directly, not secondhand summaries, before building
anything: ACP's own OpenAPI spec has no seller_id field at all, and
explicitly leaves multi-seller discovery to "the marketplace layer above
this API." UCP has the identical gap. Neither protocol says how a
buyer-agent finds sellers across many merchants, or decides whether to
trust one it's never dealt with before.

This is a real, working implementation of that missing layer. A seller
registers their own real ACP `checkout_session_url` and a listing feed
(free). A buyer-agent describes what it wants in plain language;
semantic search ranks real listings against it. Every match gets a
shared `match_id` -- the buyer reports what happened after calling the
seller directly (this API is never in that call, never sees payment),
the seller independently confirms the same match_id, and only when both
sides agree does a buyer's reputation tier advance. One real dispute
resets a key straight back to "new" rather than being averaged away.
Neither side has to trust the other's self-report alone.

It's also reachable as a second, non-interoperable protocol entirely:
Google didn't join ACP, it built UCP separately (launched 2026-01-11
with Shopify/Etsy/Wayfair/Target/Walmart). Same catalog, same search,
`/.well-known/ucp` + `/ucp/catalog/search` reusing the identical
underlying implementation -- not a second one that could drift.

The whole thing is dogfooded for real: OBSERVE lists itself in its own
marketplace (not a demo fixture), with a real ACP checkout endpoint of
its own (`POST /v1/commerce/checkout_sessions`) that creates an account
and a live Stripe session when another agent's search finds it.

There's a separate semantic code search API underneath all of this too
(the original project this was built on) -- pay-per-query search over
29 real open source repos, or your own private repo. I won't claim it
beats the dedicated code-search-for-agents tools already out there
(Semble specifically, which measured real, credible token-efficiency
numbers this year) -- I tried three techniques from that space on
OBSERVE's own 759k-chunk corpus (function-boundary chunking, BM25+dense
fusion, a definition-vs-reference reranking boost) and measured all
three as net negative, documented honestly in the README rather than
quietly dropped. Code search here is solid, not the differentiated part.

A dashboard exists now too (https://api.observe-search.online/dashboard)
for setting this up without curl -- register sellers, create reusable
buyer-agent profiles, bulk-import from JSON, or generate a ready-to-use
snippet for whatever's actually running your agent (LangChain, CrewAI,
raw HTTP).

Pricing: $5 for 50,000 credits either way (search or commerce), refunded
automatically on a failed call. New keys start with a small free trial
balance. Listing a product is free, same as OpenAI's own ACP model.

[link to landing page: https://api.observe-search.online/]
[link to GitHub repo]

Feedback very welcome, especially "this trust model doesn't actually
work because X" -- the reputation system is the newest and least
battle-tested part of this, and I'd rather hear the real gap now.

## Notes for whoever posts this

- HN specifically rewards "I built this, here's what's real and what
  isn't" over any hint of hype -- keep the "I tried this and it didn't
  work" paragraph about code search intact, don't polish it out. It's
  the single most HN-credible sentence in the post and preempts the
  "so is this actually better than $COMPETITOR" question before someone
  asks it.
- Post during US morning/early afternoon ET on a weekday for visibility;
  avoid Friday afternoon/weekend.
- Expect the first comments to poke at: (a) whether the reputation
  system can be gamed by a buyer key colluding with a seller key it also
  controls -- now has a real, disclosed partial mitigation:
  VERIFIED_MIN_DISTINCT_SELLERS requires "verified"-tier confirmations
  to span more than one distinct seller, so a single colluding pair
  tops out at "trusted," not "verified," no matter how many fake
  fulfillments they confirm with each other. Honest framing if asked:
  this raises the cost (a determined operator can still stand up
  multiple fake seller identities, no KYC exists in v1), it doesn't
  eliminate the gap -- say that directly rather than overclaiming it's
  solved; (b) whether "self-reported, independently confirmed by both
  sides" is meaningfully different from a simple two-party escrow --
  the honest answer is it's a trust
  *signal*, not a payment guarantee, OBSERVE never touches the money;
  (c) pricing, same as before.
- The original code-search-focused draft is preserved in git history
  (this file's prior version) if a future pivot wants to lead with that
  story again instead.
