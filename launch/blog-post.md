# Blog post draft: "Pricing an API for machines, not people"

Post this under your own name/site (dev.to, Hashnode, personal blog,
wherever) -- draft only, edit before publishing. Longer and more
technical than the Show HN post; goes deeper on the pricing reasoning
and the engine itself, aimed at developers building agents rather than
a general HN audience.

---

I've spent a while building OBSERVE, a semantic code search tool that
runs entirely locally -- embedding-based search, FAISS. Every number in
its README is from a written-up, reproducible benchmark, including the
ones that make it look bad: grep still beats it when you already know the
exact function name you're looking for. It only earns its keep on the
other kind of query -- "where does this handle retrying a failed upload"
-- where you can describe the thing but not name it.

Before shipping the hosted version I tried fine-tuning the embedding
model on real code, since that's the obvious next differentiator. Found
and fixed two real bugs in the original training pipeline first (both
GitHub-streaming data sources it relied on are dead, and its own
validation metric was silently broken -- a constant label array making
Pearson/Spearman correlation undefined, not a training failure).
Retrained clean on real data. It still came out *worse* than the stock
embedding model on actual retrieval -- 5% result overlap with stock, 0/20
exact top-1 matches, several wrong-repo results, on the same 20-query
benchmark. Root cause looks like a mismatch between training-anchor style
(function/class names, raw comments) and real query style (full
natural-language questions). So this ships on stock all-MiniLM-L6-v2, not
a fine-tune -- I'd rather undersell it than claim a differentiator that
measured worse than the baseline.

Recently I turned it into a hosted API, because the more interesting
question wasn't "is this useful" (the benchmarks already answered that)
but "who's the customer when the caller is an AI agent, not a person
clicking through a pricing page."

## The pricing problem is different when your customer doesn't get tired

A human evaluating a $10/month tool weighs it against a vague sense of
"is this worth it." An agent (or more precisely, whoever's paying for
the agent's API calls) can actually do the math: does spending $X on
this call save more than $X worth of tokens on the alternative.

OBSERVE's own benchmark says its hosted successor saves about 66% of the
tokens a plain search would cost on the same query. At typical LLM API
pricing -- a few dollars per million tokens -- a few thousand tokens
saved is worth a fraction of a cent. If I priced each search at even a
cent, I'd already be eating most or all of that saving myself. The
product would be *correctly* not worth using, and no amount of
marketing changes that math.

So the actual pricing decision wasn't "what does this cost me to run"
(a search's marginal compute cost is close to zero once the embedding
model is warm in memory) -- it was "what's the ceiling implied by the
value it delivers." I landed on $5 for 50,000 credits: about 0.0001
cents per search, one to two orders of magnitude under the token cost
being saved. Fixed costs -- the always-on host, the model resident in
memory -- get covered by volume, the same way any infrastructure with
near-zero marginal cost per unit does. That's not a new idea, but it's
one that's easy to skip when you default to pricing around your own
costs instead of the buyer's alternative.

## What actually differentiates this from GitHub's own code search

Not smarter ranking -- I tested that claim and it didn't hold up (see
above). What's real: GitHub's code search API can't reach several of the
most popular repos at all. I checked all 15 in this corpus directly --
react, django, fastapi, cargo, and tokio return zero results from
GitHub's search, even for the literal word "the" (large/high-traffic
repos get excluded from their index). For the repos it *can* reach, a
natural-language question returns nothing unless you manually strip it
down to bare keywords -- the API treats a normal question as a literal
phrase match. And the rate limit is tight enough (~6-10 requests/minute)
that I hit a 403 partway through running this exact comparison. None of
that is about AI quality; it's coverage, query format, and being built
for something that calls it in a loop rather than a person typing a
handful of searches by hand.

## What's actually available

- A raw HTTP API: `/v1/signup`, `/v1/search`, `/v1/balance`, `/v1/repos`
- An MCP server, so it drops into Claude Code, Claude Desktop, or Cursor
  as a tool
- LangChain and CrewAI wrappers for agents built on those frameworks
- Twenty-nine real repos indexed to start (React, Django, NumPy, Tokio,
  and more), refreshed from their default branch, plus on-demand private
  indexing of any repo you point it at

## What it isn't (yet)

One shared index, not per-customer, for the *public* repo catalog --
private indexing exists (`/v1/private/index`) and is isolated per API
key, but that's a real one-time compute cost, priced differently from a
marginal search. One fixed credit package, no tiers. All disclosed in
the repo's README, not hidden behind a "contact sales" wall.

## The same pricing logic, applied to a second problem

Since writing the above, the same "what does the buyer's alternative
actually cost" question came up again, in a different shape: the
Agentic Commerce Protocol (OpenAI + Stripe) and Google's UCP both define
checkout between a buyer-agent and one merchant it already knows about.
Neither says how a buyer-agent *finds* sellers across many merchants, or
decides whether to trust one it hasn't dealt with before -- I read both
specs directly to confirm this wasn't just an impression, and both
explicitly punt that problem to "the marketplace layer above this API."

That layer is now a real part of this project too: sellers list for
free (same logic as OpenAI's own ACP model -- discovery costs the
platform nothing marginal, so it shouldn't cost the seller anything
either), buyers search by intent using the exact same embedding pipeline
described above, and a shared `match_id` lets both sides of a real
transaction independently confirm what happened -- so a buyer's
reputation is earned agreement between two disconnected parties, not a
self-report either side could inflate alone. It's reachable as both ACP
and UCP from one implementation, and it's dogfooded for real: this
product lists itself in its own marketplace, with its own real ACP
checkout endpoint.

I did try to close the code-search gap with a real competitor
(Semble, `MinishLab/semble` -- tree-sitter chunking + a static embedding
model + BM25/RRF, a credible ~98%-fewer-tokens-than-grep claim that got
real traction on HN this year) before writing this. Three of its
techniques, tested against OBSERVE's own 759k-chunk corpus: function-
boundary chunking, RRF fusion, and a definition-vs-reference reranking
signal. All three measured net negative, for related reasons -- mostly
a cross-repo contamination failure mode a shared multi-repo index is
specifically exposed to. Documented honestly in the README instead of
quietly shelved. I'd rather this post undersell code search than
overclaim it; the commerce/reputation layer is the newer, less
contested part of the pitch.

[link to landing page]
[link to GitHub repo]

Feedback welcome -- especially on the pricing model, since it's the part
I'm least sure is right.
