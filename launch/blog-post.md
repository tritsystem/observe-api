# Blog post draft: "Pricing an API for machines, not people"

Post this under your own name/site (dev.to, Hashnode, personal blog,
wherever) -- draft only, edit before publishing. Longer and more
technical than the Show HN post; goes deeper on the pricing reasoning
and the engine itself, aimed at developers building agents rather than
a general HN audience.

---

I've spent a while building OBSERVE, a semantic code search tool that
runs entirely locally -- fine-tuned embedding model, ternary-quantized
index, FAISS search. Every number in its README is from a written-up,
reproducible benchmark, including the ones that make it look bad: grep
still beats it when you already know the exact function name you're
looking for. It only earns its keep on the other kind of query -- "where
does this handle retrying a failed upload" -- where you can describe the
thing but not name it.

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

## What's actually available

- A raw HTTP API: `/v1/signup`, `/v1/search`, `/v1/balance`, `/v1/repos`
- An MCP server, so it drops into Claude Code, Claude Desktop, or Cursor
  as a tool
- LangChain and CrewAI wrappers for agents built on those frameworks
- Fifteen real repos indexed to start (React, Django, NumPy, Tokio,
  and more), refreshed from their default branch

## What it isn't (yet)

One shared index, not per-customer -- if you want to search *your own*
private code through this, that's a real multi-tenant isolation feature
that doesn't exist yet, not something I'm pretending is already solved.
No rate limiting beyond the credit balance itself. One fixed credit
package, no tiers. All disclosed in the repo's README, not hidden
behind a "contact sales" wall.

[link to landing page]
[link to GitHub repo]

Feedback welcome -- especially on the pricing model, since it's the part
I'm least sure is right.
