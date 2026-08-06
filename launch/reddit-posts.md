# Reddit drafts — r/LocalLLaMA, r/MachineLearning, r/SideProject

Post these yourself under your own account -- not doing this on your
behalf. Stagger by a few days each per marketing-plan.md's own sequencing
advice: same audience overlap problem as posting identical copy
same-day. r/LocalLLaMA first (closest audience match), r/MachineLearning
a few days later (most unforgiving of anything that reads like
marketing -- the honesty angle has to carry it), r/SideProject last (most
tolerant of a direct "I launched X" framing).

Content below reflects the FULL current feature set, including what
shipped after show-hn.md was last written: Ed25519/JWS-signed receipts
on every match/feedback event (independently verifiable against a
published public key, no OBSERVE trust required), quote-binding (locks a
listing's price with a real expiry and a signed receipt), and payment-
rail-agnostic seller registration (real Bech32/Base58Check-validated
Bitcoin/Lightning/stablecoin addresses, still never touching payment
itself). If posting show-hn.md too, consider updating it to match rather
than posting two launches with different feature pictures.

---

## r/LocalLLaMA

**Title:**
```
Built the marketplace-discovery layer ACP and UCP both explicitly punt on -- plus signed receipts and quote-binding, still never touching payment
```

**Body:**

Posting here because this sub is the closest match for anyone actually
running agent stacks that might need this.

The Agentic Commerce Protocol (OpenAI/Stripe) and Google's UCP both
define checkout between a buyer-agent and ONE already-known merchant.
Read both specs directly before building anything, not secondhand
summaries: ACP's own OpenAPI spec has no seller_id field at all, and
explicitly leaves multi-seller discovery to "the marketplace layer above
this API." UCP has the identical gap. Neither says how a buyer-agent
finds sellers across many merchants, or decides whether to trust one it's
never dealt with.

What I built: sellers register a real ACP checkout endpoint + listings
(free), buyer-agents search by intent, every match gets a shared
`match_id` both sides can independently confirm against -- a buyer's
reputation only advances when the SELLER also confirms the same real
event, not from a one-sided claim. Added a real collusion mitigation
after realizing the obvious hole: one person controlling both a buyer key
and a seller key could confirm fake transactions with only each other --
now "verified" tier requires confirmations from 2+ *distinct* sellers,
not just N confirmations. Disclosed honestly: raises the cost, doesn't
eliminate it (no KYC in v1).

Newer stuff: every match/feedback event now gets an Ed25519-signed
receipt (JWS) -- verifiable by anyone against a published public key,
without trusting my API or database at all. And quote-binding: lock a
listing's price with a real expiry before acting on it, so a seller can't
silently change price between when your agent sees a match and when it
acts. Sellers can also register a Bitcoin/Lightning/stablecoin payment
option now (real Bech32/Base58Check checksum validation, not a regex) --
still never touches the actual payment, same trust boundary as the
checkout URL always had.

I did NOT try to compete with dedicated code-search-for-agents tools on
token efficiency (there's a separate semantic code search API in the same
project) -- tested three techniques from that space (function-boundary
chunking, RRF fusion, a definition-boost reranker) against a real 759k-
chunk corpus, all three measured net negative, documented in the README
instead of dropped quietly. Not the differentiated part, said plainly.

Dashboard for setting this up without curl, MCP server, LangChain/CrewAI
tools, raw HTTP -- whatever's actually running your agent.

[link to landing page] / [link to dashboard]

Genuinely want the reputation model pushed on -- it's the least
battle-tested part of this.

---

## r/MachineLearning

**Title:**
```
[P] A verifiable (Ed25519-signed) reputation layer for AI agent commerce, with an honest negative-results writeup on the retrieval side
```

**Body:**

Sharing because the honest-negative-results part seems like exactly what
this sub actually wants over marketing copy.

The project has two halves. The newer, actually-differentiated one: a
cross-merchant discovery + reputation layer for agent commerce (filling a
real gap both the Agentic Commerce Protocol and Google's UCP leave
explicitly undefined in their own specs -- confirmed by reading the specs
directly, not assuming). Every match and feedback event gets a real
Ed25519/JWS-signed receipt, independently verifiable against a published
public key -- not a promise, an actual cryptographic artifact anyone can
check without trusting my database.

The older half is a semantic code search API, and this is the part I
think is actually relevant to this sub: I tried closing the gap with a
credible competing tool's technique set (function-boundary/AST chunking,
BM25+dense RRF fusion, a definition-vs-reference reranking signal) on a
real 759k-chunk, 29-repo corpus. All three measured **net negative**
against what's already deployed (a simpler retrieve-then-rerank
approach), for related reasons -- mostly a cross-repo contamination
failure mode a shared multi-repo index is specifically exposed to.
Documented with the actual win/regression/tie counts in the README
instead of quietly dropped, same as two earlier fine-tuning/quantization
experiments that also measured worse than the shipped baseline.

Not claiming either half is the last word -- posting the reputation
model specifically because I want it stress-tested by people who'd
actually try to break it.

[link] -- full negative-results writeup is in the repo's README, not just
summarized here.

---

## r/SideProject

**Title:**
```
Launched the missing piece of AI agent commerce -- cross-merchant discovery + a reputation system neither OpenAI's nor Google's protocol defines
```

**Body:**

Been heads-down on this and just got it to a state I'm comfortable
sharing.

The pitch: AI agents are starting to buy things (ChatGPT via OpenAI's
Agentic Commerce Protocol, Google's UCP with Shopify/Etsy/Target/Walmart
already on board) -- but both protocols only handle checkout with ONE
merchant a buyer-agent already knows about. Neither says how an agent
*finds* sellers across many merchants, or trusts one it's never dealt
with. I read both real specs to confirm that gap exists before building
anything.

Built the missing layer: free seller listings, semantic search by buyer
intent, and a two-sided reputation system where a buyer's trust tier only
advances when an independent seller confirms the same real transaction --
not a one-sided claim. It's dogfooded for real: the product lists
*itself* in its own marketplace, with a real checkout endpoint that
creates an account and a live Stripe session when another agent finds it
searching.

Newer additions: cryptographically signed (Ed25519) receipts on every
real event so a third party never has to trust my server, price-locking
with a real expiry so a seller can't quietly change terms mid-flow, and
Bitcoin/Lightning/stablecoin payment options for sellers (still never
touching the actual payment -- that boundary was there from day one and
stayed).

Free trial credits, no payment needed to try it, a dashboard for setting
it up without curl.

[link to landing page]

Would love feedback from anyone who's tried building something adjacent
to this -- especially where the trust model would actually break in
practice.
