# Product Hunt draft

Post this yourself at producthunt.com/posts/new under your own account --
I'm not going to post on your behalf. Draft only, edit before publishing.
Different launch day than Show HN (see marketing-plan.md's Week 1 note --
don't split attention across both platforms same day).

## Name

```
OBSERVE Search API
```

## Tagline (under 60 chars)

```
Discovery + reputation for AI agent commerce, plus code search
```

## Description

```
A real ACP and Google UCP compatible marketplace layer for AI
buyer-agents: sellers list for free, buyers search by intent, and a
two-sided reputation system means trust is earned agreement between
buyer and seller, not a self-report either side could inflate alone.
OBSERVE even lists itself in its own marketplace as a working example.
A dashboard (no curl required) lets you set up sellers and buyer-agent
profiles, bulk-import both from JSON, or grab a ready snippet for
LangChain, CrewAI, or raw HTTP.

Also includes the original product underneath: pay-per-query semantic
code search over 29 real open source repos (React, Django, Rails,
Spring Boot, Tokio, and more), or your own private repo.

The honest part: I tried fine-tuning the embedding model on real code
before shipping, fixed two real bugs in the training pipeline first,
and it still measured worse than the stock model on real retrieval --
so this runs stock all-MiniLM-L6-v2, not a fine-tune. Same story with
a competing tool's chunking/fusion/reranking techniques, tried against
this project's own corpus after they'd worked well elsewhere: net
negative, three separate times, documented instead of dropped quietly.

$5 buys 50,000 credits (~$0.0001 each), spent on either search or
commerce lookups -- priced under the token cost a call saves an agent,
not around infrastructure cost. New keys get free trial credits, no
payment needed to try it.

API key + prepaid credits, fully automated signup-to-search, available
as raw HTTP, an MCP server (Claude Code/Desktop, Cursor), LangChain/
CrewAI tool packages, A2A, ACP, and UCP.
```

## First comment (post immediately after, from your own account)

```
Maker here. Happy to answer anything, especially "this pricing/positioning
doesn't make sense because X" -- genuinely want to hear it before assuming
the model's right. The two negative results in the description (fine-tuning,
quantization) are real and reproducible if anyone wants specifics on either.
```

## Topics/categories to tag

Developer Tools, Artificial Intelligence, API, Open Source

## Notes

- Screenshot/gallery: use the landing page itself (https://api.observe-search.online/)
  as the first image -- PH listings with zero visuals get far less traffic than
  ones with even a simple screenshot.
- Post on a different day than Show HN (see marketing-plan.md), ideally also
  a weekday morning ET -- PH's daily ranking resets at midnight PT, so an
  early post (12:01 AM PT) gets the longest visibility window, unlike HN
  where "morning ET" is what matters. These optimize for different clocks;
  don't apply HN's timing advice to PH.
