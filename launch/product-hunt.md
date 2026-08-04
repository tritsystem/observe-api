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
Semantic code search for AI agents, priced under what it saves
```

## Description

```
Pay-per-query semantic code search over 29 real open source repos
(React, Django, Rails, Spring Boot, Tokio, and more) -- built for AI
agents to call directly, not for a person clicking through a pricing
page.

The honest part: I tried fine-tuning the embedding model on real code
before shipping, fixed two real bugs in the training pipeline first,
and it still measured worse than the stock model on real retrieval --
so this runs stock all-MiniLM-L6-v2, not a fine-tune. Same story with
index compression: ternary quantization measured a 40% top-1 flip
rate, so the index ships at full precision instead of the smaller,
flashier option.

$5 buys 50,000 searches (~$0.0001 each) -- priced under the token cost
a search saves an agent, not around infrastructure cost. New keys get
free trial credits, no payment needed to try it. Point it at your own
private repo too (/v1/private/index) if the 29 curated ones aren't
what you need.

API key + prepaid credits, fully automated signup-to-search, available
as raw HTTP, an MCP server (Claude Code/Desktop, Cursor), and
LangChain/CrewAI tool packages.
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
