# Show HN draft

Post this yourself at news.ycombinator.com/submit under your own account --
I'm not going to post to HN on your behalf. Edit freely; this is a
starting draft, not a final copy.

## Title

```
Show HN: OBSERVE Search API – pay-per-query semantic code search for AI agents
```

(HN strips marketing language from titles fast -- this one states what it
is and who it's for, nothing more. Don't add adjectives.)

## Body (as a text post, or link + first comment)

I built OBSERVE (https://github.com/gbranaa4-hue/012-trit-search) as a
local semantic code search tool a while back -- fine-tuned embedding
model, ternary-quantized index (~20x smaller on disk), FAISS search,
function-boundary chunking. Every claim in that repo is backed by a
written-up, reproducible benchmark, including the unflattering one: grep
beats it 5/5 vs 3/5 when you already know the exact identifier you're
looking for. It only wins the vocabulary-mismatch / "describe it, don't
name it" case -- and it says so in its own MCP tool descriptions, so an
AI assistant using it routes queries correctly on its own instead of me
having to explain it every time.

This is that same engine, hosted, with an API key and prepaid credits in
front of it, aimed at AI agents that want to search real open source
code (React, Django, NumPy, Tokio, and a dozen more right now) without
running the embedding model themselves.

The pricing is the part I think is actually interesting: I priced it
under the token cost it saves, not around what it costs me to run. A
search's marginal compute cost is close to zero once the model's warm in
memory -- so $5 buys 50,000 searches (0.01 cents each), because the real
constraint isn't infrastructure, it's whether paying for the call beats
just burning more tokens on exploratory greps.

Available as a raw HTTP API, an MCP server (Claude Code / Claude
Desktop / Cursor), and LangChain/CrewAI tool packages.

[link to landing page / API docs]

Feedback very welcome, especially "this pricing model doesn't make
sense because X" -- I'd rather hear that now.

## Notes for whoever posts this

- HN specifically rewards "I built this, here's what's real and what
  isn't" over any hint of hype -- the draft above leans into that on
  purpose, don't polish the honesty out of it.
- Post during US morning/early afternoon ET on a weekday for visibility;
  avoid Friday afternoon/weekend.
- Expect the first comments to poke at the pricing model and the
  single-shared-index limitation (no private repo indexing yet) --
  both are already disclosed honestly in the README, worth linking to
  directly if asked rather than re-explaining from scratch.
