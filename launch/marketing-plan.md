# Marketing plan — $0 budget

Every channel below is free (no ad spend). Organized by what's already
built vs. what's new, then sequenced into weeks. Cross-references the
other files in this folder rather than repeating them.

## Already built (see the other files in launch/)

- `show-hn.md` — Show HN post, ready to post under your own account
- `blog-post.md` — longer technical writeup for dev.to/Hashnode/personal blog
- `registry-submissions.md` — awesome-mcp-servers, Smithery, mcp.so, official
  MCP registry, LangChain integrations docs, PyPI

Nothing below duplicates those — this file is the channels *not* yet covered,
plus the order to run all of it in.

## Principle this plan follows

Every existing asset leads with a disclosed negative result (fine-tuning
made things worse, quantization flips 40% of top-1s) before the pitch.
Keep that discipline in every new post below — it's the actual
differentiator in a space full of hype, and it's what makes technical
audiences (HN, Reddit, Discord communities) trust a self-promo post
instead of downvoting it on sight.

## New channels

### Communities (post where the actual audience already is)

- **r/LocalLLaMA** — closest audience match (people building/running their
  own agent stacks). Post as "I built X, here's what I measured" not
  "check out my product." Reference the negative results up front.
- **r/MachineLearning** — "Show" / "Project" flair only; this sub is
  unforgiving of pure marketing, the honesty angle matters most here.
- **r/SideProject**, **r/EntrepreneurRideAlong** — fine with a more direct
  "I launched X" framing, different norms than r/MachineLearning.
- **Indie Hackers** — post the launch as a build-in-public log entry, not
  just an announcement; this audience engages more with process than result.
- **LangChain Discord**, **MCP Discord** (if one exists by launch time),
  **Cursor/Windsurf community Discords** — check each server's
  self-promo channel rules before posting; most have a dedicated
  `#showcase` or `#built-with-x` channel, posting outside it gets removed.
- **Dev.to** and **Hashnode** — cross-post `blog-post.md` to both (different
  audiences, zero extra writing cost, both free).

### Directories and listings (beyond the MCP-specific ones already planned)

- **Product Hunt** — free to launch. Prep a short tagline + the same
  honest framing; PH audiences respond well to "here's what didn't work"
  posts because most PH launches are pure hype.
- **awesome-langchain**, **awesome-ai-agents**, **awesome-devtools**,
  **awesome-python** (if it fits their scope) — same PR-based submission
  pattern as `registry-submissions.md`'s awesome-mcp-servers entry.
- **AlternativeTo**, **G2** (free listing tiers) — lower priority, but
  zero cost and picked up by some search traffic.
- **Google Search Console** — submit the domain's sitemap once live;
  free, and the only way organic Google search actually indexes the
  landing page and `/docs` promptly.

### Distribution to people who cover this space (free — pitching, not paying)

- **AI-focused newsletters that take community submissions**: TLDR AI,
  Ben's Bites, The Rundown AI, Import AI. Most have a "submit a tool/link"
  form — free, no guarantee of pickup, costs only the time to fill the form.
  Lead with the negative-result framing in the one-line pitch; it's what
  gets picked over generic "new AI tool" submissions.
- **Open an issue/PR against agent framework repos** offering the
  integration directly — LangChain, CrewAI, AutoGPT, Continue.dev. This
  overlaps with `registry-submissions.md`'s LangChain docs entry but
  extend it to any framework with a public "community tools" doc or
  Discord.
- **Reply to relevant HN/Reddit threads** (not just your own launch post)
  when someone's asking "how do I get an agent to search a codebase" —
  answer the question first, mention the tool second. This is slower but
  compounds over months and reads as help, not spam.

### Low-effort, high-leverage technical SEO (all free, all one-time setup)

- Make sure `llms.txt` (already correct) and the landing page's meta
  description are crawlable — no login wall, no JS-only rendering blocking
  it (FastAPI serving raw HTML already satisfies this).
- Add the repo topics/tags on GitHub (`mcp`, `semantic-search`,
  `ai-agents`, `code-search`, `langchain-tools`) — free, and GitHub's own
  topic pages are a real discovery surface once starred a few times.
- Pin the repo, write a real README badge row (build status, license,
  PyPI version once published) — costs nothing, measurably increases
  perceived legitimacy for a first-time visitor deciding whether to try it.

## Sequencing (four weeks, assuming a solo operator)

**Week 0 (pre-launch, do before any public post)**
- Confirm the API is stable under real traffic for a day first — a
  Show HN or Product Hunt spike hitting a 502 kills the launch's only
  shot at those front pages.
- Fix the `launch/show-hn.md` "no private indexing yet" issue (already
  done this session) — reread every draft once more right before posting,
  things drift as the product changes.

**Week 1 — the front-page shots (each only works once, so land them well)**
- Show HN (Tuesday-Thursday, US morning ET, per the existing notes in
  `show-hn.md`)
- Product Hunt (same week, different day — don't split attention across
  both on the same day)
- Cross-post `blog-post.md` to dev.to + Hashnode the same week, timed to
  land right after the HN post so early visitors have a deeper writeup
  to land on

**Week 2 — communities**
- r/LocalLLaMA, r/MachineLearning, r/SideProject (stagger by a few days
  each, don't post identical copy to all three same-day — each sub's
  regulars overlap and repetition reads as spam)
- Relevant Discord servers' showcase channels
- Start replying to existing HN/Reddit threads where the tool is a genuine
  answer (ongoing from here on, not just week 2)

**Week 3 — registries and directories**
- Execute everything in `registry-submissions.md`
- awesome-list PRs from this file
- Newsletter submissions (these often have lead times, submit early even
  though pickup may land later)

**Week 4 — retrospective, not a new push**
- Whatever channel actually drove signups (check `/v1/signup` volume by
  referrer if you added UTM params, or just ask new users where they
  found it) — double down there specifically instead of spreading thin
  again. A free-channel launch's real signal only shows up after the
  first wave, not during it.

## What's deliberately not on this list

No paid ads, no Twitter/X ad spend, no growth-hacking (fake urgency,
follower-buying, engagement pods). Not because they don't work — because
this plan is scoped to $0 as asked, and the honesty-first positioning
this product already leans on doesn't pair well with growth tactics that
depend on the audience not looking too closely.
