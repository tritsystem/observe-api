"""
Real re-fine-tune, fixing both root causes found in the original
012-trit-search/trit_embed_train.py:

  1. Training data source: the original streamed from bigcode/the-stack-smol
     (gated, needs HF auth the script never did) and codeparrot/github-code
     (uses a dataset-script format HF's `datasets` lib no longer supports).
     Both failed silently for every language -> empty pairs_*.json, training
     ran on local-repo-only pairs (narrow, self-referential). Fix: mine pairs
     from the 15 real, diverse, already-cloned repos in ./repos instead --
     no external dataset dependency, no availability risk.

  2. Evaluator: the original fed EmbeddingSimilarityEvaluator a CONSTANT
     label array ([1.0] * N, "all pairs are positive") -- correlation against
     a constant is undefined, guaranteed NaN regardless of model quality.
     Fix: no built-in evaluator during training (avoids reintroducing a
     subtle bug); real validation happens after training via an actual
     self-retrieval check on held-out pairs, plus the same real 20-query
     benchmark already used to test stock vs. the original fine-tune.

extract_pairs_from_code() below is copied verbatim from trit_embed_train.py
-- that heuristic pair-mining logic (function name->body, comment->code,
class->body) has no identified bug, only its data source and evaluator did.
"""
import json
import os
import random
import re
import time
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH  = Path("./models/code-minilm-v2")
BATCH_SIZE = 64     # bumped from the original's 32 -- more in-batch negatives
                    # for MultipleNegativesRankingLoss, real GPU headroom (8GB, tiny model)
EPOCHS         = 3
WARMUP_STEPS   = 100
LR             = 2e-5

EXT_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".cs": "C#",
    ".rs": "Rust", ".gd": "GDScript", ".go": "Go", ".c": "C", ".cpp": "C++",
    ".php": "PHP", ".java": "Java",
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             "dist", "build", "target", "models", "search_index"}


# ---- verbatim from trit_embed_train.py ----
def extract_pairs_from_code(code: str, lang: str) -> list:
    pairs = []
    lines = code.splitlines()

    func_patterns = [
        r'(?:def|func|fn|function|fun|sub|proc)\s+(\w+)\s*\(',
        r'(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\(',
    ]
    for pat in func_patterns:
        for i, line in enumerate(lines):
            m = re.search(pat, line)
            if m:
                fname = m.group(1)
                if len(fname) < 3 or fname in ('if', 'for', 'while', 'return'):
                    continue
                body_lines = lines[i:i+15]
                body = "\n".join(body_lines).strip()
                if len(body) > 50:
                    readable = re.sub(r'([A-Z])', r' \1', fname)
                    readable = readable.replace('_', ' ').lower().strip()
                    pairs.append((readable, body))
                    pairs.append((fname, body))

    comment_patterns = [
        r'^\s*#\s*(.+)$',
        r'^\s*//\s*(.+)$',
        r'^\s*--\s*(.+)$',
    ]
    for i, line in enumerate(lines):
        for pat in comment_patterns:
            m = re.match(pat, line)
            if m:
                comment = m.group(1).strip()
                if len(comment) < 10 or len(comment) > 200:
                    continue
                rest = "\n".join(lines[i+1:i+10]).strip()
                if len(rest) > 30:
                    pairs.append((comment, rest))
                break

    class_patterns = [
        r'class(?:_name)?\s+(\w+)',
        r'(?:class|struct|interface)\s+(\w+)',
    ]
    for i, line in enumerate(lines):
        for pat in class_patterns:
            m = re.search(pat, line)
            if m:
                cname = m.group(1)
                body = "\n".join(lines[i:i+20]).strip()
                if len(body) > 60:
                    readable = re.sub(r'([A-Z])', r' \1', cname).lower().strip()
                    pairs.append((f"{readable} class", body))
                break

    for line in lines:
        m = re.match(r'^\s*signal\s+(\w+)', line)
        if m:
            sname = m.group(1).replace('_', ' ')
            pairs.append((f"signal {sname}", line.strip()))

    seen, result = set(), []
    for anchor, positive in pairs:
        key = anchor[:50]
        if key in seen:
            continue
        seen.add(key)
        if len(anchor) < 4 or len(anchor) > 300:
            continue
        if len(positive) < 20:
            continue
        result.append((anchor, positive))
    return result
# ---- end verbatim section ----


def extract_from_repos(scan_dirs):
    all_pairs = []
    per_repo = {}
    for base in scan_dirs:
        repo_pairs = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext not in EXT_MAP:
                    continue
                path = os.path.join(root, f)
                try:
                    code = open(path, encoding="utf-8", errors="ignore").read()
                    if len(code.strip()) < 100:
                        continue
                    new_pairs = extract_pairs_from_code(code, EXT_MAP[ext])
                    all_pairs.extend(new_pairs)
                    repo_pairs += len(new_pairs)
                except Exception:
                    pass
        per_repo[base] = repo_pairs
        print(f"[extract] {base}: {repo_pairs:,} pairs", flush=True)
    return all_pairs


def main():
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import numpy as np

    manifest = json.load(open("repo_manifest.json"))
    scan_dirs = list(manifest.values())

    print("[retrain] extracting pairs from the 15 real repos...", flush=True)
    all_pairs = extract_from_repos(scan_dirs)
    print(f"[retrain] total pairs: {len(all_pairs):,}", flush=True)

    if len(all_pairs) < 100:
        raise RuntimeError("not enough pairs extracted -- aborting")

    random.seed(0)
    random.shuffle(all_pairs)

    split = int(len(all_pairs) * 0.95)
    train_pairs = all_pairs[:split]
    eval_pairs = all_pairs[split:][:500]  # cap eval pool, it's just a sanity check
    print(f"[retrain] train: {len(train_pairs):,}  eval (held-out): {len(eval_pairs):,}", flush=True)

    examples = [InputExample(texts=[a, p]) for a, p in train_pairs]

    print(f"[retrain] loading base model {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(loader) * EPOCHS
    print(f"[retrain] training {EPOCHS} epochs ({total_steps:,} steps, batch={BATCH_SIZE})...", flush=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={"lr": LR},
        output_path=str(SAVE_PATH),
        show_progress_bar=True,
    )
    print(f"[retrain] training done in {time.time()-t0:.1f}s, saved to {SAVE_PATH}", flush=True)

    # Real, bug-free sanity check: for each held-out anchor, does its true
    # positive rank #1 among the full eval-pool of positives (including
    # other pairs' positives as distractors)? No constant-label correlation
    # involved -- this has actual variance and actual right/wrong answers.
    print("\n[retrain] held-out self-retrieval sanity check...", flush=True)
    anchors = [a for a, _ in eval_pairs]
    positives = [p for _, p in eval_pairs]
    a_vecs = model.encode(anchors, normalize_embeddings=True, show_progress_bar=False)
    p_vecs = model.encode(positives, normalize_embeddings=True, show_progress_bar=False)
    sims = a_vecs @ p_vecs.T
    ranks = np.argmax(sims, axis=1)
    correct = int(np.sum(ranks == np.arange(len(anchors))))
    print(f"[retrain] self-retrieval top-1 accuracy: {correct}/{len(anchors)} ({correct/len(anchors)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
