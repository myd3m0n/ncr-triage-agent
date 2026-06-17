import json, os, re
from collections import Counter
from pathlib import Path
from classifier import classify
ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "eval_set.json"
NCR_FILE = ROOT / "data" / "ncrs.jsonl"
# Fixed MRB disposition vocabulary — same set src/generate_ncrs.py emits. Keep in sync.
DISPOSITIONS = ["Use-As-Is", "Rework", "Repair", "Scrap", "Return to Vendor", "Use Under Deviation"]
# k neighbors used for the RAG disposition prediction (configurable; default 3).
K = int(os.getenv("EVAL_K", "3"))
def main():
    cases = json.loads(EVAL_FILE.read_text())
    total = len(cases); correct = 0; misses = []
    print(f"Running eval on {total} cases...\n")
    for i, case in enumerate(cases, 1):
        predicted = classify(case["input"])["defect_type"]
        expected = case["expected_defect_type"]
        ok = predicted == expected; correct += ok
        print(f"{'OK ' if ok else 'XX '}[{i:>2}] {case['id']}  pred={predicted!r}  exp={expected!r}")
        if not ok: misses.append((expected,))
    acc = correct/total if total else 0
    print(f"\n=== Accuracy: {correct}/{total} = {acc:.0%} ===")
    if misses:
        print("\nMisses by expected type:")
        for t, n in Counter(m[0] for m in misses).most_common():
            print(f"  {n}  {t}")
    disposition_eval()


# ---- disposition eval (RAG end-to-end) -------------------------------------------
# Predicts each eval NCR's disposition by retrieving its nearest past NCRs (with the
# source NCR excluded to prevent leakage) and letting the LLM choose one disposition
# from the precedent. Reported alongside two no-LLM baselines for context.

DISPO_SYSTEM = (
    "You are an MRB (Material Review Board) disposition engineer at a precision "
    "optics/photonics manufacturer. Given a new nonconforming material report (NCR) "
    "and how the most similar past NCRs were dispositioned, choose the single most "
    "appropriate disposition. Weigh the precedent of the similar past cases heavily. "
    "Never invent categories."
)


def _extract_json(text):
    cleaned = re.sub(r"```(json)?", "", text.strip()).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))


def _coerce_disposition(value):
    d = str(value).strip()
    if d in DISPOSITIONS:
        return d
    for label in DISPOSITIONS:  # tolerant case-insensitive match
        if label.lower() == d.lower():
            return label
    return "Unknown"


def _rag_predict(query, neighbors):
    """Full RAG loop: show the LLM the query + neighbor dispositions, get one disposition."""
    from llm import complete
    precedent = "\n".join(
        f'{j}. (sim {nb["similarity"]:.2f}) disposition="{nb["disposition"]}" :: {nb["description"]}'
        for j, nb in enumerate(neighbors, 1)
    )
    allowed = "\n".join(f"- {d}" for d in DISPOSITIONS)
    prompt = (
        "A new NCR needs an MRB disposition.\n\n"
        "Choose the disposition from EXACTLY this list (copy the label verbatim):\n"
        f"{allowed}\n\n"
        "Most similar past NCRs and how they were dispositioned (your precedent):\n"
        f"{precedent}\n\n"
        "New NCR description:\n"
        f'"""{query}"""\n\n'
        "Respond with ONLY a JSON object, no preamble, no markdown:\n"
        '{"disposition": "..."}'
    )
    raw = complete(prompt, system=DISPO_SYSTEM, temperature=0.0)
    try:
        obj = _extract_json(raw)
    except Exception:
        return "Unknown"
    return _coerce_disposition(obj.get("disposition", ""))


def _neighbor_vote(neighbors):
    """Majority disposition among neighbors; ties broken toward the nearest neighbor."""
    counts = Counter(nb["disposition"] for nb in neighbors)
    top = max(counts.values())
    tied = {d for d, c in counts.items() if c == top}
    if len(tied) == 1:
        return next(iter(tied))
    for nb in neighbors:  # neighbors are most-similar-first
        if nb["disposition"] in tied:
            return nb["disposition"]
    return next(iter(tied))


def disposition_eval():
    print("\n\n" + "=" * 64)
    print(f"Disposition eval — RAG end-to-end (k={K})")
    print("=" * 64)
    try:
        from retriever import retrieve
    except Exception as exc:
        print(f"  [skipped] disposition eval needs the .venv (chromadb): {exc!r}")
        print("  run:  cd src && ../.venv/bin/python run_eval.py")
        return
    cases = json.loads(EVAL_FILE.read_text())
    corpus = [json.loads(line) for line in open(NCR_FILE) if line.strip()]
    by_id = {r["ncr_id"]: r for r in corpus}
    majority_disp = Counter(r["disposition"] for r in corpus).most_common(1)[0][0]
    # Over-fetch generously: eval inputs are EXACT corpus text, and the corpus reuses the
    # same description across different ncr_ids, so the query's verbatim duplicates (the
    # self NCR AND identical-text siblings) all retrieve at similarity ~1.0. Filtering self
    # by ncr_id alone would leave those siblings in — leaking the answer directly. We exclude
    # every verbatim copy of the query; the largest such cluster is small (<=6), so K+20
    # leaves ample room for K genuinely-different precedent NCRs.
    n_fetch = min(len(corpus), K + 20)

    rows = []
    self_in_raw = sib_in_raw = sib_filtered = leak_in_used = 0
    for case in cases:
        cid, query = case["id"], case["input"]
        true = by_id[cid]["disposition"]
        raw_hits = retrieve(query, k=n_fetch)
        if any(h["ncr_id"] == cid for h in raw_hits):
            self_in_raw += 1
        sibs = [h for h in raw_hits if h["ncr_id"] != cid and h["description"] == query]
        if sibs:
            sib_in_raw += 1
            sib_filtered += len(sibs)
        # Exclude self AND any verbatim copy of the query: precedent must be a *different* NCR.
        used = [h for h in raw_hits if h["ncr_id"] != cid and h["description"] != query][:K]
        if any(h["ncr_id"] == cid or h["description"] == query for h in used):
            leak_in_used += 1
        neighbors = [{
            "ncr_id": h["ncr_id"], "description": h["description"], "similarity": h["similarity"],
            "disposition": by_id.get(h["ncr_id"], {}).get("disposition", "Unknown"),
        } for h in used]
        rows.append((cid, true, _rag_predict(query, neighbors), _neighbor_vote(neighbors)))

    n = len(rows)
    print(f"\nLeakage check (raw top-{n_fetch} per case):")
    print(f"  self NCR (same ncr_id) present in raw      : {self_in_raw}/{n}")
    print(f"  verbatim-dup siblings present in raw       : {sib_in_raw}/{n}  ({sib_filtered} sibling hits filtered)")
    print(f"  direct-answer leaks left in USED neighbors : {leak_in_used}/{n}  (must be 0)")
    print("  -> neighbors exclude self AND any identical-description NCR; precedent is always a different case")

    print("\nPer case — RAG predicted vs true disposition:")
    rag_c = vote_c = maj_c = 0
    for cid, true, rag, vote in rows:
        ok = rag == true
        rag_c += ok; vote_c += (vote == true); maj_c += (majority_disp == true)
        print(f"  {'OK ' if ok else 'XX '} {cid}  pred={rag!r}  true={true!r}")

    print(f"\n=== Disposition accuracy (full RAG): {rag_c}/{n} = {rag_c/n:.0%} ===")

    print("\nPer-true-disposition breakdown (correct / total, with predicted spread):")
    for t in sorted({r[1] for r in rows}):
        preds = Counter(rag for _, tt, rag, _ in rows if tt == t)
        correct = preds.get(t, 0)
        spread = ", ".join(f"{c}x {p}" for p, c in preds.most_common())
        print(f"  {t}: {correct}/{sum(preds.values())}  [{spread}]")

    print("\n3-way comparison (same leakage-filtered neighbors):")
    print(f"  majority-class (always {majority_disp!r}) : {maj_c}/{n} = {maj_c/n:.0%}")
    print(f"  neighbor-vote  (k={K}, no LLM)            : {vote_c}/{n} = {vote_c/n:.0%}")
    print(f"  full RAG       (LLM + neighbors)          : {rag_c}/{n} = {rag_c/n:.0%}")


if __name__ == "__main__":
    main()
