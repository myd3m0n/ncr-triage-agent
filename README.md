# NCR Triage Agent

An agentic **RAG** system that triages manufacturing **Nonconformance Reports (NCRs)** and recommends **Material Review Board (MRB) dispositions** by reasoning over similar past cases.

> **Status:** Working prototype on synthetic data — built to demonstrate how retrieval-augmented reasoning applies to manufacturing quality decisions. It is intentionally *not* production software; see [Path to Production](#path-to-production).

---

## The problem

When a part fails inspection on a manufacturing floor, it's written up as an NCR. Someone then has to decide what happens to it — the **MRB disposition**: use it as-is, rework it, repair it, scrap it, return it to the vendor, or use it under a documented deviation.

That decision is *judgment*, not a lookup. The same defect type can resolve differently depending on severity, which part it's on, and how it was made — a dimensional part 2 thou over tolerance might be "use under deviation" on one assembly and "scrap" on another. In practice, engineers make the call by recalling how similar past nonconformances were dispositioned.

That recall-and-reason-over-precedent loop is exactly what retrieval-augmented generation is for. This project tests whether a RAG agent can replicate it.

*(Domain note: I worked precision optical/fiber assembly and MRB nonconforming-material documentation at SpaceX, which is where the problem and the synthetic defect patterns come from.)*

---

## Results

Evaluated on a locked 20-case held-out set, with leakage controls (see below):

| Task | Method | Accuracy |
|---|---|---|
| Defect-type classification | Prompt-tuned classifier | **20/20 (100%)** |
| Disposition | **RAG (retrieve neighbors → reason)** | **11/20 (55%)** |
| Disposition | Majority-class baseline | 8/20 (40%) |
| Disposition | Neighbor-vote baseline (no LLM) | 8/20 (40%) |

The headline isn't the absolute 55% — it's that **the RAG loop beats both baselines by 15 points**, which is the evidence that retrieval-plus-reasoning is doing real work rather than guessing the most common answer.

**Leakage-safe evaluation.** The corpus contains 200 NCRs but only 131 distinct descriptions, so naive `ncr_id`-only filtering let verbatim-duplicate siblings retrieve at similarity ≈ 1.0 and hand the model the answer on 14/20 cases. The eval excludes both the source NCR *and* any byte-identical-description NCR, so neighbors are genuinely *different* precedents. Confirmed 0/20 direct-answer leaks.

**Where it's weak (honestly):** the model leans toward `Rework` (the corpus-dominant disposition) and under-predicts rare classes like `Return to Vendor`. That's a known RAG limitation on imbalanced corpora — the fix is more distinct data or class-aware retrieval, not prompt-hacking individual cases.

---

## How it works

```mermaid
flowchart LR
    A[Query NCR] --> B[Embed<br/>nomic-embed-text]
    B --> C[Retrieve k nearest<br/>past NCRs from Chroma]
    C --> D[Leakage filter<br/>drop self + identical text]
    D --> E[LLM picks 1 disposition<br/>from neighbors' history]
    E --> F[Recommended disposition]
```

Each NCR carries `defect_type`, `disposition`, and `root_cause`. Dispositions and root causes were synthesized deterministically from `sha256(ncr_id)` as a *probabilistic function* of defect type, process, and severity — dominant-with-tail, not uniform and not 1:1 — so the data mirrors real MRB decision patterns (e.g. dimensional out-of-tolerance skews toward "use under deviation"; solder defects skew toward "repair").

---

## Stack & key decisions

- **Python**, **Ollama** for local embeddings (`nomic-embed-text`) and LLM calls, **ChromaDB** for the persistent vector store. Everything else is standard library.
- **Chroma over FAISS** — the project needs persistence *and* metadata filtering (to carry disposition/root_cause alongside each vector and to filter neighbors at query time). Chroma gives both out of the box; FAISS would mean bolting on a separate metadata layer.
- **Deterministic synthetic data** — generation is hash-derived and reproducible, so the locked eval set stays byte-identical across regenerations and results are comparable over time.
- **Model-agnostic LLM wrapper** — `llm.py` dispatches on a backend env var, so the embedding/chat models can be swapped without touching application logic.

---

## Repo layout

```
data/
  ncrs.jsonl          # 200 synthetic NCRs (source of truth)
  ncrs.csv            # same, tabular
eval/
  eval_set.json       # locked 20-case held-out eval set
src/
  generate_ncrs.py    # deterministic synthetic-data generator
  llm.py              # model-agnostic embed() + complete() wrapper
  retriever.py        # Chroma index build + k-NN retrieval
  classifier.py       # defect-type classifier
  run_eval.py         # defect-type + disposition RAG eval
  CLAUDE.md           # working notes / project state
```

---

## Running it

Requires a local [Ollama](https://ollama.com) instance. The code is cross-platform (Python + `pathlib`); commands are shown for both macOS/Linux and Windows.

**1. Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

**2. Create the venv and install dependencies**

macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**3. Build the index + run a sample retrieval**

macOS / Linux:
```bash
cd src && ../.venv/bin/python retriever.py
```

Windows (PowerShell):
```powershell
cd src; ..\.venv\Scripts\python retriever.py
```

**4. Run the full evaluation**

macOS / Linux:
```bash
cd src && ../.venv/bin/python run_eval.py
```

Windows (PowerShell):
```powershell
cd src; ..\.venv\Scripts\python run_eval.py
```

---

## Path to Production

This is a prototype. Putting NCR triage into a real controlled manufacturing environment would require, roughly in order of effort:

- **Real data** — actual historical NCRs and engineer-made dispositions, replacing synthetic ones.
- **Trust & correctness** — confidence thresholds, a **human-in-the-loop gate** (an AI does not disposition material unsupervised in aerospace), handling for weak retrieval, and continuously monitored accuracy.
- **Infrastructure** — authn/authz, audit logging (who recommended what, when), error handling, input validation, monitoring, index/model versioning.
- **Operational hardening** — graceful failure when the model or vector store is down, concurrency, embedding-model migration without invalidating stored vectors.
- **Regulatory/process** — AS9100 and configuration-management compliance, a validation strategy, formal sign-off, auditable controls.

Knowing this gap is the point: the prototype proves the reasoning loop; the list above is what stands between it and a system you'd trust on a real floor.

---

## About

Built by **Kyrylo Lynnyk** — laser production technician with a background in precision optical/fiber assembly and MRB nonconforming-material documentation, translating hands-on manufacturing-quality experience into applied AI.
