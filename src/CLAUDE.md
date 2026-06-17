# CLAUDE.md — NCR Triage Agent

Context for Claude Code. Read this first.

## What this project is
An agentic AI system that triages **Nonconforming Material Reports (NCRs)** end-to-end:
classify the defect → retrieve similar past NCRs (RAG) → reason about root cause →
recommend a disposition (use-as-is / rework / repair / scrap) with cited evidence →
draft the writeup → **flag for human approval**.

Built as a portfolio piece tied to the owner's manufacturing/quality background (laser/optical/
fiber assembly at SpaceX). Also the build vehicle for the IBM RAG & Agentic AI cert.

## Hard rules
- **Synthetic/public data only.** Never use or invent real proprietary employer data.
- **Human-in-the-loop:** the agent recommends; a human approves. Never auto-finalize a disposition.
- **Structured outputs:** model calls return validated JSON, never trusted free text.
- **Measure everything:** changes are validated by re-running the eval, not by vibes.

## Owner preferences
- Direct, concise, no corporate fluff. Show diffs before committing.
- Currently building model-agnostic; default backend is **local Ollama (llama3.1)** — free, no API key.
  Swappable via env vars in `src/llm.py` (LLM_BACKEND / OLLAMA_MODEL / OLLAMA_HOST).

## Current state (as of Jun 16, 2026)
- ✅ Phase 0: synthetic data generator + repo scaffold
- ✅ Defect classifier returning validated JSON {defect_type, severity, confidence}
- ✅ Eval harness — defect_type **20/20 (100%)** after the prompt-tuning pass; plus an end-to-end
  **disposition RAG eval** (**11/20 = 55%**, beating majority-class & neighbor-vote baselines at 40%).
  Leakage-controlled: the eval NCR's own record *and* any byte-identical-description NCR are excluded
  from the retrieved neighbors (eval inputs are exact corpus text and descriptions repeat across ids).
- ✅ Prompt-tuning pass — fixed the ambiguous fiber miss ("Core offset suspected; IL 0.6 dB vs
  0.5 dB target, re-termination needed." was classed Dimensional) by adding defect-keyword hints to
  the SYSTEM prompt in `src/classifier.py` (incl. the "optical loss in dB is ALWAYS Fiber" rule).
- ✅ RAG retriever — past NCRs embedded into a persistent Chroma store (`data/chroma/`); NCRs now
  carry MRB `disposition` + `root_cause`, so retrieved cases show real decision history.

## Defect types (the fixed label set — keep these exact strings everywhere)
- Surface Scratch/Dig
- Fiber Alignment / Insertion Loss
- Dimensional Out-of-Tolerance
- Solder / Bond Defect
- Contamination / Coating Defect

## Layout
```
src/generate_ncrs.py   synthetic NCR generator (stdlib only; emits disposition + root_cause)
src/llm.py             model-agnostic completion + embedding wrapper (Ollama default)
src/classifier.py      NCR description -> validated JSON classification
src/retriever.py       RAG retriever: embeds NCRs into Chroma, retrieves similar past cases
src/run_eval.py        defect_type eval (vs eval_set.json) + end-to-end disposition RAG eval w/ baselines
data/ncrs.jsonl|csv    200 synthetic NCRs (the agent's "history")
data/chroma/           persistent Chroma vector store (built by retriever.build_index)
eval/eval_set.json     20 held-out labeled cases
.venv/                 virtualenv for the RAG deps (chromadb); stdlib-only scripts don't need it
```

## Roadmap (next steps in order)
1. Finish the prompt-tuning pass (lock in or revert based on the eval number).
2. **RAG retriever** — embed past NCRs into a vector store (Chroma first, then FAISS),
   retrieve similar cases for a new NCR.
3. Measurement-check tool — exact pass/fail math done in code, not by the LLM.
4. Turn the pieces into an agent (tool calling), then a LangGraph state machine with the
   human-approval gate.
5. Wrap tools as an MCP server. Add tracing. Write README. Deploy (HF Spaces or Railway).

## How to run
```
# stdlib only — system python is fine:
python3 src/generate_ncrs.py   # regenerate data + eval set (incl. disposition + root_cause)
python3 src/classifier.py      # classify one sample

# RAG pieces — need the .venv (chromadb); run from src/ so `import llm`/`import retriever` resolve:
cd src && ../.venv/bin/python retriever.py   # build index if stale, then run a sample query
cd src && ../.venv/bin/python run_eval.py    # defect_type eval + end-to-end disposition RAG eval
#   EVAL_K env var sets the neighbor count (default 3). Plain `python3 src/run_eval.py` still runs the
#   defect eval and gracefully skips the disposition section when chromadb isn't installed.
```
Requires Ollama running locally with: `ollama pull llama3.1 && ollama pull nomic-embed-text`
(llama3.1 = classifier/reasoning, nomic-embed-text = retriever embeddings).