"""RAG retriever for past NCRs.

Embeds each NCR description with a local Ollama embedding model (nomic-embed-text,
via llm.embed) and stores them in a persistent Chroma vector store at data/chroma/.
build_index() populates it once; retrieve(query, k) returns the k most similar past
NCRs with metadata and a cosine similarity score.
"""
import json
from pathlib import Path
import chromadb
from llm import embed, OLLAMA_EMBED_MODEL

ROOT = Path(__file__).resolve().parent.parent
NCR_FILE = ROOT / "data" / "ncrs.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "ncrs"
# Metadata we keep per NCR. ncr_id/defect_type/part_name/process/date exist in the
# synthetic data today; disposition/root_cause do not yet (the generator doesn't emit
# them) — they fall back to "unknown" until the data carries them.
META_FIELDS = ["ncr_id", "defect_type", "disposition", "root_cause", "part_name", "process", "date"]


def _load_ncrs():
    records = []
    with open(NCR_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _collection(client=None):
    client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
    # cosine space so distance maps cleanly to a [0,1]-ish similarity (1 - distance)
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def _metadata(rec):
    # Chroma metadata values must be scalars and non-null -> default missing fields to "unknown".
    return {k: (rec.get(k) if rec.get(k) is not None else "unknown") for k in META_FIELDS}


def build_index(force=False):
    """Embed every NCR description and (re)populate the Chroma collection. Idempotent:
    skips work when the store already holds exactly len(data) vectors unless force=True."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    records = _load_ncrs()
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    if not force and col.count() == len(records):
        return col
    # stale/partial/forced -> wipe and rebuild clean
    client.delete_collection(COLLECTION)
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    ids = [r["ncr_id"] for r in records]
    docs = [r["description"] for r in records]
    metas = [_metadata(r) for r in records]
    embs = [embed(d) for d in docs]
    B = 100
    for i in range(0, len(ids), B):
        col.add(ids=ids[i:i + B], documents=docs[i:i + B], metadatas=metas[i:i + B], embeddings=embs[i:i + B])
    return col


def retrieve(query, k=3):
    """Return the k most similar past NCRs to `query` as a list of dicts:
    {ncr_id, description, similarity, metadata}, ordered most-similar-first."""
    col = _collection()
    res = col.query(query_embeddings=[embed(query)], n_results=k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "ncr_id": meta.get("ncr_id"),
            "description": doc,
            "similarity": round(1.0 - dist, 4),  # cosine distance -> similarity
            "metadata": meta,
        })
    return hits


if __name__ == "__main__":
    n = len(_load_ncrs())
    col = _collection()
    if col.count() != n:
        print(f"Building index: embedding {n} NCRs with {OLLAMA_EMBED_MODEL} (store: {CHROMA_DIR})...")
        build_index()
        print("Done.\n")
    else:
        print(f"Index ready: {col.count()} NCRs in {CHROMA_DIR}\n")

    query = "Connector failing optical insertion loss test; measured loss too high, needs re-termination."
    print(f"Query: {query!r}\n")
    for i, hit in enumerate(retrieve(query, k=3), 1):
        m = hit["metadata"]
        print(f"[{i}] {hit['ncr_id']}  similarity={hit['similarity']}")
        print(f"    defect_type : {m.get('defect_type')!r}")
        print(f"    disposition : {m.get('disposition')!r}   root_cause: {m.get('root_cause')!r}")
        print(f"    part/process: {m.get('part_name')!r} / {m.get('process')!r}")
        print(f"    description : {hit['description']}")
