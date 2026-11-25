#!/usr/bin/env python3
"""
Local RAG assistant for product-knowledge questions in B2B sales.

Prototype for the BSc thesis "Local retrieval-augmented language models for
product-knowledge support in the B2B sales of industrial SMEs" (Onni Ovaskainen,
University of Vaasa).

Everything runs on this computer: Ollama serves the LLM and the embedding model,
Chroma stores the vectors on disk. No external API is called.

Usage (from this folder, with the virtual environment active):
    python rag.py ingest                      # read PDFs in docs/, build the vector store
    python rag.py ask "Which power supply options are available?"
    python rag.py test --questions evaluation/questions/nokeval_2021.csv --tag nokeval
    python rag.py test --no-rag --questions evaluation/questions/nokeval_2021.csv   # baseline

One manual per collection keeps the tests separate:
    python rag.py --collection nokeval ingest --docs docs/nokeval
    python rag.py --collection nokeval test --questions evaluation/questions/nokeval_2021.csv
"""
import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import chromadb
import pymupdf as fitz  # PyMuPDF
import ollama
from chromadb.config import Settings

# --------------------------------------------------------------------------
# Settings. Change them only on purpose and re-run ingest for chunk settings.
# Every test run also saves them automatically to results/run_info_*.json.
# --------------------------------------------------------------------------
LLM_MODEL = "gemma4:e4b"         # Gemma 4 E4B, open-weight (Apache 2.0), ~4B effective params
EMBED_MODEL = "nomic-embed-text"  # English embedding model
CHUNK_SIZE = 1000                 # characters per chunk
CHUNK_OVERLAP = 150               # characters shared between neighbouring chunks
MIN_CHUNK = 30                    # drop fragments shorter than this
TOP_K = 5                         # chunks retrieved per question
NUM_CTX = 8192                    # model context window (tokens)
TEMPERATURE = 0.0                 # deterministic answers for repeatable tests
SEED = 42

BASE = Path(__file__).resolve().parent
DB_DIR = BASE / "chroma_db"
RESULTS_DIR = BASE / "results"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are a product-knowledge assistant for a salesperson at an industrial "
    "manufacturer. Answer the question using ONLY the documentation excerpts given. "
    "If the excerpts do not contain the answer, reply exactly: "
    "\"The documentation provided does not answer this.\" Do not guess and do not "
    "use outside knowledge. Keep the answer short and factual. After each fact, "
    "cite its source in the form [file, p. N]."
)

NO_RAG_SYSTEM_PROMPT = (
    "You are a product-knowledge assistant for a salesperson at an industrial "
    "manufacturer. Answer the question briefly and factually. If you do not know, "
    "say so."
)


# ----------------------------- helpers ------------------------------------
def chroma_client():
    # Telemetry off, so the vector store makes no network calls at all.
    return chromadb.PersistentClient(
        path=str(DB_DIR), settings=Settings(anonymized_telemetry=False)
    )


def doc_prefix(embed_model):
    return "search_document: " if "nomic" in embed_model else ""


def query_prefix(embed_model):
    return "search_query: " if "nomic" in embed_model else ""


def clean(text):
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one page into overlapping chunks, preferring line or sentence breaks."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed(texts, embed_model, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        resp = ollama.embed(model=embed_model, input=texts[i:i + batch])
        out.extend(resp["embeddings"])
    return out


def chat(model, messages, options):
    """Call the LLM with thinking mode switched off (Gemma 4 and others can reason
    before answering, which would distort latency and token counts)."""
    try:
        resp = ollama.chat(model=model, messages=messages, options=options, think=False)
    except (TypeError, ollama.ResponseError) as e:
        # older ollama library (no think argument) or a model without thinking support
        if isinstance(e, ollama.ResponseError) and "think" not in str(e).lower():
            raise
        resp = ollama.chat(model=model, messages=messages, options=options)
    return resp


def strip_thoughts(text):
    """Safety net: remove any reasoning block that slips into the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<\|channel\|?>thought.*?<channel\|>", "", text, flags=re.S)
    return text.strip()


def ollama_version():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=5) as r:
            return json.loads(r.read()).get("version")
    except Exception:
        return None


def model_details(name):
    try:
        d = ollama.show(name)["details"]
        return {
            "family": d.get("family") if isinstance(d, dict) else getattr(d, "family", None),
            "parameter_size": d.get("parameter_size") if isinstance(d, dict) else getattr(d, "parameter_size", None),
            "quantization_level": d.get("quantization_level") if isinstance(d, dict) else getattr(d, "quantization_level", None),
        }
    except Exception as e:
        return {"error": str(e)}


def loaded_models():
    """Memory used by the models currently loaded in Ollama."""
    try:
        rows = []
        for m in ollama.ps()["models"]:
            g = m.get if isinstance(m, dict) else lambda k, m=m: getattr(m, k, None)
            rows.append({
                "model": g("model") or g("name"),
                "size_GB": round((g("size") or 0) / 1e9, 2),
                "size_on_gpu_GB": round((g("size_vram") or 0) / 1e9, 2),
            })
        return rows
    except Exception as e:
        return [{"error": str(e)}]


def machine_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "chromadb": chromadb.__version__,
        "pymupdf": getattr(fitz, "__version__", getattr(fitz, "VersionBind", None)),
    }
    if sys.platform == "darwin":
        for key, cmd in [("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                         ("ram_bytes", ["sysctl", "-n", "hw.memsize"]),
                         ("model", ["sysctl", "-n", "hw.model"])]:
            try:
                info[key] = subprocess.check_output(cmd, text=True).strip()
            except Exception:
                pass
        if "ram_bytes" in info:
            info["ram_GB"] = round(int(info["ram_bytes"]) / 2**30)
    return info


# ----------------------------- ingest -------------------------------------
def cmd_ingest(args):
    docs_dir = BASE / args.docs
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {docs_dir}. Put the documentation PDFs there first.")

    t0 = time.perf_counter()
    ids, texts, metas, report = [], [], [], []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        empty_pages, n_chunks, n_chars = [], 0, 0
        for pno in range(doc.page_count):
            text = clean(doc[pno].get_text("text"))
            if len(text) < MIN_CHUNK:
                empty_pages.append(pno + 1)
                continue
            n_chars += len(text)
            for ci, chunk in enumerate(split_text(text)):
                ids.append(f"{pdf.name}|p{pno + 1}|c{ci}")
                texts.append(chunk)
                metas.append({"file": pdf.name, "page": pno + 1})
                n_chunks += 1
        report.append({"file": pdf.name, "pages": doc.page_count, "characters": n_chars,
                       "chunks": n_chunks, "pages_without_text": empty_pages})
        doc.close()
    t_parse = time.perf_counter() - t0

    print(f"Parsed {len(pdfs)} PDFs into {len(texts)} chunks in {t_parse:.1f} s. Embedding...")
    t1 = time.perf_counter()
    vectors = embed([doc_prefix(args.embed) + t for t in texts], args.embed)
    t_embed = time.perf_counter() - t1

    client = chroma_client()
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    col = client.create_collection(
        args.collection,
        metadata={"hnsw:space": "cosine", "embed_model": args.embed,
                  "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
    )
    for i in range(0, len(ids), 500):
        col.add(ids=ids[i:i + 500], embeddings=vectors[i:i + 500],
                documents=texts[i:i + 500], metadatas=metas[i:i + 500])

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection, "docs_dir": str(args.docs),
        "embed_model": args.embed, "embed_model_details": model_details(args.embed),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "documents": report, "total_chunks": len(texts),
        "parse_seconds": round(t_parse, 1), "embed_seconds": round(t_embed, 1),
        "ollama_version": ollama_version(), "machine": machine_info(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"ingest_{args.collection}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Embedded in {t_embed:.1f} s. Stored in collection '{args.collection}'.")
    for r in report:
        flag = f"  (no text on pages {r['pages_without_text']})" if r["pages_without_text"] else ""
        print(f"  {r['file']}: {r['pages']} pages, {r['chunks']} chunks{flag}")
    print(f"Ingest log written to {out.relative_to(BASE)}")


# ----------------------------- retrieval + answer -------------------------
def get_collection(name, embed_model):
    col = chroma_client().get_collection(name)
    stored = (col.metadata or {}).get("embed_model")
    if stored and stored != embed_model:
        sys.exit(f"Collection '{name}' was built with '{stored}', not '{embed_model}'. "
                 f"Use --embed {stored} or re-ingest.")
    return col


def retrieve(col, question, embed_model, k=TOP_K):
    qv = embed([query_prefix(embed_model) + question], embed_model)[0]
    res = col.query(query_embeddings=[qv], n_results=k,
                    include=["documents", "metadatas", "distances"])
    return [{"text": d, "file": m["file"], "page": m["page"], "distance": round(dist, 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def generate(question, hits, model, use_rag=True):
    if use_rag:
        context = "\n\n".join(f"[{i + 1}] ({h['file']}, p. {h['page']})\n{h['text']}"
                              for i, h in enumerate(hits))
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"}]
    else:
        messages = [{"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
    resp = chat(model, messages,
                {"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "seed": SEED})
    answer = strip_thoughts(resp["message"]["content"])
    out_tokens = resp.get("eval_count") or 0
    eval_ns = resp.get("eval_duration") or 0
    return {
        "answer": answer,
        "prompt_tokens": resp.get("prompt_eval_count") or 0,
        "output_tokens": out_tokens,
        "tokens_per_s": round(out_tokens / (eval_ns / 1e9), 1) if eval_ns else None,
    }


def cmd_ask(args):
    col = get_collection(args.collection, args.embed)
    t0 = time.perf_counter()
    hits = retrieve(col, args.question, args.embed)
    t1 = time.perf_counter()
    g = generate(args.question, hits, args.model)
    t2 = time.perf_counter()
    print("\n" + g["answer"] + "\n")
    print("Retrieved excerpts:")
    for h in hits:
        print(f"  {h['file']} p. {h['page']}  (distance {h['distance']})")
    print(f"\nRetrieval {t1 - t0:.2f} s, generation {t2 - t1:.1f} s, "
          f"{g['output_tokens']} tokens at {g['tokens_per_s']} tok/s")


# ----------------------------- test run -----------------------------------
def parse_pages(s):
    pages = set()
    for part in re.split(r"[,;\s]+", str(s or "").strip()):
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                pages.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            pages.add(int(part))
    return pages


def cmd_test(args):
    qfile = BASE / args.questions
    with open(qfile, newline="", encoding="utf-8") as f:
        questions = [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]
    if not questions:
        sys.exit(f"No questions in {qfile}.")

    use_rag = not args.no_rag
    col = get_collection(args.collection, args.embed) if use_rag else None
    mode = "rag" if use_rag else "norag"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / f"results_{mode}{tag}_{stamp}.csv"

    # Warm-up so that loading the model into memory is not counted in question 1.
    t_load = time.perf_counter()
    chat(args.model, [{"role": "user", "content": "Hi"}],
         {"num_ctx": NUM_CTX, "num_predict": 1})
    load_seconds = round(time.perf_counter() - t_load, 1)
    print(f"Model loaded/warmed up in {load_seconds} s. Running {len(questions)} questions ({mode})...\n")

    fields = ["id", "category", "question", "expected_answer", "expected_file", "expected_page",
              "answer", "retrieved_sources", "expected_page_retrieved", "top1_distance",
              "retrieval_s", "generation_s", "total_s", "prompt_tokens", "output_tokens",
              "tokens_per_s", "verdict", "source_correct", "failure_mode", "notes"]
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retrieve(col, q["question"], args.embed) if use_rag else []
        t1 = time.perf_counter()
        g = generate(q["question"], hits, args.model, use_rag)
        t2 = time.perf_counter()

        exp_pages = parse_pages(q.get("expected_page"))
        exp_file = (q.get("expected_file") or "").strip().lower()
        if use_rag and exp_pages:
            found = any(h["page"] in exp_pages and (not exp_file or exp_file in h["file"].lower())
                        for h in hits)
            page_hit = "yes" if found else "no"
        else:
            page_hit = "n/a"

        row = {k: q.get(k, "") for k in ["id", "category", "question", "expected_answer",
                                          "expected_file", "expected_page"]}
        row.update({
            "answer": g["answer"],
            "retrieved_sources": "; ".join(f"{h['file']} p.{h['page']}" for h in hits),
            "expected_page_retrieved": page_hit,
            "top1_distance": hits[0]["distance"] if hits else "",
            "retrieval_s": round(t1 - t0, 2), "generation_s": round(t2 - t1, 1),
            "total_s": round(t2 - t0, 1),
            "prompt_tokens": g["prompt_tokens"], "output_tokens": g["output_tokens"],
            "tokens_per_s": g["tokens_per_s"],
            "verdict": "", "source_correct": "", "failure_mode": "", "notes": "",
        })
        rows.append(row)
        print(f"[{q.get('id')}] {row['total_s']} s  retrieved expected page: {page_hit}")
        print(f"    Q: {q['question']}")
        print(f"    A: {g['answer'][:300]}{'...' if len(g['answer']) > 300 else ''}\n")

    # utf-8-sig so that Excel/Numbers open ä, ö, é etc. correctly
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    totals = [r["total_s"] for r in rows]
    hits_known = [r for r in rows if r["expected_page_retrieved"] in ("yes", "no")]
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "tag": args.tag, "questions_file": args.questions,
        "llm_model": args.model, "llm_details": model_details(args.model), "thinking": "off",
        "embed_model": args.embed if use_rag else None,
        "embed_details": model_details(args.embed) if use_rag else None,
        "collection": args.collection if use_rag else None,
        "settings": {"top_k": TOP_K, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                     "num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": SEED},
        "system_prompt": SYSTEM_PROMPT if use_rag else NO_RAG_SYSTEM_PROMPT,
        "model_load_seconds": load_seconds,
        "n_questions": len(rows),
        "mean_total_s": round(sum(totals) / len(totals), 1),
        "max_total_s": max(totals),
        "expected_page_retrieved": f"{sum(r['expected_page_retrieved'] == 'yes' for r in hits_known)}/{len(hits_known)}",
        "models_in_memory": loaded_models(),
        "ollama_version": ollama_version(),
        "machine": machine_info(),
        "results_file": out_csv.name,
    }
    info_path = RESULTS_DIR / f"run_info_{mode}{tag}_{stamp}.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"Done. Mean {info['mean_total_s']} s per question, "
          f"expected page retrieved {info['expected_page_retrieved']}.")
    print(f"Results:  {out_csv.relative_to(BASE)}  (fill in verdict, source_correct, failure_mode)")
    print(f"Run info: {info_path.relative_to(BASE)}")


# ----------------------------- CLI ----------------------------------------
def main():
    p = argparse.ArgumentParser(description="Local RAG assistant for product-knowledge questions")
    p.add_argument("--model", default=LLM_MODEL, help="Ollama LLM (default %(default)s)")
    p.add_argument("--embed", default=EMBED_MODEL, help="Ollama embedding model (default %(default)s)")
    p.add_argument("--collection", default="product_docs", help="vector store collection name")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="read PDFs and build the vector store")
    pi.add_argument("--docs", default="docs", help="folder with the PDFs (default docs)")

    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("question")

    pt = sub.add_parser("test", help="run all questions in questions.csv")
    pt.add_argument("--questions", default="questions.csv")
    pt.add_argument("--no-rag", action="store_true", help="baseline: model answers without documents")
    pt.add_argument("--tag", default="", help="label added to result file names, e.g. fr")

    args = p.parse_args()
    try:
        {"ingest": cmd_ingest, "ask": cmd_ask, "test": cmd_test}[args.cmd](args)
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e}. Is the model pulled? Try: ollama pull {args.model} / ollama pull {args.embed}")
    except (ConnectionError, OSError) as e:
        if "Connection" in type(e).__name__ or "refused" in str(e).lower():
            sys.exit("Cannot reach Ollama on localhost:11434. Start the Ollama app first.")
        raise


if __name__ == "__main__":
    main()
