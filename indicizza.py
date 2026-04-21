#!/usr/bin/env python3
"""
Indicizza i documenti in /Users/federico/cervello-federico/sorgenti/
verso Qdrant. Supporta PDF, MD e TXT.

Uso:
  python indicizza.py            # aggiunta/aggiornamento incrementale
  python indicizza.py --forza    # svuota la collection e reindicizza tutto
"""

import argparse
import hashlib
import sys
from pathlib import Path

import httpx
import pymupdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
OLLAMA_BASE   = "http://192.168.178.145:11434"
QDRANT_HOST   = "192.168.178.150"
QDRANT_PORT   = 6333
COLLECTION    = "cervello_federico"
EMBED_MODEL   = "nomic-embed-text"
DOCS_PATH     = Path("/Users/federico/cervello-federico/sorgenti")
EMBED_DIM     = 768   # nomic-embed-text
CHUNK_CHARS   = 500
OVERLAP_CHARS = 50
EXTENSIONS    = {".pdf", ".md", ".markdown", ".txt"}
# ---------------------------------------------------------------------------


def get_embedding(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = pymupdf.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_CHARS].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


def make_point_id(path: Path, chunk_idx: int) -> int:
    digest = hashlib.md5(f"{path}:{chunk_idx}".encode()).hexdigest()
    return int(digest[:15], 16)  # 60-bit int, safe for Qdrant


def ensure_collection(client: QdrantClient, forza: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if forza and COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' eliminata per reindicizzazione completa.")
        existing.discard(COLLECTION)
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION}' creata ({EMBED_DIM}d, cosine).")
    else:
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' esistente — {info.points_count} punti presenti.")


def index_file(client: QdrantClient, path: Path) -> int:
    try:
        text = extract_text(path)
    except Exception as exc:
        print(f"  [ERRORE estrazione] {path.name}: {exc}")
        return 0

    chunks = chunk_text(text)
    if not chunks:
        print(f"  [saltato, vuoto] {path.name}")
        return 0

    points: list[PointStruct] = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
        except Exception as exc:
            print(f"  [ERRORE embedding] chunk {i} di {path.name}: {exc}")
            continue
        points.append(
            PointStruct(
                id=make_point_id(path, i),
                vector=embedding,
                payload={
                    "testo":          chunk,
                    "sorgente":       path.name,
                    "percorso":       str(path),
                    "chunk":          i,
                    "totale_chunks":  len(chunks),
                },
            )
        )

    if points:
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indicizzatore documenti personali")
    parser.add_argument(
        "--forza", action="store_true",
        help="Svuota la collection e reindicizza tutti i documenti da zero"
    )
    args = parser.parse_args()

    if not DOCS_PATH.exists():
        sys.exit(f"Cartella sorgenti non trovata: {DOCS_PATH}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, args.forza)

    files = sorted(p for p in DOCS_PATH.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f"Nessun documento trovato in {DOCS_PATH}")

    print(f"\nDocumenti trovati: {len(files)}\n")
    total_chunks = 0
    errors = 0

    for i, path in enumerate(files, 1):
        print(f"[{i:>3}/{len(files)}] {path.name} ...", end=" ", flush=True)
        n = index_file(client, path)
        if n:
            print(f"{n} chunk")
            total_chunks += n
        else:
            errors += 1

    print(f"\n{'─'*50}")
    print(f"Completato.  Chunk indicizzati: {total_chunks}   Errori: {errors}")
    info = client.get_collection(COLLECTION)
    print(f"Totale punti in Qdrant: {info.points_count}")


if __name__ == "__main__":
    main()
