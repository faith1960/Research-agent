#!/usr/bin/env python3
"""
Backend RAG per Cervello Federico.
Porta 8002 — interfaccia web in italiano.
"""

import json
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
OLLAMA_BASE = "http://192.168.178.145:11434"
QDRANT_HOST = "192.168.178.150"
QDRANT_PORT = 6333
COLLECTION  = "cervello_federico"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "gemma3:4b"
TOP_K       = 5
PORT        = 8002
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app       = FastAPI(title="Cervello Federico")
qdrant    = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def search_docs(query: str) -> list[dict]:
    embedding = get_embedding(query)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=embedding,
        limit=TOP_K,
        with_payload=True,
    )
    return [
        {
            "testo":    h.payload["testo"],
            "sorgente": h.payload["sorgente"],
            "score":    round(h.score, 3),
        }
        for h in hits
    ]


def build_prompt(domanda: str, docs: list[dict]) -> str:
    contesto = "\n\n---\n\n".join(
        f"[{d['sorgente']}]\n{d['testo']}" for d in docs
    )
    return (
        "Sei un assistente personale che risponde sempre in italiano.\n"
        "Usa il contesto estratto dai documenti personali per rispondere.\n"
        "Se il contesto non è sufficiente, dillo chiaramente senza inventare.\n\n"
        f"Contesto:\n{contesto}\n\n"
        f"Domanda: {domanda}\n\n"
        "Risposta:"
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stato")
async def stato():
    try:
        info = qdrant.get_collection(COLLECTION)
        return {
            "collection":  COLLECTION,
            "punti":       info.points_count,
            "dimensione":  info.config.params.vectors.size,
            "ollama":      OLLAMA_BASE,
            "modello_llm": LLM_MODEL,
            "embed":       EMBED_MODEL,
        }
    except Exception as exc:
        return JSONResponse({"errore": str(exc)}, status_code=503)


@app.post("/domanda")
async def domanda(request: Request):
    """
    Riceve {"domanda": "..."} e restituisce una stream SSE con gli eventi:
      {"tipo":"sorgenti","sorgenti":[...]}
      {"tipo":"token","contenuto":"..."}
      {"tipo":"fine"}
      {"tipo":"errore","messaggio":"..."}
    """
    data  = await request.json()
    query = data.get("domanda", "").strip()

    if not query:
        return JSONResponse({"errore": "Domanda vuota"}, status_code=400)

    async def stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 1. Recupera documenti rilevanti
        try:
            docs = search_docs(query)
        except Exception as exc:
            yield sse({"tipo": "errore", "messaggio": f"Ricerca fallita: {exc}"})
            return

        if not docs:
            yield sse({"tipo": "errore", "messaggio": "Nessun documento trovato nella knowledge base."})
            return

        # 2. Invia le sorgenti subito
        sorgenti = list(dict.fromkeys(d["sorgente"] for d in docs))  # ordine preservato, no duplicati
        yield sse({"tipo": "sorgenti", "sorgenti": sorgenti})

        # 3. Genera risposta in streaming
        prompt = build_prompt(query, docs)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("response"):
                            yield sse({"tipo": "token", "contenuto": chunk["response"]})
                        if chunk.get("done"):
                            break
        except Exception as exc:
            yield sse({"tipo": "errore", "messaggio": f"Generazione fallita: {exc}"})
            return

        yield sse({"tipo": "fine"})

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Avvio
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
