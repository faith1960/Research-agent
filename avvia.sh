#!/usr/bin/env bash
# Avvia il backend RAG Cervello Federico.
# Da usare direttamente o tramite launchd.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Crea l'ambiente virtuale la prima volta
if [ ! -d "venv" ]; then
  echo "[avvia.sh] Creo ambiente virtuale..."
  python3 -m venv venv
fi

# Installa/aggiorna le dipendenze
echo "[avvia.sh] Verifico dipendenze..."
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt

# Crea la cartella log se non esiste
mkdir -p logs

echo "[avvia.sh] Avvio app.py sulla porta 8002..."
exec venv/bin/python app.py
