Kurze Erklärung (Dokumentation folgt dann zur gegebenen Zeit)

Für Deployment/Automatisierung ist die **Python-Pipeline** der bevorzugte Weg:

- Pipeline Runner: `scripts/gold/pipeline.py` (führt alle Schritte in richtiger Reihenfolge aus)
- Notebooks (`scripts/**.ipynb`) sind nur noch für Entwicklung/Exploration gedacht und können vom `.py` Stand abweichen.

Windows (mit venv):

- `\.venv\Scripts\python.exe scripts\gold\pipeline.py`

Die Dateipfade sind aktuell (meine ich) für Windows angegeben. Für Linux muss man die "\\" in der .env Datei ändern.
