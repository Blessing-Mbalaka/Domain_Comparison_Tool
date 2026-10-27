# Research Atlas — semantic research dashboard

This project upgrades the verified static semantic heatmaps into a persistent FastAPI + React research workspace.

The main Stage Alignment workspace has three modes: **Bi-compare** places research stages on the Y-axis and project-management stages on the X-axis; **Self-compare** compares a lifecycle with itself; and **Corpus evidence** retains the document-by-stage view. Stage cells expose model alignment, corpus co-support, joint exact presence, evidence passages, relevant terms and practical PM-to-research recommendations.

On first run the database also receives six papers labelled `Demo — ...` with source `synthetic demo presence corpus`. They provide transparent, realistic-looking presence, term-count and co-occurrence examples for previewing the interface; they are never represented as uploaded or empirical evidence and can be deleted like ordinary papers.

## Run

From this folder in PowerShell:

```powershell
.\start.ps1
```

Open <http://127.0.0.1:9001>. Data is stored in `data/dashboard.sqlite3`; uploaded source files are kept in `data/uploads`. The database is seeded only when first created, so edits and deletions survive restarts.

Each new upload must be assigned to an existing analytical stage. The dashboard keeps exact phrase presence as a separate audit signal and also calculates concept presence: a document counts as present when its assigned stage matches, the stage name appears exactly, a configured synonym appears, or the selected model reaches at least 0.80 cosine similarity. Corpus presence is the proportion of selected documents meeting that rule and is recalculated incrementally after uploads or edits.

Single and bulk ingestion are supported. A bulk batch accepts up to 100 PDF, DOCX or TXT files, uses each filename as the paper title, applies one primary stage and an optional comparative stage to the batch, and processes the papers through one restart-safe background job. Corpus Analytics presents raw corpus and per-document frequency, plus a semantic top-100 explorer ranked against a selected stage's label, description and synonyms. The explorer supports corpus or document scope, combined or individual models, an adjustable similarity threshold, and automatic removal of standard and corpus-specific stop words. The page also reports document and vocabulary totals, stage-presence rates, and semantic, coverage, concept-presence and exact-presence metrics for every model.

## Development

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 9001
```

The four names from the legacy export are preserved as selectable analytical projections. Original stage-to-stage scores are imported without modification. New uploads use deterministic, cached local feature-hash vectors so the application works without downloading large neural model checkpoints.

## Deploy tonight

The repository includes a production `Dockerfile` and a Render Blueprint. Push it to GitHub, create a new Render Blueprint from the repository, and Render will provision the web service plus a persistent 1 GB disk. The health check is `/api/health`.

SQLite and uploaded files require the included persistent disk. Do not deploy this configuration to an ephemeral filesystem without attaching a volume.
