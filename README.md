# Research Atlas — semantic research dashboard

This project upgrades the verified static semantic heatmaps into a persistent FastAPI + React research workspace.

The main Stage Alignment workspace has three modes: **Bi-compare** places research stages on the Y-axis and project-management stages on the X-axis; **Self-compare** compares a lifecycle with itself; and **Corpus evidence** retains the document-by-stage view. Stage cells expose model alignment, corpus co-support, joint exact presence, evidence passages, relevant terms and practical PM-to-research recommendations.

## Run

From this folder in PowerShell:

```powershell
.\start.ps1
```

Open <http://127.0.0.1:9001>. Data is stored in `data/dashboard.sqlite3`; uploaded source files are kept in `data/uploads`. The database is seeded only when first created, so edits and deletions survive restarts.

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
